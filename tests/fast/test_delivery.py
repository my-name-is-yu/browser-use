"""Structured delivery must preserve evidence, missing fields and recovery accounting."""

import json
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from benchmarks.bu_fast import measure_goal_loop as measure
from browser_use.fast.delivery import CompleteDelivery, ResultRecord, save_delivery, validate_delivery
from tests.fast.test_goal_loop import Choices, runtime


def delivery(
	disposition: Literal['complete', 'continue', 'incomplete'] = 'complete',
	value: str | None = 'Applied',
	quote: str | None = 'Applied result',
):
	return CompleteDelivery(
		disposition=disposition,
		browser_summary='Observed applied result',
		records=[
			ResultRecord(id='result', value=value, quote=quote, source_url='https://example.test/' if value is not None else None)
		],
		issues=[] if value else ['Missing result'],
		final_answer='Applied [source](https://example.test/)',
	)


def evidence():
	return {
		'id': 'o',
		'url': 'https://example.test/',
		'title': 'Page',
		'text': 'Applied result',
		'fragment_text': '',
		'headings': [],
		'links': [],
		'targets': [],
		'text_truncated': False,
	}


def test_provenance_missing_and_artifacts(tmp_path):
	data = delivery()
	validate_delivery(data, ['result'], [measure.evidence_item(evidence())])
	receipt = save_delivery(data, tmp_path)
	assert receipt.files == ['results.json', 'report.md']
	assert json.loads((tmp_path / 'results.json').read_text())[0]['quote'] == 'Applied result'
	assert '[source](https://example.test/)' in (tmp_path / 'report.md').read_text()
	assert (tmp_path / 'report.md').read_text().endswith(data.final_answer + '\n')
	with pytest.raises(ValueError, match='Quote/source'):
		validate_delivery(delivery(quote='Invented'), ['result'], [evidence()])
	with pytest.raises(ValueError, match='Missing field'):
		validate_delivery(delivery(value=None), ['result'], [])
	missing = delivery('incomplete', None, None)
	validate_delivery(missing, ['result'], [])
	save_delivery(missing, tmp_path)
	assert json.loads((tmp_path / 'results.json').read_text())[0] == {
		'id': 'result',
		'value': None,
		'quote': None,
		'source_url': None,
	}


def test_continue_needs_no_final_coverage_or_provenance():
	partial = delivery('continue', quote='Not fetched yet')
	partial.records = []
	partial.final_answer = ''
	partial.issues = ['Need the changed state observation']
	validate_delivery(partial, ['baseline', 'changed'], [])
	partial.records = delivery('continue', quote='Not fetched yet').records
	validate_delivery(partial, ['baseline', 'changed'], [])


def test_quote_fragment_and_whitespace_keep_observed_url_identity():
	data = delivery(quote='Applied\n  result')
	data.records[0].source_url = 'https://example.test/#observed-section'
	validate_delivery(data, ['result'], [measure.evidence_item(evidence())])
	data.records[0].source_url = 'https://example.test/?different=1#observed-section'
	with pytest.raises(ValueError, match='Quote/source'):
		validate_delivery(data, ['result'], [measure.evidence_item(evidence())])
	data.records[0].source_url = 'https://other.test/#observed-section'
	with pytest.raises(ValueError, match='Quote/source'):
		validate_delivery(data, ['result'], [measure.evidence_item(evidence())])
	linked = measure.evidence_item(evidence())
	linked['links'] = [{'href': 'https://other.test/'}]
	validate_delivery(data, ['result'], [linked])


async def test_done_repair_keeps_goal_identity_deadline_and_budget(tmp_path):
	r = runtime(tmp_path, Choices('every requested', 'every requested'))
	first = await r.run(measure.BrowserTask(url='https://example.test/', goal='Observe the applied result'))
	assert first.status == 'done'
	final = await r.run(resume=first, repair='The submitted result was not observed')
	assert final.run_id == first.run_id and final.task == first.task
	assert final.budget.deadline_at == first.budget.deadline_at
	assert final.budget.decisions_used == first.budget.decisions_used + 1
	assert r.driver.page.goto.await_count == 1
	assert r.decisions.calls[-1][0]['completion_deficiency'] == 'The submitted result was not observed'
	final.budget.max_decisions = final.budget.decisions_used
	blocked = await r.run(resume=final, repair='Still missing source')
	assert blocked.status == 'blocked' and blocked.reason == 'decision_budget_exhausted'
	assert len(r.decisions.calls) == 2


@pytest.mark.parametrize(
	'scenario, expected_calls', [('normal', 2), ('missing', 4), ('repair', 4), ('incomplete', 2), ('timeout', 0), ('error', 0)]
)
async def test_mock_trial_delivery_and_recovery(tmp_path, monkeypatch, scenario, expected_calls):
	monkeypatch.setenv('OPENAI_API_KEY', 'mock-key')
	browser = SimpleNamespace(start=AsyncMock(), kill=AsyncMock(), agent_focus_target_id='tab')
	monkeypatch.setattr(measure, 'BrowserSession', lambda **kw: browser)
	monkeypatch.setattr(measure, 'Page', lambda *a: object())
	monkeypatch.setattr(measure, 'BrowserDriver', lambda *a: object())
	monkeypatch.setattr(measure, 'DecisionClient', lambda **kw: SimpleNamespace(close=AsyncMock()))
	runs = []

	class Runtime:
		def __init__(self, *args, **options):
			self.options = options

		async def run(self, task, **options):
			runs.append((task, options))
			await self.options['on_event']({'event': 'observed', 'observation': evidence()})
			if scenario in ('timeout', 'error'):
				raise TimeoutError('mock timeout') if scenario == 'timeout' else RuntimeError('mock error')
			return SimpleNamespace(status='done', reason='model_done', summary=lambda: {'status': 'done'})

	monkeypatch.setattr(measure, 'DecisionRuntime', Runtime)

	async def capture(page, trace, action, result, *a, **kw):
		trace.agent_steps.append(json.dumps({'action': action, 'result': result}))

	monkeypatch.setattr(measure.common, 'capture_step', capture)

	class Item:
		type = 'function_call'

		def __init__(self, name, arguments):
			self.name, self.arguments, self.call_id = name, json.dumps(arguments), name

		def model_dump(self, **kw):
			return vars(self) | {'type': self.type}

	steps = [[Item('read_evidence', {'ids': [0]})]]
	if scenario == 'missing':
		steps += [[Item('complete_delivery', delivery(quote='Invented').model_dump())], [Item('read_evidence', {'ids': [0]})]]
	if scenario == 'repair':
		steps += [
			[Item('repair_browser', {'deficiency': 'Search was only typed, not applied'})],
			[Item('read_evidence', {'ids': [1]})],
		]
	data = delivery('incomplete', None, None) if scenario == 'incomplete' else delivery()
	steps += [[Item('complete_delivery', data.model_dump())]]
	inputs = []

	async def respond(**options):
		inputs.append(json.loads(json.dumps(options)))
		return SimpleNamespace(status='completed', output=steps.pop(0), output_text='', model_dump_json=lambda: '{}')

	monkeypatch.setattr(
		measure, 'AsyncOpenAI', lambda **kw: SimpleNamespace(responses=SimpleNamespace(create=respond), close=AsyncMock())
	)
	directory = tmp_path / 'trial'
	result = await measure.execute_trial(
		{
			'id': 'test',
			'goal': 'Observe applied result',
			'start_url': 'https://example.test/',
			'exact_input_values': [],
			'required_fields': [{'id': 'result'}],
		},
		'goal',
		0,
		directory,
		measure.ExperimentSettings(variant='structured-delivery'),
	)
	if scenario in ('timeout', 'error'):
		assert result.status == scenario and result.decisions_loop_seconds > 0
		assert result.parent_calls == 0
		assert (directory / 'delegation-trace.json').exists()
		assert (directory / 'final-delegation-trace.json').exists()
		return
	assert result.status == 'finished', result.error
	assert result.parent_calls == expected_calls and not steps
	assert result.evidence_calls == (2 if scenario in ('repair', 'missing') else 1)
	assert result.evidence_bytes > 0 and result.evidence_tokens is None
	assert result.delivery_seconds is not None
	assert result.decisions_loop_seconds > 0 and result.delivery_seconds >= result.decisions_loop_seconds
	assert result.repair_calls == (1 if scenario == 'repair' else 0)
	assert len(runs) == (2 if scenario == 'repair' else 1)
	initial_trace = measure.Trace.model_validate_json((directory / 'delegation-trace.json').read_text())
	final_delegate_trace = measure.Trace.model_validate_json((directory / 'final-delegation-trace.json').read_text())
	assert len(initial_trace.agent_steps) == 1
	assert len(final_delegate_trace.agent_steps) == len(runs)
	assert (directory / 'final.txt').read_text() == data.final_answer
	trace = measure.Trace.model_validate_json((directory / 'trace.json').read_text())
	assert 'results.json' in trace.output_files_text and 'report.md' in trace.output_files_text
	exchanges = [json.loads(line) for line in (directory / 'tool-exchanges.jsonl').read_text().splitlines()]
	assert len(exchanges) == expected_calls
	assert exchanges[-1]['name'] == 'complete_delivery'
	assert exchanges[1]['status'] == ('error' if scenario == 'missing' else 'completed')
	assert all(options['model'] == 'gpt-6.1-sol' and options['reasoning'] == {'effort': 'xhigh'} for options in inputs)
	assert 'Applied result' not in inputs[0]['input'][2]['content']
	assert 'Applied result' in inputs[1]['input'][-2]['output']
	assert all(tool['name'] not in ('finish_browser', 'write_artifact') for tool in inputs[0]['tools'])


def test_lossless_projection_resolves_all_references():
	trace = measure.Trace(
		screenshots_b64=[],
		screenshot_steps=[],
		final_result='answer',
		agent_steps=[json.dumps({'text': 'x' * 900, 'inputs': ['state'] * 100})] * 3,
	)
	projected = measure.compact_trace(trace)
	assert len('\n'.join(projected.agent_steps)) < len('\n'.join(trace.agent_steps))
	assert [json.loads(step) for step in measure.restore_trace(projected).agent_steps] == [
		json.loads(step) for step in trace.agent_steps
	]
