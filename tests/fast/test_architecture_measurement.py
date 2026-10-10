"""Experiment wiring must activate additions and deliver their actual results."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.bu_fast import measure_architectures as plan
from benchmarks.bu_fast import measure_goal_loop as measure
from browser_use.fast.views import Contract


class TextAdvice(Contract):
	text: str


@pytest.mark.parametrize('variant', plan.VARIANTS)
async def test_trial_routes_feature_and_preserves_preparation(tmp_path, monkeypatch, variant):
	calls = []
	parent_inputs = []
	modules = {
		'browser_use.fast.views': SimpleNamespace(Advice=TextAdvice),
		'browser_use.fast.experience': SimpleNamespace(load_memos=lambda store, ids: {'ids': ids}),
		'browser_use.fast.parallel': SimpleNamespace(APIParentReader=lambda client: client, ParallelEvidence=lambda *a, **kw: kw),
	}
	original_import = measure.importlib.import_module
	monkeypatch.setattr(
		measure.importlib, 'import_module', lambda name: modules[name] if name in modules else original_import(name)
	)
	monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
	browser = SimpleNamespace(start=AsyncMock(), kill=AsyncMock(), agent_focus_target_id='tab')
	monkeypatch.setattr(measure, 'BrowserSession', lambda **kw: browser)
	monkeypatch.setattr(measure, 'Page', lambda *a: object())
	monkeypatch.setattr(measure, 'BrowserDriver', lambda *a: object())
	monkeypatch.setattr(measure, 'DecisionClient', lambda **kw: SimpleNamespace(close=AsyncMock()))

	async def capture(page, trace, action, result, *args, **kw):
		trace.agent_steps.append(json.dumps({'action': action, 'result': result}))

	monkeypatch.setattr(measure.common, 'capture_step', capture)

	class Saved:
		directory = str(tmp_path / 'saved')
		status = 'done'
		reason = 'model_done'

		def summary(self):
			return {'status': self.status, 'reason': self.reason}

	class Runtime:
		def __init__(self, *args, **options):
			self.options = options

		async def run(self, task, **options):
			calls.append((self.options, options))
			saved = Saved()
			if variant == 'advice' and len(calls) == 1:
				saved.status, saved.reason = 'needs_advice', 'model_blocked'
			if 'parallel_evidence' in self.options:
				p = Path(saved.directory) / 'parallel'
				p.mkdir(parents=True, exist_ok=True)
				(p / 'result.json').write_text(json.dumps({'state': 'reconciled', 'draft': 'SOURCE DRAFT'}))
				await self.options['on_event']({'event': 'parent_started', 'phase': 'prepare', 'request_id': 'prepare'})
			await self.options['on_event'](
				{
					'event': 'observed',
					'observation': {'id': 'o', 'url': 'https://example.test', 'title': 'Page', 'fragment_text': 'Source'},
				}
			)
			return saved

	monkeypatch.setattr(measure, 'DecisionRuntime', Runtime)

	class Item:
		type = 'function_call'
		call_id = 'test-call'

		def __init__(self, name, arguments):
			self.name, self.arguments = name, json.dumps(arguments)

		def model_dump(self, **kw):
			return vars(self) | {'type': self.type, 'call_id': self.call_id}

	steps = []
	if variant == 'advice':
		steps.append([Item('supply_advice', {'text': 'The results have not updated; inspect submission.'})])
	steps.extend([[Item('write_artifact', {'path': 'report.md', 'content': 'SOURCE DRAFT'})], []])

	async def respond(**options):
		parent_inputs.append(options)
		items = steps.pop(0)
		return SimpleNamespace(status='completed', output=items, output_text='Finished', model_dump_json=lambda: '{}')

	monkeypatch.setattr(
		measure, 'AsyncOpenAI', lambda **kw: SimpleNamespace(responses=SimpleNamespace(create=respond), close=AsyncMock())
	)
	settings = measure.ExperimentSettings(variant=variant, memo_store=tmp_path, memo_ids=['a' * 32])
	result = await measure.execute_trial(
		{'id': 'test', 'start_url': 'https://example.test', 'goal': 'Read sources', 'exact_input_values': []},
		'goal',
		0,
		tmp_path / 'trial',
		settings,
	)
	assert result.status == 'finished', result.error
	assert result.experiment == variant and result.wall_seconds >= result.seconds
	if variant == 'advice':
		assert result.advice_calls == 1 and len(calls) == 2
		assert calls[1][1]['advice'].text.startswith('The results')
		assert calls[1][1]['resume'].status == 'needs_advice'
	elif variant == 'experience':
		assert calls[0][1]['experience'] == {'ids': ['a' * 32]}
	elif variant.startswith('parallel'):
		assert calls[0][0]['parallel_evidence']['concurrent'] == (variant == 'parallel')
		assert result.parent_preparation_requests == 1
		assert 'SOURCE DRAFT' in parent_inputs[0]['input'][2]['content']
	else:
		assert 'experience' not in calls[0][1]
		assert 'parallel_evidence' not in calls[0][0]
		assert 'max_advice_requests' not in calls[0][0]
	assert 'SOURCE DRAFT' in json.loads((tmp_path / 'trial/trace.json').read_text())['output_files_text']


def test_freeze_rejects_changed_source_or_memo_before_dispatch(tmp_path):
	memo = tmp_path / 'memo.json'
	memo.write_text('original')
	protocol = plan.Protocol(
		tasks=[],
		grading_specs={},
		order=[],
		sources={str(memo): plan.digest(memo)},
		commits={},
		env_file=tmp_path,
		upstream=tmp_path,
		memo_preparation=[],
	)
	plan.validate_frozen(protocol)
	memo.write_text('changed')
	with pytest.raises(ValueError, match='Frozen bytes changed'):
		plan.validate_frozen(protocol)
