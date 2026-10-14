"""Exercise the adopted workflow's evidence, recovery and failure paths."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from browser_use.fast import workflow
from browser_use.fast.views import Observation, RunBudget, RunResult


@pytest.fixture
def harness(tmp_path, monkeypatch):
	monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
	page = SimpleNamespace(goto=AsyncMock())
	browser = SimpleNamespace(start=AsyncMock(), kill=AsyncMock(), agent_focus_target_id='tab')
	parent_close, decision_close = AsyncMock(), AsyncMock()
	monkeypatch.setattr(workflow, 'BrowserSession', lambda **kw: browser)
	monkeypatch.setattr(workflow, 'Page', lambda *a: page)
	monkeypatch.setattr(workflow, 'DecisionClient', lambda **kw: SimpleNamespace(close=decision_close))
	observation = Observation(
		id='source',
		document_id='doc',
		target_id='tab',
		url='https://example.test/',
		title='Source',
		text='SOURCE FACT',
		viewport_text='SOURCE FACT',
		targets=[],
		total_targets=0,
	)
	after = observation.model_copy(update={'id': 'changed', 'url': 'https://example.test/recovered'})
	monkeypatch.setattr(workflow, 'BrowserDriver', lambda *a: SimpleNamespace(observe=AsyncMock(return_value=after)))
	state = SimpleNamespace(
		browser=browser,
		page=page,
		parent_close=parent_close,
		decision_close=decision_close,
		calls=[],
		inputs=[],
		steps=[],
		status='done',
		startup_error=None,
	)

	class Runtime:
		def __init__(self, *args, **options):
			self.options = options

		async def run(self, task=None, **options):
			state.calls.append((task, options))
			if state.startup_error:
				raise state.startup_error
			item = after if options.get('recovery') else observation
			self.options['ledger'].add(item.model_dump(exclude={'screenshot'}))
			if options.get('resume'):
				return options['resume'].model_copy(update={'status': 'done', 'reason': 'model_done', 'observation': item})
			return RunResult(
				run_id='same-run',
				directory=str(tmp_path / 'saved'),
				task=task,
				initial_inputs=task.inputs,
				budget=RunBudget(max_decisions=self.options['max_decisions'], deadline_at=time.time() + 180),
				status=state.status,
				reason='model_' + state.status,
				observation=item,
				cdp_url='http://browser',
				target_id='tab',
			)

	monkeypatch.setattr(workflow, 'LedgerExplorer', Runtime)

	async def respond(**options):
		state.inputs.append(json.loads(json.dumps(options['input'])))
		step = state.steps.pop(0)
		if callable(step):
			step = step(options['input'])
		name, arguments = step
		item = SimpleNamespace(type='function_call', call_id=str(len(state.inputs)), name=name, arguments=json.dumps(arguments))
		item.model_dump = lambda **kw: {
			'type': item.type,
			'call_id': item.call_id,
			'name': item.name,
			'arguments': item.arguments,
		}
		return SimpleNamespace(status='completed', output=[item], model_dump_json=lambda: '{}')

	monkeypatch.setattr(
		workflow, 'AsyncOpenAI', lambda **kw: SimpleNamespace(responses=SimpleNamespace(create=respond), close=parent_close)
	)
	return state


def cited(messages, *, missing=False, forged=False):
	citations = [
		c
		for message in messages
		if message.get('role') == 'user'
		for packet in [json.loads(message['content']) if message['content'].startswith('{') else {}]
		for c in packet.get('retained_sources', {}).get('citations', [])
	]
	return (
		'complete_delivery',
		{
			'disposition': 'incomplete' if missing else 'complete',
			'records': [
				{
					'id': 'answer',
					'value': None if missing else 'SOURCE FACT',
					'citation_ids': [] if missing else ['invented'] if forged else [citations[0]['id']],
				}
			],
			'issues': ['Missing answer'] if missing else [],
		},
	)


async def run_case(tmp_path, **settings):
	task = workflow.WorkflowTask(
		id='source',
		start_url='https://example.test/',
		goal='Read the observed source fact',
		required_fields=[{'id': 'answer', 'requirement': 'Observed source fact'}],
	)
	return await workflow.run_workflow(task, tmp_path / 'workflow', workflow.WorkflowSettings(**settings))


@pytest.mark.parametrize('missing', [False, True])
async def test_cited_delivery_saves_validated_outputs_and_closes_resources(tmp_path, harness, missing):
	harness.steps = [lambda messages: cited(messages, missing=missing)]
	result = await run_case(tmp_path)
	assert result.status == ('incomplete' if missing else 'complete') and (result.delivery is not None)
	directory = tmp_path / 'workflow'
	delivery = json.loads((directory / 'delivery.json').read_text())
	assert delivery['disposition'] == ('incomplete' if missing else 'complete')
	assert delivery['records'] == json.loads((directory / 'outputs/results.json').read_text())
	assert (directory / 'final.txt').read_text() == delivery['final_answer']
	assert delivery['final_answer'] in (directory / 'outputs/report.md').read_text()
	assert result.parent_calls == 1
	harness.browser.kill.assert_awaited_once()
	harness.parent_close.assert_awaited_once()
	harness.decision_close.assert_awaited_once()


async def test_forged_citation_is_rejected_then_valid_delivery_can_recover(tmp_path, harness):
	harness.steps = [lambda m: cited(m, forged=True), cited]
	result = await run_case(tmp_path, parent_turn_budget=2)
	assert (result.delivery is not None) and result.parent_calls == 2
	exchanges = [json.loads(line) for line in (tmp_path / 'workflow/tool-exchanges.jsonl').read_text().splitlines()]
	assert exchanges[0]['status'] == 'error' and 'not offered' in exchanges[0]['result']['error']
	assert exchanges[1]['status'] == 'completed'


async def test_continue_does_not_save_terminal_delivery(tmp_path, harness):
	harness.steps = [('complete_delivery', {'disposition': 'continue', 'records': [], 'issues': ['Need more evidence']}), cited]
	result = await run_case(tmp_path, parent_turn_budget=2)
	assert result.delivery is not None
	assert json.loads((tmp_path / 'workflow/attempted-delivery-00.json').read_text())['disposition'] == 'continue'


@pytest.mark.parametrize('tool,status', [('resume_browser', 'needs_input'), ('repair_browser', 'done')])
async def test_handoff_keeps_same_task_and_budget(tmp_path, harness, tool, status):
	harness.status = status
	harness.steps = [
		(tool, {'inputs': ['exact input']} if tool == 'resume_browser' else {'deficiency': 'Missing applied result'}),
		cited,
	]
	result = await run_case(tmp_path, parent_turn_budget=2)
	assert (result.delivery is not None) and len(harness.calls) == 2
	assert harness.calls[1][0] is None
	assert harness.calls[1][1]['resume'].run_id == 'same-run'
	assert harness.calls[1][1]['resume'].task.goal == 'Read the observed source fact'
	assert harness.calls[1][1]['resume'].budget.max_decisions == 90


async def test_retained_search_returns_source_ids_without_browser_replay(tmp_path, harness):
	harness.steps = [('read_retained_evidence', {'query': 'SOURCE FACT'}), cited]
	result = await run_case(tmp_path, parent_turn_budget=2)
	assert (result.delivery is not None) and len(harness.calls) == 1
	assert 'SOURCE FACT' in json.dumps(harness.inputs[1])
	harness.page.goto.assert_not_awaited()


@pytest.mark.parametrize('fail', [False, True])
async def test_only_acknowledged_native_recovery_reenters(tmp_path, harness, fail):
	harness.status = 'blocked'
	if fail:
		harness.page.goto.side_effect = RuntimeError('effect unknown')
	harness.steps = [('browser_batch', {'actions': [{'op': 'goto', 'value': 'https://example.test/recovered'}]}), cited]
	result = await run_case(tmp_path, parent_turn_budget=2)
	assert result.delivery is not None
	assert len(harness.calls) == (1 if fail else 2)
	assert result.unknown_operation_effect is fail
	if not fail:
		receipt = harness.calls[1][1]['recovery']
		assert receipt.run_id == 'same-run' and receipt.before.id == 'source' and receipt.after.id == 'changed'


async def test_unknown_effect_cannot_be_cleared_by_later_success(tmp_path, harness):
	harness.status = 'blocked'
	harness.page.goto.side_effect = [RuntimeError('effect unknown'), None]
	harness.steps = [
		('browser_batch', {'actions': [{'op': 'goto', 'value': 'https://example.test/first'}]}),
		('browser_batch', {'actions': [{'op': 'goto', 'value': 'https://example.test/recovered'}]}),
		cited,
	]
	result = await run_case(tmp_path, parent_turn_budget=3)
	assert result.unknown_operation_effect and len(harness.calls) == 1
	assert 'unresolved effect' in json.dumps(harness.inputs[-1])


async def test_pending_dispatch_times_out_without_reentry(tmp_path, harness):
	harness.status = 'blocked'

	async def pending(*a):
		await asyncio.Event().wait()

	harness.page.goto.side_effect = pending
	harness.steps = [('browser_batch', {'actions': [{'op': 'goto', 'value': 'https://example.test/pending'}]})]
	result = await run_case(tmp_path, deadline_seconds=1, parent_turn_budget=1)
	assert result.status == 'timeout' and result.unknown_operation_effect and not (result.delivery is not None)
	harness.browser.kill.assert_awaited_once()


@pytest.mark.parametrize('error', [TimeoutError('deadline'), RuntimeError('failed')])
async def test_exploration_failure_still_closes_resources(tmp_path, harness, error):
	harness.startup_error = error
	result = await run_case(tmp_path)
	assert result.status == ('timeout' if isinstance(error, TimeoutError) else 'error')
	assert not (result.delivery is not None) and result.parent_calls == 0
	harness.browser.kill.assert_awaited_once()


async def test_physical_parent_cap_rejects_extra_http_request(tmp_path):
	from browser_use.fast.workflow_support import RequestTransport

	dispatch = AsyncMock(return_value=httpx.Response(200, json={'usage': {'input_tokens': 2}}))
	rows = []
	transport = RequestTransport('parent', rows, tmp_path / 'http.jsonl', 2)
	transport.inner.handle_async_request = dispatch
	try:
		for _ in range(2):
			await transport.handle_async_request(
				httpx.Request('POST', 'https://api.openai.com/v1/responses', json={'model': 'gpt-6.1-sol'})
			)
		with pytest.raises(ValueError, match='budget exhausted'):
			await transport.handle_async_request(
				httpx.Request('POST', 'https://api.openai.com/v1/responses', json={'model': 'gpt-6.1-sol'})
			)
		assert dispatch.await_count == 2 and len(rows) == 2
	finally:
		await transport.aclose()


async def test_cancellation_releases_workflow_owned_resources(tmp_path, harness):
	harness.startup_error = asyncio.CancelledError()
	with pytest.raises(asyncio.CancelledError):
		await run_case(tmp_path)
	assert json.loads((tmp_path / 'workflow/result.json').read_text())['status'] == 'interrupted'
	harness.browser.kill.assert_awaited_once()
	harness.parent_close.assert_awaited_once()
	harness.decision_close.assert_awaited_once()


async def test_shared_table_missing_field_saves_once_without_fabricated_value(tmp_path, harness):
	def missing_table(messages):
		name, draft = cited(messages)
		draft['disposition'] = 'incomplete'
		draft['records'][0]['value'] = None
		draft['records'][0]['table_ids'] = ['contact']
		draft['records'].append({'id': 'phone', 'value': None, 'citation_ids': [], 'table_ids': ['contact']})
		draft['tables'] = [
			{
				'id': 'contact',
				'title': 'Contact',
				'columns': ['fact', 'phone'],
				'rows': [['SOURCE FACT', '']],
				'csv_path': 'contact.csv',
			}
		]
		draft['issues'] = ['Phone absent']
		return name, draft

	harness.steps = [missing_table]
	task = workflow.WorkflowTask(
		id='contact',
		start_url='https://example.test/',
		goal='Find fact and phone',
		required_fields=[{'id': 'answer', 'requirement': 'Source fact'}, {'id': 'phone', 'requirement': 'Phone, null if absent'}],
	)
	result = await workflow.run_workflow(task, tmp_path / 'workflow', workflow.WorkflowSettings(parent_turn_budget=1))
	assert (result.delivery is not None) and result.parent_calls == 1
	delivery = json.loads((tmp_path / 'workflow/delivery.json').read_text())
	assert delivery['disposition'] == 'incomplete' and delivery['records'][1]['value'] is None
	assert delivery['records'][1]['quote'] is None
	assert (tmp_path / 'workflow/outputs/contact.csv').read_text() == 'fact,phone\nSOURCE FACT,\n'
	assert delivery['final_answer'].count('Contact') == 1


async def test_unvalidated_attempt_never_becomes_final_output(tmp_path, harness):
	harness.steps = [lambda m: cited(m, forged=True)]
	result = await run_case(tmp_path, parent_turn_budget=1)
	assert result.status == 'error' and result.delivery is None and 'not offered' in result.error
	directory = tmp_path / 'workflow'
	assert (directory / 'attempted-delivery-00.json').exists()
	assert not (directory / 'final.txt').exists() and not (directory / 'delivery.json').exists()
	assert not (directory / 'outputs/report.md').exists()


async def test_file_save_failure_is_not_completion(tmp_path, harness, monkeypatch):
	harness.steps = [cited]
	monkeypatch.setattr(workflow, 'save_delivery', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
	result = await run_case(tmp_path, parent_turn_budget=1)
	assert result.status == 'error' and result.delivery is None and 'disk full' in result.error
	assert not (tmp_path / 'workflow/final.txt').exists()


async def test_cleanup_error_releases_other_resources_and_reports_failure(tmp_path, harness):
	harness.steps = [cited]
	harness.parent_close.side_effect = RuntimeError('close failed')
	result = await run_case(tmp_path)
	assert result.status == 'error' and 'Cleanup failed' in result.error
	harness.browser.kill.assert_awaited_once()
	harness.decision_close.assert_awaited_once()
	assert result.delivery is not None


@pytest.mark.parametrize(
	'fields',
	[
		[{'id': 'a', 'requirement': 'First'}, {'id': 'a', 'requirement': 'Second'}],
		[{'id': 'COMPLETE', 'requirement': 'Reserved ID'}],
	],
)
def test_task_rejects_invalid_requirements_before_browser_start(fields):
	with pytest.raises(ValueError, match='unique IDs'):
		workflow.WorkflowTask(id='invalid', start_url='https://example.test/', goal='Read', required_fields=fields)


def test_task_supports_more_than_eight_requirements_and_validates_url():
	task = workflow.WorkflowTask(
		id='many',
		start_url='https://example.test/',
		goal='Read all facts',
		required_fields=[{'id': str(i), 'requirement': f'Fact {i}'} for i in range(9)],
	)
	assert len(task.required_fields) == 9
	with pytest.raises(ValueError, match='HTTP'):
		workflow.WorkflowTask(id='invalid', start_url='file:///tmp/data', goal='Read', required_fields=task.required_fields)


async def test_parent_timeout_cancels_http_request_and_keeps_unknown_usage(tmp_path, harness, monkeypatch):
	async def create(**options):
		transport = client_options['http_client']._transport

		async def pending(request):
			await asyncio.Event().wait()

		transport.inner.handle_async_request = pending
		return await transport.handle_async_request(
			httpx.Request('POST', 'https://api.openai.com/v1/responses', json={'model': options['model']})
		)

	client_options = {}

	def client(**options):
		client_options.update(options)
		return SimpleNamespace(responses=SimpleNamespace(create=create), close=harness.parent_close)

	monkeypatch.setattr(workflow, 'AsyncOpenAI', client)
	result = await run_case(tmp_path, deadline_seconds=1)
	assert result.status == 'timeout' and len(result.requests) == 1
	assert result.requests[0].status == 'cancelled' and result.requests[0].usage.input_tokens is None
	saved = json.loads((tmp_path / 'workflow/result.json').read_text())
	assert saved['requests'][0]['status'] == 'cancelled' and saved['seconds'] >= 1


def test_terminal_status_requires_matching_delivery(tmp_path):
	with pytest.raises(ValueError, match='matching validated delivery'):
		workflow.WorkflowResult(task='invalid', directory=str(tmp_path), status='complete')


async def test_invalid_native_input_does_not_taint_recovery(tmp_path, harness):
	harness.status = 'blocked'
	harness.steps = [('browser_batch', {'actions': [{'op': 'goto', 'value': 'file:///tmp/data'}]}), cited]
	result = await run_case(tmp_path, parent_turn_budget=2)
	assert result.status == 'complete' and not result.unknown_operation_effect
	harness.page.goto.assert_not_awaited()


async def test_interrupt_during_native_dispatch_preserves_unknown_effect(tmp_path, harness):
	harness.status = 'blocked'
	harness.steps = [('browser_batch', {'actions': [{'op': 'goto', 'value': 'https://example.test/pending'}]})]
	harness.page.goto.side_effect = asyncio.CancelledError()
	with pytest.raises(asyncio.CancelledError):
		await run_case(tmp_path)
	saved = json.loads((tmp_path / 'workflow/result.json').read_text())
	assert saved['status'] == 'interrupted' and saved['unknown_operation_effect']
	harness.browser.kill.assert_awaited_once()


async def test_browser_start_obeys_whole_task_deadline(tmp_path, harness):
	async def pending():
		await asyncio.Event().wait()

	harness.browser.start.side_effect = pending
	result = await run_case(tmp_path, deadline_seconds=1)
	assert result.status == 'timeout' and not harness.calls and result.parent_calls == 0
	harness.browser.kill.assert_awaited_once()
