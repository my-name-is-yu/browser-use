"""Negative controls for denominators, score arithmetic and a frozen holdout."""

import json
from typing import Literal

import httpx
import pytest

from benchmarks.bu_fast.evaluation_audit import (
	Price,
	Recorder,
	RequestUsage,
	Span,
	TaskGroup,
	TokenUsage,
	artifact_receipt,
	assert_frozen,
	assert_unused,
	collect_artifact,
	freeze_protocol,
	summarize_cost,
	summarize_timings,
	validate_judgement,
)
from benchmarks.bu_fast.validate_evaluation import (
	MeteredTransport,
	ResponsesJudge,
	Trace,
	configure_project_environment,
	judge_collected_trace,
	responses_judge_input,
)


def test_nested_intervals_do_not_double_count_or_label_unknown_as_inference():
	spans = [
		Span(id=0, parent=None, category='browser', start=1, end=8),
		Span(id=1, parent=0, category='wait', start=2, end=4),
		Span(id=2, parent=0, category='observe', start=5, end=6),
		Span(id=3, parent=None, category='parent', start=8, end=10),
	]
	result = summarize_timings(spans, 0, 12)
	assert result.exclusive_seconds == {'browser': 4, 'wait': 2, 'observe': 1, 'parent': 2}
	assert result.unattributed_seconds == 3
	assert sum(result.exclusive_seconds.values()) + result.unattributed_seconds == result.wall_seconds


@pytest.mark.parametrize(
	'spans',
	[
		[Span(id=0, parent=None, category='a', start=0, end=4), Span(id=1, parent=None, category='b', start=3, end=5)],
		[Span(id=0, parent=None, category='a', start=0, end=4), Span(id=1, parent=0, category='b', start=3, end=5)],
		[Span(id=0, parent=1, category='a', start=0, end=4), Span(id=1, parent=0, category='b', start=0, end=4)],
		[Span(id=0, parent=7, category='a', start=0, end=4)],
	],
)
def test_invalid_or_concurrent_interval_partition_is_rejected(spans):
	with pytest.raises(ValueError):
		summarize_timings(spans, 0, 6)


def test_exception_span_is_still_recorded():
	recorder = Recorder()
	with pytest.raises(RuntimeError), recorder.span('failed_request'):
		raise RuntimeError('failed')
	assert len(recorder.spans) == 1 and recorder.spans[0].end >= recorder.spans[0].start


def request(
	endpoint: Literal['decisions', 'responses', 'chat/completions', 'codex_cli'] = 'responses',
	status: Literal['completed', 'error', 'timeout'] = 'completed',
	**usage,
):
	return RequestUsage(
		role='decisions' if endpoint == 'decisions' else 'parent',
		model='test',
		endpoint=endpoint,
		status=status,
		service_tier='default',
		latency_seconds=1,
		usage=TokenUsage(**usage),
	)


def test_cost_separates_cache_read_write_and_does_not_charge_reasoning_twice():
	price = Price(model='test', endpoint='responses', input=2, cached=0.1, cache_write=2.5, output=10)
	result = summarize_cost([request(input_tokens=1000, cached_tokens=200, cache_write_tokens=100, output_tokens=300)], [price])
	assert result.complete
	assert result.known_usd == pytest.approx((700 * 2 + 200 * 0.1 + 100 * 2.5 + 300 * 10) / 1e6)


def test_failed_and_timed_out_attempts_prevent_complete_total():
	price = Price(model='test', endpoint='decisions', input=0.1, cached=0, cache_write=0, output=0)
	result = summarize_cost(
		[request('decisions', input_tokens=100), request('decisions', status='error'), request('decisions', status='timeout')],
		[price],
	)
	assert result.known_usd == pytest.approx(0.00001)
	assert not result.complete and result.unknown_attempts == 2


def test_missing_usage_or_cache_fields_are_not_zero_cost():
	price = Price(model='test', endpoint='responses', input=2, cached=0.1, cache_write=2.5, output=10)
	result = summarize_cost([request(input_tokens=1000, output_tokens=30)], [price])
	assert not result.complete and result.unknown_attempts == 1
	assert result.missing_fields == ['0:cached_tokens', '0:cache_write_tokens']
	assert result.known_usd == 0  # Known subtotal only; the total is unavailable.


def test_decisions_only_input_is_charged_with_long_context_and_region():
	price = Price(model='test', endpoint='decisions', input=0.1, cached=0, cache_write=0, output=0, regional_multiplier=1.1)
	result = summarize_cost([request('decisions', input_tokens=300000, cached_tokens=10000, output_tokens=900)], [price])
	assert result.complete and result.known_usd == pytest.approx(0.066)


def test_codex_subscription_usage_cannot_be_relabelled_as_api_spend():
	result = summarize_cost([request('codex_cli', input_tokens=100, output_tokens=10)], [])
	assert not result.complete and result.missing_fields == ['0:pricing_or_billing_basis']


def test_unpinned_processing_tier_cannot_receive_standard_price():
	price = Price(model='test', endpoint='responses', input=2, cached=0.1, cache_write=2.5, output=10)
	row = request(input_tokens=100, output_tokens=10, cached_tokens=0, cache_write_tokens=0)
	row.service_tier = 'priority'
	result = summarize_cost([row], [price])
	assert not result.complete and result.missing_fields == ['0:service_tier']


def judgement(statuses=('met', 'met'), score: float = 1):
	return {
		'score': score,
		'judgement': {
			'findings': [
				{'item': item, 'status': status, 'not_assessable_reason': None} for item, status in zip(('A', 'B'), statuses)
			]
		},
	}


def test_full_semantic_score_cannot_override_wrong_persisted_state():
	result = validate_judgement(judgement(), {'A': 30, 'B': 70}, oracle=False)
	assert result.score == 1 and not result.complete and result.reasons == ['oracle_rejected']


def test_partial_credit_and_collector_loss_are_distinct():
	partial = validate_judgement(judgement(('met', 'violated'), 0.3), {'A': 30, 'B': 70}, oracle=False)
	assert partial.assessable and not partial.complete
	lost = judgement(('met', 'not_assessable'), 0.3)
	lost['judgement']['findings'][1]['not_assessable_reason'] = 'missing_evidence'
	assert not validate_judgement(lost, {'A': 30, 'B': 70}, oracle=None).assessable
	clipped = {**judgement(), 'evidence_clipped_sections': ['files']}
	assert not validate_judgement(clipped, {'A': 30, 'B': 70}, oracle=True).complete


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unknown', 'arithmetic', 'invalid_reason'])
def test_malformed_judge_does_not_enter_score_denominator(mutation):
	value = judgement()
	if mutation == 'missing':
		value['judgement']['findings'].pop()
	elif mutation == 'duplicate':
		value['judgement']['findings'][1]['item'] = 'A'
	elif mutation == 'unknown':
		value['judgement']['findings'][1]['item'] = 'C'
	elif mutation == 'arithmetic':
		value['score'] = 0.5
	else:
		value['judgement']['findings'][0]['not_assessable_reason'] = 'missing_evidence'
	with pytest.raises(ValueError):
		validate_judgement(value, {'A': 30, 'B': 70}, oracle=True)


@pytest.mark.parametrize('field', ['id', 'domain', 'template', 'family'])
def test_holdout_rejects_each_kind_of_exposure(field):
	seen = TaskGroup(id='dev', domain='a', template='table', family='lookup')
	fresh = TaskGroup(id='holdout', domain='b', template='form', family='export')
	setattr(fresh, field, getattr(seen, field))
	with pytest.raises(ValueError, match=field):
		assert_unused([fresh], [seen])


def test_unused_group_passes_but_code_or_protocol_mutation_fails(tmp_path):
	assert_unused([TaskGroup(id='h', domain='b', template='form', family='export')], [])
	source = tmp_path / 'executor.py'
	source.write_text('initial')
	manifest = tmp_path / 'protocol.json'
	freeze_protocol(manifest, {'phase': 'acceptance'}, [source])
	assert assert_frozen(manifest, [source])['phase'] == 'acceptance'
	with pytest.raises(FileExistsError):
		freeze_protocol(manifest, {'phase': 'acceptance'}, [source])
	source.write_text('adjustment')
	with pytest.raises(ValueError):
		assert_frozen(manifest, [source])
	source.write_text('initial')
	value = json.loads(manifest.read_text())
	value['protocol']['phase'] = 'development'
	manifest.write_text(json.dumps(value))
	with pytest.raises(ValueError):
		assert_frozen(manifest, [source])


@pytest.mark.parametrize('status', [200, 429, 500])
async def test_http_attempt_is_durable_before_caller_interprets_response(tmp_path, status):
	rows = []
	ledger = tmp_path / 'requests.jsonl'
	transport = MeteredTransport('parent', rows, ledger)
	transport.inner = httpx.MockTransport(
		lambda _: httpx.Response(
			status,
			headers={'x-request-id': 'probe'},
			json={
				'usage': {'input_tokens': 10, 'output_tokens': 2, 'input_tokens_details': {'cached_tokens': 0}},
				'service_tier': 'default',
			},
		)
	)
	async with httpx.AsyncClient(transport=transport) as client:
		await client.post(
			'https://api.openai.com/v1/responses',
			headers={'Authorization': 'Bearer secret'},
			json={'model': 'test', 'input': 'private'},
		)
	record = json.loads(ledger.read_text())
	assert len(rows) == 1 and rows[0].status == ('completed' if status == 200 else 'error')
	assert record['usage']['input_tokens'] == (10 if status == 200 else None)
	assert 'secret' not in ledger.read_text() and 'private' not in ledger.read_text()


async def test_cancelled_request_is_not_lost_from_cost_denominator(tmp_path):
	import asyncio

	rows = []
	ledger = tmp_path / 'requests.jsonl'
	transport = MeteredTransport('parent', rows, ledger)

	async def cancel(_):
		raise asyncio.CancelledError()

	transport.inner = httpx.MockTransport(cancel)
	async with httpx.AsyncClient(transport=transport) as client:
		with pytest.raises(asyncio.CancelledError):
			await client.post('https://api.openai.com/v1/responses', json={'model': 'test'})
	assert rows[0].status == 'timeout' and rows[0].usage.input_tokens is None
	assert json.loads(ledger.read_text())['status'] == 'timeout'


def test_selected_project_key_wins_over_inherited_key_without_logging_secrets(monkeypatch, tmp_path):
	import os

	monkeypatch.setenv('OPENAI_API_KEY', 'inherited-secret')
	path = tmp_path / '.env.local'
	path.write_text('OPENAI_API_KEY=project-secret\n')
	binding = configure_project_environment(path)
	assert os.environ['OPENAI_API_KEY'] == 'project-secret'
	assert binding.source == 'project_env_file' and binding.project_key_differs_from_inherited
	assert 'secret' not in binding.model_dump_json()


def test_inherited_key_is_supported_when_no_project_key_exists(monkeypatch, tmp_path):
	monkeypatch.setenv('OPENAI_API_KEY', 'inherited-secret')
	binding = configure_project_environment(tmp_path / '.env.local')
	assert binding.source == 'inherited_environment' and not binding.project_key_differs_from_inherited


def test_missing_credential_is_an_explicit_configuration_failure(monkeypatch, tmp_path):
	monkeypatch.delenv('OPENAI_API_KEY', raising=False)
	with pytest.raises(RuntimeError, match='not configured'):
		configure_project_environment(tmp_path / '.env.local')


@pytest.mark.parametrize('mutation', ['none', 'remove', 'change'])
def test_collection_receipt_detects_loss_and_changes_independently_of_claims(tmp_path, mutation):
	path = tmp_path / 'delivery.csv'
	path.write_text('name,city\nAlice,Tokyo\n')
	receipt = artifact_receipt(path)
	if mutation == 'remove':
		path.unlink()
	elif mutation == 'change':
		path.write_text('name,city\nTokyo,Alice\n')
	content, gap = collect_artifact(receipt)
	if mutation == 'none':
		assert content == 'name,city\nAlice,Tokyo\n' and gap is None
	else:
		assert content is None and gap is not None
		assert gap.reason == ('unreadable' if mutation == 'remove' else 'content_changed')
		result = {'score': 1.0, 'judgement': {'findings': [{'item': 'A', 'status': 'met', 'not_assessable_reason': None}]}}
		verdict = validate_judgement(result, {'A': 100}, oracle=None, collection_gaps=[gap])
		assert verdict.score == 1.0 and not verdict.complete and not verdict.assessable
		assert verdict.reasons == ['collection_gap']


def test_responses_judge_translation_preserves_roles_order_text_and_image_detail():
	from types import SimpleNamespace as Part

	messages = [
		Part(role='system', content='Exact public prompt'),
		Part(
			role='user',
			content=[
				Part(type='text', text='Screenshot at step 5'),
				Part(type='image_url', image_url=Part(url='data:image/png;base64,eA==', detail='high')),
			],
		),
	]
	assert responses_judge_input(messages) == [
		{'role': 'system', 'content': [{'type': 'input_text', 'text': 'Exact public prompt'}]},
		{
			'role': 'user',
			'content': [
				{'type': 'input_text', 'text': 'Screenshot at step 5'},
				{'type': 'input_image', 'image_url': 'data:image/png;base64,eA==', 'detail': 'high'},
			],
		},
	]
	with pytest.raises(ValueError):
		responses_judge_input([Part(role='assistant', content='Injected answer')])


@pytest.mark.parametrize('status', ['completed', 'incomplete'])
async def test_responses_judge_uses_pinned_schema_and_full_usage_and_rejects_truncation(status):
	from types import SimpleNamespace

	from pydantic import BaseModel

	class Result(BaseModel):
		met: bool

	seen = []

	def respond(request):
		seen.append(json.loads(request.content))
		return httpx.Response(
			200,
			json={
				'id': 'resp_test',
				'object': 'response',
				'created_at': 0,
				'status': status,
				'output': [
					{
						'id': 'msg_test',
						'type': 'message',
						'role': 'assistant',
						'status': 'completed',
						'content': [{'type': 'output_text', 'text': '{"met":true}', 'annotations': []}],
					}
				],
				'usage': {
					'input_tokens': 1200,
					'output_tokens': 20,
					'total_tokens': 1220,
					'input_tokens_details': {'cached_tokens': 100, 'cache_write_tokens': 800},
					'output_tokens_details': {'reasoning_tokens': 5},
				},
				'service_tier': 'default',
			},
		)

	async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
		judge = ResponsesJudge(
			SimpleNamespace(
				model='pinned-model',
				api_key='fake',
				base_url=None,
				reasoning_effort='xhigh',
				max_completion_tokens=32768,
				timeout=30,
				max_retries=2,
			),
			client,
		)
		messages = [SimpleNamespace(role='system', content='Public prompt')]
		if status == 'completed':
			result = await judge.ainvoke(messages, Result)
			assert isinstance(result.completion, Result) and result.usage is not None
			assert result.completion.met and result.usage.prompt_cache_creation_tokens == 800
			assert result.usage.completion_tokens == 20
		else:
			with pytest.raises(RuntimeError, match='completed structured'):
				await judge.ainvoke(messages, Result)
	assert seen[0]['model'] == 'pinned-model' and seen[0]['reasoning'] == {'effort': 'xhigh'}
	assert seen[0]['max_output_tokens'] == 32768 and seen[0]['service_tier'] == 'default'
	assert seen[0]['text']['format']['strict'] and seen[0]['text']['format']['schema']['additionalProperties'] is False


@pytest.mark.parametrize('verified_gap', [True, False])
async def test_collection_admission_skips_only_harness_verified_failures(tmp_path, verified_gap):
	path = tmp_path / 'created.csv'
	path.write_text('observed delivery')
	receipt = artifact_receipt(path)
	path.unlink()
	_, gap = collect_artifact(receipt)
	assert gap is not None
	trace = Trace(
		final_result='The file is missing.',
		agent_steps=[],
		screenshots_b64=[],
		screenshot_steps=[],
		collection_gaps=[gap] if verified_gap else [],
	)
	calls = []

	async def fake_judge(task, trace, judge, artifact_dir):
		calls.append(task)
		return {'semantic_score': 0.5}

	result = await judge_collected_trace({'id': 'control'}, trace, None, fake_judge, tmp_path / 'judge')
	if verified_gap:
		assert not calls and result['assessment_status'] == 'collection_failure'
		assert result['semantic_score'] is None
	else:
		assert len(calls) == 1 and result['semantic_score'] == 0.5


async def test_incomplete_model_output_remains_billed_with_its_terminal_reason(tmp_path):
	rows = []
	transport = MeteredTransport('judge', rows, tmp_path / 'requests.jsonl')
	transport.inner = httpx.MockTransport(
		lambda _: httpx.Response(
			200,
			json={
				'status': 'incomplete',
				'incomplete_details': {'reason': 'max_output_tokens'},
				'service_tier': 'default',
				'usage': {
					'input_tokens': 10,
					'output_tokens': 32768,
					'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
				},
			},
		)
	)
	async with httpx.AsyncClient(transport=transport) as client:
		await client.post('https://api.openai.com/v1/responses', json={'model': 'test'})
	assert rows[0].response_status == 'incomplete' and rows[0].incomplete_reason == 'max_output_tokens'
	price = Price(model='test', endpoint='responses', input=0.2, cached=0.02, cache_write=0.25, output=1.2)
	cost = summarize_cost(rows, [price])
	assert cost.complete and cost.known_usd == pytest.approx((10 * 0.2 + 32768 * 1.2) / 1e6)
