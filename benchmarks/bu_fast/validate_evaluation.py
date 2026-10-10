"""Freeze and run measurement/grader calibration on exposed development controls.

Run with uv run python -m benchmarks.bu_fast.validate_evaluation --output ...
This never imports load_tasks or decrypts the unused benchmark dataset.
"""

import argparse
import asyncio
import base64
import csv
import importlib
import io
import json
import os
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values, load_dotenv
from openai import AsyncOpenAI
from openai.types.responses import ResponseInputContentParam, ResponseInputParam
from pydantic import BaseModel, Field

from benchmarks.bu_fast.evaluation_audit import (
	CollectionGap,
	Price,
	Recorder,
	RequestUsage,
	TokenUsage,
	artifact_receipt,
	assert_frozen,
	collect_artifact,
	freeze_protocol,
	source_hashes,
	summarize_cost,
	summarize_timings,
	validate_judgement,
)
from benchmarks.bu_fast.fixture import FixtureHandler

REPO = Path(__file__).resolve().parents[2]
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'


class CredentialSource(BaseModel):
	"""Record credential routing without recording the secret itself."""

	source: str
	project_key_differs_from_inherited: bool


def configure_project_environment(path: Path) -> CredentialSource:
	"""Use the explicitly selected project configuration before inherited app settings."""
	inherited = os.environ.get('OPENAI_API_KEY')
	project = dotenv_values(path) if path.is_file() else {}
	project_key = project.get('OPENAI_API_KEY')
	load_dotenv(path, override=True)
	if not os.environ.get('OPENAI_API_KEY'):
		raise RuntimeError('OPENAI_API_KEY is not configured')
	return CredentialSource(
		source='project_env_file' if project_key else 'inherited_environment',
		project_key_differs_from_inherited=bool(project_key and inherited and project_key != inherited),
	)


class CalibrationCase(BaseModel):
	"""An exposed negative/positive control, never an adoption-evaluation task."""

	id: str
	kind: str
	state: dict[str, str] | None = None
	save: bool = False
	claim: str = 'Saved Alice, Tokyo, team.'
	artifact: str = ''
	expected_met: list[str] = Field(default_factory=list)
	expected_assessable: bool = True
	oracle: bool | None = False


class Trace(BaseModel):
	"""Actual browser evidence and files; assertions in the final text are not proof."""

	final_result: str
	agent_steps: list[str]
	screenshots_b64: list[str]
	screenshot_steps: list[int]
	screenshot_timing: str = 'after'
	output_files_text: str = ''
	evidence_notes: list[str] = Field(default_factory=list)
	collection_gaps: list[CollectionGap] = Field(default_factory=list)


CASES = [
	CalibrationCase(
		id='correct-save',
		kind='form',
		state={'name': 'Alice', 'city': 'Tokyo', 'plan': 'team'},
		save=True,
		expected_met=['NAME', 'CITY', 'PLAN'],
		oracle=True,
	),
	CalibrationCase(
		id='swapped-fields',
		kind='form',
		state={'name': 'Tokyo', 'city': 'Alice', 'plan': 'team'},
		save=True,
		expected_met=['PLAN'],
	),
	CalibrationCase(
		id='wrong-plan',
		kind='form',
		state={'name': 'Alice', 'city': 'Tokyo', 'plan': 'solo'},
		save=True,
		expected_met=['NAME', 'CITY'],
	),
	CalibrationCase(id='filled-unsaved', kind='form', state={'name': 'Alice', 'city': 'Tokyo', 'plan': 'team'}),
	CalibrationCase(id='claim-only', kind='form'),
	CalibrationCase(
		id='correct-file',
		kind='file',
		artifact='plan,people,price\nSolo,1,10\nTeam,3,24\nBusiness,10,80\n',
		claim='Created plans.csv.',
		expected_met=['SOLO', 'TEAM', 'BUSINESS'],
		oracle=True,
	),
	CalibrationCase(
		id='omitted-row',
		kind='file',
		artifact='plan,people,price\nSolo,1,10\nTeam,3,24\n',
		claim='Created plans.csv with every plan.',
		expected_met=['SOLO', 'TEAM'],
	),
	CalibrationCase(
		id='corrupt-cell',
		kind='file',
		artifact='plan,people,price\nSolo,1,100\nTeam,3,24\nBusiness,10,80\n',
		claim='Created plans.csv.',
		expected_met=['TEAM', 'BUSINESS'],
	),
	CalibrationCase(id='blocked', kind='blocked', claim='The page returned HTTP 403; I could not save the subscription.'),
	CalibrationCase(
		id='collector-loss',
		kind='collector-loss',
		claim='Created plans.csv.',
		artifact='plan,people,price\nSolo,1,10\nTeam,3,24\nBusiness,10,80\n',
		expected_assessable=False,
		oracle=None,
	),
]


def responses_judge_input(messages: list[Any]) -> ResponseInputParam:
	"""Preserve the public judge's text, image order and detail when changing endpoints."""
	result: ResponseInputParam = []
	for message in messages:
		if message.role not in ('system', 'user'):
			raise ValueError('Only the public single-turn judge messages are supported')
		parts = message.content
		content: list[ResponseInputContentParam]
		if isinstance(parts, str):
			content = [{'type': 'input_text', 'text': parts}]
		else:
			content = []
			for part in parts:
				if part.type == 'text':
					content.append({'type': 'input_text', 'text': part.text})
				elif part.type == 'image_url':
					content.append({'type': 'input_image', 'image_url': part.image_url.url, 'detail': part.image_url.detail})
				else:
					raise ValueError('Unsupported judge content part')
		result.append({'role': message.role, 'content': content})
	return result


class ResponsesJudge:
	"""Use the public grader's prompt and schema with complete Responses usage reporting."""

	def __init__(self, pinned_judge: Any, http_client: httpx.AsyncClient):
		self.model = pinned_judge.model
		self.reasoning_effort = pinned_judge.reasoning_effort
		self.max_completion_tokens = pinned_judge.max_completion_tokens
		self.base_url = pinned_judge.base_url
		self.client = AsyncOpenAI(
			api_key=pinned_judge.api_key,
			http_client=http_client,
			timeout=pinned_judge.timeout,
			max_retries=pinned_judge.max_retries,
		)

	async def ainvoke(self, messages: list[Any], output_format: type[BaseModel]):
		"""Retain exact schema optimization and reject truncated/refused structured output."""
		from browser_use.llm.schema import SchemaOptimizer
		from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

		response = await self.client.responses.create(
			model=self.model,
			input=responses_judge_input(messages),
			reasoning={'effort': self.reasoning_effort},
			max_output_tokens=self.max_completion_tokens,
			service_tier='default',
			text={
				'format': {
					'type': 'json_schema',
					'name': 'agent_output',
					'strict': True,
					'schema': SchemaOptimizer.create_optimized_json_schema(output_format),
				}
			},
		)
		if response.status != 'completed' or not response.output_text:
			raise RuntimeError('Judge did not deliver a completed structured response')
		usage = response.usage
		return ChatInvokeCompletion(
			completion=output_format.model_validate_json(response.output_text),
			stop_reason='stop',
			usage=ChatInvokeUsage(
				prompt_tokens=usage.input_tokens,
				prompt_cached_tokens=usage.input_tokens_details.cached_tokens,
				prompt_cache_creation_tokens=getattr(usage.input_tokens_details, 'cache_write_tokens', None),
				prompt_image_tokens=None,
				completion_tokens=usage.output_tokens,
				total_tokens=usage.total_tokens,
			)
			if usage
			else None,
		)


class MeteredTransport(httpx.AsyncBaseTransport):
	"""Capture every physical HTTP attempt without storing authorization or request bodies."""

	def __init__(self, role: str, requests: list[RequestUsage], ledger: Path | None = None):
		self.role = role
		self.requests = requests
		self.ledger = ledger
		self.inner: httpx.AsyncBaseTransport = httpx.AsyncHTTPTransport()

	async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
		start = time.monotonic()
		payload = json.loads(request.content)
		row = {
			'role': self.role,
			'model': payload.get('model', 'unknown'),
			'endpoint': request.url.path.removeprefix('/v1/'),
			'status': 'error',
		}
		try:
			response = await self.inner.handle_async_request(request)
			await response.aread()
			row['http_status'] = response.status_code
			row['request_id'] = response.headers.get('x-request-id')
			if response.status_code == 200:
				body = response.json()
				row.update(status='completed', raw_usage=body.get('usage'), service_tier=body.get('service_tier'))
				row['response_status'] = body.get('status')
				row['incomplete_reason'] = (body.get('incomplete_details') or {}).get('reason')
				row['usage'] = TokenUsage.from_api(body.get('usage') or {})
			return response
		except (TimeoutError, httpx.TimeoutException, asyncio.CancelledError):
			row['status'] = 'timeout'
			raise
		finally:
			row['latency_seconds'] = time.monotonic() - start
			event = RequestUsage.model_validate(row)
			self.requests.append(event)
			if self.ledger:
				with self.ledger.open('a') as stream:
					stream.write(event.model_dump_json() + '\n')
					stream.flush()
					os.fsync(stream.fileno())

	async def aclose(self) -> None:
		await self.inner.aclose()


async def judge_collected_trace(
	task: dict[str, Any], trace: Trace, judge: Any, judge_trace: Any, artifact_dir: Path
) -> dict[str, Any]:
	"""Reject verified collection failures before invoking a semantic grader."""
	if trace.collection_gaps:
		return {
			'assessment_status': 'collection_failure',
			'semantic_score': None,
			'collection_gaps': [g.model_dump() for g in trace.collection_gaps],
		}
	return await judge_trace(task, trace.model_dump(), judge, artifact_dir=artifact_dir)


def task_for(case: CalibrationCase, url: str) -> dict[str, Any]:
	"""Calibration rubric describes only deliverables, not a browser action recipe."""
	if case.kind in ('file', 'collector-loss'):
		return {
			'task_id': case.id,
			'confirmed_task': f'Open {url} and export all three plans to plans.csv with plan, people and price.',
			'website': url,
			'weights': {'SOLO': 30, 'TEAM': 30, 'BUSINESS': 40},
			'rubric': 'SOLO: The delivered CSV contains Solo,1,10. TEAM: It contains Team,3,24. '
			'BUSINESS: It contains Business,10,80. Each item requires the actual delivered file row. '
			'A source page or a claim of file creation is insufficient. Missing agent work is violated.',
		}
	return {
		'task_id': case.id,
		'confirmed_task': f'Open {url}, enter Alice and Tokyo, choose the cheapest plan for at least 3 people, and save.',
		'website': url,
		'weights': {'NAME': 30, 'CITY': 30, 'PLAN': 40},
		'rubric': 'NAME: A persisted subscription exists and its name is Alice. CITY: A persisted subscription exists and its city is Tokyo. '
		'PLAN: A persisted subscription exists and its plan is team (3 people for $24; Solo is 1 person for $10; Business is 10 for $80). '
		'Each item requires a persisted record. Filled controls, selected plans, final claims, and honest block reports do not satisfy saving.',
	}


async def collect_controls(output: Path) -> tuple[dict[str, Trace], dict[str, dict[str, Any]], dict[str, Any]]:
	"""Native UI replay plus an independent server oracle exercises evidence collection."""
	from browser_use.actor.page import Page
	from browser_use.browser.profile import BrowserProfile, ViewportSize
	from browser_use.browser.session import BrowserSession
	from browser_use.fast import CompletionCheck, Job
	from browser_use.fast.driver import BrowserDriver

	server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
	worker = threading.Thread(target=server.serve_forever, daemon=True)
	worker.start()
	base = f'http://127.0.0.1:{server.server_port}'
	browser = BrowserSession(
		browser_profile=BrowserProfile(
			headless=True,
			user_data_dir=None,
			executable_path=CHROME,
			keep_alive=True,
			enable_default_extensions=False,
			captcha_solver=False,
			viewport=ViewportSize(width=1280, height=900),
		)
	)
	recorder = Recorder()
	start = time.monotonic()
	traces: dict[str, Trace] = {}
	tasks = {}
	oracles = {}
	try:
		with recorder.span('browser_startup'):
			await browser.start()
		assert browser.agent_focus_target_id is not None
		page = Page(browser, browser.agent_focus_target_id)
		driver = BrowserDriver(page, browser.agent_focus_target_id)
		job = Job(id='collection', goal='Observe', checks=[CompletionCheck(id='pending', kind='javascript', value='false')])
		for case in CASES:
			url = f'{base}/run/{case.id}'
			tasks[case.id] = task_for(case, url)
			if case.kind == 'blocked':
				traces[case.id] = Trace(
					final_result=case.claim,
					agent_steps=[json.dumps({'op': 'goto', 'result': 'HTTP 403 forbidden'})],
					screenshots_b64=[],
					screenshot_steps=[],
				)
				continue
			events = []
			with recorder.span('navigation'):
				await page.goto(url)
			events.append(json.dumps({'op': 'goto', 'url': url}))
			if case.state:
				with recorder.span('native_execution'):
					for selector, value in (('#name', case.state['name']), ('#city', case.state['city'])):
						elements = await page.get_elements_by_css_selector(selector)
						await elements[0].fill(value)
						events.append(json.dumps({'op': 'fill', 'selector': selector, 'value': value}))
					elements = await page.get_elements_by_css_selector('#' + case.state['plan'])
					await elements[0].click()
					events.append(json.dumps({'op': 'click', 'selector': '#' + case.state['plan']}))
					if case.save:
						elements = await page.get_elements_by_css_selector('#save')
						await elements[0].click()
						events.append(json.dumps({'op': 'click', 'selector': '#save'}))
			with recorder.span('explicit_wait'):
				await asyncio.sleep(0.2)
			with recorder.span('observation'):
				observation = await driver.observe(job)
				events.append(json.dumps({'op': 'read', 'result': observation.model_dump(exclude={'screenshot'})}))
			with recorder.span('verification'):
				async with httpx.AsyncClient() as client:
					state = (await client.get(url + '/state')).json()
				persisted = state.get('state')
				correct = persisted == {'name': 'Alice', 'city': 'Tokyo', 'plan': 'team'} and state['saves'] == 1
				if case.kind == 'form':
					assert correct == case.oracle, f'Bad calibration replay: {case.id}'
				oracles[case.id] = {'saved': persisted, 'save_count': state['saves'], 'correct': correct}
				events.append(json.dumps({'harness_saved_state': persisted, 'save_count': state['saves']}))
			with recorder.span('evidence_capture'):
				image = await page.screenshot()
				image_step = len(events)
				(output / f'{case.id}.png').write_bytes(base64.b64decode(image))
			artifact_text = ''
			gaps = []
			if case.artifact:
				with recorder.span('artifact_save'):
					artifact = output / case.id / 'plans.csv'
					artifact.parent.mkdir()
					artifact.write_text(case.artifact)
					receipt = artifact_receipt(artifact)
					if case.kind == 'collector-loss':
						artifact.rename(artifact.with_suffix('.csv.lost'))
					content, gap = collect_artifact(receipt)
					if gap:
						gaps.append(gap)
					else:
						artifact_text = f'<file path="plans.csv">\n{content}\n</file>'
				events.append(
					json.dumps(
						{
							'op': 'write_file',
							'path': 'plans.csv',
							'bytes': receipt.size,
							'sha256': receipt.sha256,
							'success': True,
						}
					)
				)
			traces[case.id] = Trace(
				final_result=case.claim,
				agent_steps=events,
				screenshots_b64=[image],
				screenshot_steps=[image_step],
				output_files_text=artifact_text,
				evidence_notes=['Calibration replay with Browser Use Actor; server-owned saved state collected independently.']
				+ [
					f'Harness verified collection failure: {gap.artifact} ({gap.reason}). Delivered contents cannot be inspected.'
					for gap in gaps
				],
				collection_gaps=gaps,
			)
		end = time.monotonic()
		timing = summarize_timings(recorder.spans, start, end)
		return (
			traces,
			tasks,
			{'timings': timing.model_dump(), 'spans': [s.model_dump() for s in recorder.spans], 'oracles': oracles},
		)
	finally:
		await browser.kill()
		server.shutdown()
		server.server_close()


async def main(args: argparse.Namespace) -> None:
	from browser_use.fast import DecisionClient
	from browser_use.fast.views import Choice, ChoiceQuestion

	output = args.output.resolve()
	output.mkdir(parents=True, exist_ok=False)
	credential_source = configure_project_environment(REPO / '.env.local')
	upstream = args.upstream.resolve()
	sys.path.insert(0, str(upstream))
	# Import only the public grader. Never call load_tasks: unused task bodies stay sealed.
	evaluation = importlib.import_module('evaluation')
	create_judge, judge_config, judge_trace = evaluation.create_judge, evaluation.judge_config, evaluation.judge_trace

	sources = [Path(__file__), Path(__file__).with_name('evaluation_audit.py'), Path(__file__).with_name('fixture.py')]
	sources += sorted((REPO / 'browser_use/fast').glob('*.py'))
	sources += [upstream / name for name in ('evaluation.py', 'findings_judge.py', 'screenshot_evidence.py')]
	requests: list[RequestUsage] = []
	ledger = output / 'http-attempts.jsonl'
	judge_client = httpx.AsyncClient(transport=MeteredTransport('judge', requests, ledger))
	judge = ResponsesJudge(create_judge(), judge_client)
	prices = [
		Price(model='gpt-6-luna', endpoint='decisions', input=0.1, cached=0, cache_write=0, output=0),
		Price(model='gpt-6.1-sol', endpoint='responses', input=2, cached=0.1, cache_write=2.5, output=10),
		Price(model='gpt-5.6-luna', endpoint='responses', input=0.2, cached=0.02, cache_write=0.25, output=1.2),
	]
	protocol_path = output / 'protocol.json'
	freeze_protocol(
		protocol_path,
		{
			'phase': 'calibration',
			'credential_source': credential_source.model_dump(),
			'created_utc': datetime.now(UTC).isoformat(),
			'cases': [c.model_dump() for c in CASES],
			'repetitions': 2,
			'expected_semantic_judge_calls': 18,
			'expected_collection_admission_checks': 2,
			'judge': judge_config('BU_Bench_V2', judge),
			'judge_endpoint': 'responses; same public prompt, images, optimized output schema, model and effort',
			'prices': [p.model_dump() for p in prices],
			'price_sources': [
				'https://developers.openai.com/api/docs/pricing',
				'https://developers.openai.com/api/docs/guides/decisions#pricing-and-availability',
			],
			'criteria': 'Exact expected item credit and assessability on every control, zero false completions, full attempt usage or cost marked unavailable.',
			'controls_scope': 'Exposed deterministic fixture and explicit mutations; no generic-site capability or adoption claim.',
			'block_and_collector_loss': 'Blocked is synthetic. Collector loss deliberately removes an actually delivered file after a harness-owned receipt.',
			'collection_policy': 'A harness-verified collection gap is unscored and rejected before semantic grading. No agent-quality zero or model judgement is invented for missing collected evidence.',
			'peek_policy': 'Write raw results during the batch, inspect scores only after every frozen case/repetition finishes. No retries chosen from scores.',
			'acceptance_status': 'not_started; unused encrypted task dataset is not loaded by this program',
		},
		sources,
	)
	print(json.dumps({'event': 'calibration_frozen', 'cases': len(CASES), 'repetitions': 2}), flush=True)
	assert_frozen(protocol_path, sources)
	traces, tasks, collection = await collect_controls(output)
	(output / 'collection.json').write_text(json.dumps(collection, indent=2))
	for case in CASES:
		(output / f'{case.id}-trace.json').write_text(traces[case.id].model_dump_json(indent=2))
	# Probe real API usage, rather than inferring spend from a Codex subscription.
	assert_frozen(protocol_path, sources)
	parent_http = httpx.AsyncClient(transport=MeteredTransport('parent', requests, ledger))
	parent = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], http_client=parent_http, max_retries=0, timeout=60)
	try:
		response = await parent.responses.create(
			model='gpt-6.1-sol', input='Reply with OK.', reasoning={'effort': 'xhigh'}, max_output_tokens=256
		)
		(output / 'parent-probe.json').write_text(
			json.dumps({'status': response.status, 'usage': response.usage.model_dump() if response.usage else None}, indent=2)
		)
	except Exception as error:
		(output / 'aborted.json').write_text(
			json.dumps(
				{
					'phase': 'calibration',
					'status': 'aborted_before_grading',
					'error_type': type(error).__name__,
					'http_attempts': len(requests),
					'usage_missing_attempts': sum(r.usage.input_tokens is None for r in requests),
					'grader_checks_executed': 0,
					'unused_tasks_loaded': 0,
					'adoption_evaluation_allowed': False,
				},
				indent=2,
			)
			+ '\n'
		)
		await judge_client.aclose()
		raise
	finally:
		await parent.close()
	assert_frozen(protocol_path, sources)
	decisions = DecisionClient(transport=MeteredTransport('decisions', requests, ledger))
	try:
		await decisions.choose(
			{'instruction': 'Select the value OK.'},
			ChoiceQuestion(
				name='probe',
				instructions='Choose OK.',
				choices=[
					Choice(value='OK', description='OK'),
					Choice(value='NO', description='NO'),
				],
			),
			deadline=time.monotonic() + 30,
		)
	finally:
		await decisions.close()
	# Keep score output sealed until the entire batch is complete.
	raw_results = []
	try:
		for repetition in range(2):
			for case in CASES:
				assert_frozen(protocol_path, sources)
				print(json.dumps({'event': 'judge_start', 'case': case.id, 'repetition': repetition}), flush=True)
				try:
					result = await judge_collected_trace(
						tasks[case.id],
						traces[case.id],
						judge,
						judge_trace,
						output / 'judge' / f'{repetition}-{case.id}',
					)
					row = {'case': case.id, 'repetition': repetition, 'result': result}
				except Exception as error:
					row = {'case': case.id, 'repetition': repetition, 'error_type': type(error).__name__}
				raw_results.append(row)
				(output / 'raw-results.json').write_text(json.dumps(raw_results, indent=2))
				(output / 'request-usage.json').write_text(json.dumps([r.model_dump() for r in requests], indent=2))
	finally:
		await judge_client.aclose()
	assert_frozen(protocol_path, sources)
	checks = []
	for row in raw_results:
		case = next(c for c in CASES if c.id == row['case'])
		if 'error_type' in row:
			checks.append({**row, 'passed': False})
			continue
		if row['result'].get('assessment_status') == 'collection_failure':
			checks.append(
				{
					'case': case.id,
					'repetition': row['repetition'],
					'passed': not case.expected_assessable and bool(traces[case.id].collection_gaps),
					'assessment_status': 'collection_failure',
					'score': None,
					'complete': False,
					'assessable': False,
					'raw_judge_called': False,
					'reasons': ['collection_gap'],
				}
			)
			continue
		semantic = validate_judgement(row['result'], tasks[case.id]['weights'], oracle=case.oracle)
		verdict = validate_judgement(
			row['result'], tasks[case.id]['weights'], oracle=case.oracle, collection_gaps=traces[case.id].collection_gaps
		)
		met = [f['item'] for f in row['result']['judgement']['findings'] if f['status'] == 'met']
		passed = set(met) == set(case.expected_met) and verdict.assessable == case.expected_assessable
		checks.append(
			{
				'case': case.id,
				'repetition': row['repetition'],
				'passed': passed,
				'met': met,
				'expected_met': case.expected_met,
				'raw_judge_assessable': semantic.assessable,
				'raw_judge_called': True,
				'raw_judge_matches_control': set(met) == set(case.expected_met)
				and semantic.assessable == case.expected_assessable,
				**verdict.model_dump(),
			}
		)
	costs = {
		role: summarize_cost([r for r in requests if r.role == role], prices).model_dump()
		for role in ('parent', 'decisions', 'judge')
	}
	passed = all(c['passed'] for c in checks)
	summary = {
		'phase': 'calibration',
		'grader_checks_passed': sum(c['passed'] for c in checks),
		'grader_checks_total': len(checks),
		'grader_calibrated_for_controls': passed,
		'raw_judge_checks_passed': sum(c.get('raw_judge_matches_control', False) for c in checks),
		'raw_judge_checks_total': sum(c.get('raw_judge_called', False) for c in checks),
		'collection_admission_checks': sum(c.get('assessment_status') == 'collection_failure' for c in checks),
		'checks': checks,
		'collection_timings': collection['timings'],
		'http_attempts': len(requests),
		'usage_missing_attempts': sum(r.usage.input_tokens is None for r in requests),
		'costs': costs,
		'full_monetary_accounting_ready': all(c['complete'] for c in costs.values()),
		'development_status': 'not_started',
		'acceptance_status': 'not_started',
		'unused_tasks_loaded': 0,
		'adoption_evaluation_allowed': False,
		'limitations': [
			'Calibration is fixture-based, not generalization evidence.',
			'Existing Codex CLI timeout usage and subscription monetary basis remain unavailable.',
			'Missing cache-write billing fields are unknown; known_usd is only the complete-request subtotal.',
			'Live browser replays cover collection spans; parent and Decisions probes are API accounting probes, not end-to-end agent trials.',
		],
	}
	(output / 'summary.json').write_text(json.dumps(summary, indent=2))
	print(
		json.dumps(
			{
				'event': 'calibration_complete',
				'passed': passed,
				'checks': len(checks),
				'cost_complete': summary['full_monetary_accounting_ready'],
			}
		),
		flush=True,
	)


async def verify_recorded_controls(recording: Path, upstream: Path, output: Path) -> None:
	"""Exercise public grader plumbing with deterministic findings, not a model substitute."""
	from types import SimpleNamespace

	sys.path.insert(0, str(upstream.resolve()))
	evaluation = importlib.import_module('evaluation')
	findings = importlib.import_module('findings_judge')
	output.mkdir(parents=True, exist_ok=False)
	collection = json.loads((recording / 'collection.json').read_text())
	protocol = json.loads((recording / 'protocol.json').read_text())['protocol']
	sources = [Path(__file__), Path(__file__).with_name('evaluation_audit.py')]
	sources += [upstream / name for name in ('evaluation.py', 'findings_judge.py', 'screenshot_evidence.py')]
	sources += [recording / 'protocol.json', recording / 'collection.json']
	sources += sorted(recording.glob('*-trace.json')) + sorted(recording.glob('*/plans.csv'))
	freeze_protocol(
		output / 'protocol.json',
		{
			'phase': 'offline_adapter',
			'recording': str(recording.resolve()),
			'cases': protocol['cases'],
			'criteria': 'Every recorded control matches independent saved-state or CSV oracle; malformed coverage rejected; canary zeroes score.',
			'semantic_llm_calibration': False,
			'unused_tasks_loaded': 0,
		},
		sources,
	)
	results = []
	for raw_case in protocol['cases']:
		case = CalibrationCase.model_validate(raw_case)
		trace = Trace.model_validate_json((recording / f'{case.id}-trace.json').read_text())
		task = task_for(case, 'http://calibration.invalid/')
		met = []
		if case.kind == 'form':
			saved = collection['oracles'][case.id]['saved']
			if saved:
				for item, field, value in (('NAME', 'name', 'Alice'), ('CITY', 'city', 'Tokyo'), ('PLAN', 'plan', 'team')):
					if saved.get(field) == value:
						met.append(item)
		elif case.kind == 'file':
			rows = list(csv.DictReader(io.StringIO((recording / case.id / 'plans.csv').read_text())))
			for item, plan, people, price in (
				('SOLO', 'Solo', '1', '10'),
				('TEAM', 'Team', '3', '24'),
				('BUSINESS', 'Business', '10', '80'),
			):
				if {'plan': plan, 'people': people, 'price': price} in rows:
					met.append(item)
		entries = []
		for item in task['weights']:
			lost = case.kind == 'collector-loss'
			entries.append(
				{
					'item': item,
					'evidence': 'Deterministic control oracle; this is a scripted adapter test, not an LLM judgement.',
					'status': 'not_assessable' if lost else ('met' if item in met else 'violated'),
					'not_assessable_reason': 'missing_evidence' if lost else None,
				}
			)
		response = findings.findings_result_model(tuple(task['weights'])).model_validate(
			{
				'agent_task_reading': 'Calibration replay of a fixed browser control.',
				'findings': entries,
				'observations': [],
				'infra_error': False,
				'pii_present': False,
				'reward_hacking_suspected': False,
				'flag_notes': None,
			}
		)

		class ScriptedJudge:
			async def ainvoke(self, messages, output_format):
				return SimpleNamespace(completion=output_format.model_validate(response), stop_reason='stop')

		result = await evaluation.judge_trace(task, trace.model_dump(), ScriptedJudge(), artifact_dir=output / case.id)
		verdict = validate_judgement(result, task['weights'], oracle=case.oracle)
		passed = set(met) == set(case.expected_met) and verdict.assessable == case.expected_assessable
		results.append({'case': case.id, 'passed': passed, 'met': met, **verdict.model_dump()})
	# Public adapter must also reject incomplete or duplicated item coverage.
	coverage_checks = 0
	for items in (['NAME', 'CITY'], ['NAME', 'NAME', 'PLAN'], ['NAME', 'CITY', 'UNKNOWN']):
		malformed = SimpleNamespace(findings=[SimpleNamespace(item=i) for i in items])
		try:
			evaluation._validate_findings_coverage(malformed, ('NAME', 'CITY', 'PLAN'))
		except ValueError:
			coverage_checks += 1
		else:
			raise AssertionError('Public judge accepted malformed coverage')
	canary = findings.score(
		{'weights': {'NAME': 100}, 'canary': 'PRIVATE_CANARY'},
		SimpleNamespace(findings=[SimpleNamespace(item='NAME', status='met')], reward_hacking_suspected=False),
		agent_texts=['PRIVATE_CANARY'],
	)
	assert canary['score'] == 0 and not canary['verdict']
	summary = {
		'type': 'offline_public_adapter_check_with_scripted_findings',
		'checks': results,
		'control_checks_passed': sum(r['passed'] for r in results),
		'control_checks_total': len(results),
		'coverage_rejections': coverage_checks,
		'canary_tripwire_passed': True,
		'semantic_llm_calibration_executed': False,
		'adoption_evaluation_allowed': False,
		'unused_tasks_loaded': 0,
		'source_hashes': source_hashes(sources),
	}
	assert_frozen(output / 'protocol.json', sources)
	(output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
	assert all(r['passed'] for r in results)
	print(
		json.dumps(
			{
				'event': 'offline_adapter_verified',
				'controls': len(results),
				'malformed_coverage_rejected': coverage_checks,
				'live_judge_verified': False,
			}
		)
	)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--upstream', type=Path, default=Path('/private/tmp/bu-fast-upstream-bench-v2.1.1'))
	parser.add_argument(
		'--verify-recording', type=Path, help='Offline public-adapter checks on existing exposed calibration evidence.'
	)
	arguments = parser.parse_args()
	if arguments.verify_recording:
		asyncio.run(verify_recorded_controls(arguments.verify_recording, arguments.upstream, arguments.output))
	else:
		asyncio.run(main(arguments))
