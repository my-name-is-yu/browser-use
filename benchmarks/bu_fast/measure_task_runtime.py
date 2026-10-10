"""Frozen, serial development screening of task contracts, jobs and native batches.

Use public tasks already explored in Phase 2; this is not held-out adoption
measurement. Raw HTTP bodies omit all authorization headers and API keys.
"""

import argparse
import asyncio
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from benchmarks.bu_fast import evaluate_parent_loop as harness
from benchmarks.bu_fast.evaluation_audit import assert_frozen, freeze_protocol, validate_judgement
from benchmarks.bu_fast.validate_evaluation import (
	REPO,
	MeteredTransport,
	ResponsesJudge,
	Trace,
	configure_project_environment,
	judge_collected_trace,
)
from browser_use.fast.views import Plan

TASK_IDS = ['docs-python-statistics', 'docs-mdn-array', 'github-cpython-bugs', 'settings-length']
ARMS = ['hybrid', 'task', 'native']


class AuditedTransport(MeteredTransport):
	"""Save exactly submitted public payloads before sending, without HTTP headers."""

	async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
		assert self.ledger
		root = self.ledger.parent / 'payloads' / self.role
		root.mkdir(parents=True, exist_ok=True)
		index = len(list(root.glob('*.request.json')))
		body = bytes(request.content)
		(root / f'{index:03d}.request.json').write_bytes(body)
		(root / f'{index:03d}.sha256').write_text(hashlib.sha256(body).hexdigest())
		response = await super().handle_async_request(request)
		(root / f'{index:03d}.response.json').write_bytes(response.content)
		return response


def tasks_from_public_contracts(path: Path) -> list[harness.DevelopmentTask]:
	"""Require the same output fields and evidence in every arm; never supply answers."""
	items = {item['id']: item for item in json.loads(path.read_text())}
	result = []
	for task_id in TASK_IDS:
		item = items[task_id]
		fields = item['required_fields']
		public = item['goal'] + '\nKnown exact input values: ' + json.dumps(item['exact_input_values'])
		public += '\nBefore finishing, write results.json with one record per required field (id, value, quote, source_url) and a cited table in report.md. Include issue direct links and ranks or setting/time attribution where requested. Do not omit requirements or fabricate evidence. Use the specified start URL as the first navigation.'
		result.append(
			harness.DevelopmentTask(
				id=task_id,
				instruction=public,
				website=item['start_url'],
				domain=item['allowed_domains'][0],
				template=item['family'],
				family=item['family'],
				weights={field['id']: 1 for field in fields},
				rubric=' '.join(
					field['id']
					+ ': '
					+ field['requirement']
					+ '. The delivered results.json and cited table must contain this field, with an actual observed quote and correct source URL supporting this exact subject and setting. Page arrival, prior knowledge and assertions are insufficient.'
					for field in fields
				)
				+ " Both files must have been written before task completion. Issue title, href and rank must belong to the same observed filtered list row. Input editing is not evidence that conversion output was updated; the before and after applied values/results must be supported separately. Judge against each run's actual observations, not a historical issue list. Inspect code/result examples and attribution in source evidence. Missing required files violates every item.",
			)
		)
	return result


def burden(directory: Path, result: dict[str, Any]) -> dict[str, Any]:
	"""Measure observable preparation, returns, retrievals and physical API usage."""
	exchanges_path = directory / 'tool-exchanges.jsonl'
	exchanges = [json.loads(line) for line in exchanges_path.read_text().splitlines()] if exchanges_path.exists() else []
	roles = {}
	for role in ('parent', 'decisions'):
		requests = [r for r in result['requests'] if r['role'] == role]
		roles[role] = {
			'requests': len(requests),
			'input_tokens': sum(r['usage']['input_tokens'] or 0 for r in requests),
			'output_tokens': sum(r['usage']['output_tokens'] or 0 for r in requests),
			'cached_tokens': sum(r['usage']['cached_tokens'] or 0 for r in requests),
			'usage_unknown_requests': sum(
				r['usage']['input_tokens'] is None or r['usage']['output_tokens'] is None for r in requests
			),
			'latency_seconds': sum(r['latency_seconds'] for r in requests),
		}
	delegations = [e for e in exchanges if e['name'] in ('delegate_job', 'delegate_task')]
	details = [e for e in delegations if e['name'] == 'delegate_task' and json.loads(e['arguments'])['operation'] == 'details']
	responses = sorted(directory.glob('parent-response-*.json'))
	phases = []
	for index, path in enumerate(responses):
		response = json.loads(path.read_text())
		calls = [o for o in response['output'] if o['type'] == 'function_call']
		tags = []
		if index == 0:
			tags.append('initial_planning')
		if not calls:
			tags.append('final_answer')
		for call in calls:
			if call['name'] == 'delegate_job':
				tags.append('job_contract')
			elif call['name'] == 'delegate_task':
				tags.append('task_' + json.loads(call['arguments'])['operation'])
			else:
				tags.append(call['name'])
		phases.append({'response': path.name, 'tags': sorted(set(tags)), 'usage': response.get('usage')})
	return {
		'api': roles,
		'delegation_calls': len(delegations) - len(details),
		'contract_and_patch_argument_bytes': sum(len(e['arguments'].encode()) for e in delegations if e not in details),
		'delegation_return_bytes': sum(
			len(json.dumps(e['result'], ensure_ascii=False).encode()) for e in delegations if e not in details
		),
		'detail_calls': len(details),
		'detail_return_bytes': sum(len(json.dumps(e['result'], ensure_ascii=False).encode()) for e in details),
		'all_tool_return_bytes': sum(len(json.dumps(e['result'], ensure_ascii=False).encode()) for e in exchanges),
		'tool_errors': sum(e['status'] != 'completed' for e in exchanges),
		'direct_browser_batches': sum(e['name'] == 'browser_batch' for e in exchanges),
		'parent_phases': phases,
		'interpretation': 'Phase tags describe observed output actions, not invisible internal thinking. Bytes are not tokens. Unknown usage is explicitly counted; token totals then are lower bounds.',
	}


async def main(args: argparse.Namespace) -> None:
	"""Freeze all conditions, run sequentially, then grade without modifying execution."""
	credential = configure_project_environment(REPO / '.env.local')
	output = args.output.resolve()
	output.mkdir(parents=True, exist_ok=False)
	tasks = tasks_from_public_contracts(args.tasks)
	order = []
	for index, task in enumerate(tasks):
		arms = ARMS[index % 3 :] + ARMS[: index % 3]
		order.extend({'task': task.id, 'arm': arm} for arm in arms)
	sources = [
		Path(__file__),
		Path(harness.__file__),
		args.tasks,
		REPO / 'benchmarks/bu_fast/evaluation_audit.py',
		REPO / 'benchmarks/bu_fast/validate_evaluation.py',
	]
	sources += sorted((REPO / 'browser_use/fast').glob('*.py'))
	sources += [REPO / 'browser_use/actor' / name for name in ('page.py', 'element.py')]
	sources += [args.upstream / name for name in ('evaluation.py', 'findings_judge.py', 'screenshot_evidence.py')]
	protocol = output / 'protocol.json'
	freeze_protocol(
		protocol,
		{
			'phase': 'development-screening',
			'credential_source': credential.model_dump(),
			'tasks': [t.model_dump() for t in tasks],
			'order': order,
			'concurrency': 1,
			'parent_model': 'gpt-6.1-sol',
			'parent_effort': 'xhigh',
			'decisions_model': 'gpt-6-luna',
			'deadline_seconds': 240,
			'parent_turn_limit': 12,
			'decisions_limit': 60,
			'detail_limit': 10,
			'parent_tools': {a: harness.browser_tools(a == 'hybrid', a == 'task') for a in ARMS},
			'parent_instructions': {
				a: harness.parent_instructions(a == 'hybrid', Plan.model_json_schema(), a == 'task').replace(
					'180 seconds total', '240 seconds total'
				)
				for a in ARMS
			},
			'first_browser_action': 'The specified public start URL; no initial correct paths/selectors/answers supplied.',
			'deliverables': 'Identical results.json and cited report.md; parent authors all contracts and final files.',
			'grading': 'Existing public BU Bench findings judge after the full batch; collection gaps are unscorable. Independently inspect cited sources and final artifacts after grading.',
			'peek_policy': 'No architecture/prompt/source changes during the batch. Preserve every error/timeout. No score-driven reruns.',
			'pricing': 'Not monetized: report actual tokens and requests, no unverified dollar rates.',
			'limits': 'One fresh run per cell, previously exposed tasks; API parent controller, not actual Codex app or ChatBrowserUse Agent. Native actions are batched, not forced per-click. This screens the thin implementation, not every feature in the Astra design.',
			'timing': 'wall starts before browser startup and includes initial planning, execution and final files/answer; settlement and cleanup recorded separately. delivery_seconds excludes post-deadline settlement.',
		},
		sources,
	)
	for source in sources:
		target = output / 'source-snapshot' / str(source.resolve()).lstrip('/')
		target.parent.mkdir(parents=True, exist_ok=True)
		target.write_bytes(source.read_bytes())
	harness.MeteredTransport = AuditedTransport
	results = []
	for index, cell in enumerate(order):
		assert_frozen(protocol, sources)
		print(json.dumps({'event': 'start', 'index': index, **cell}), flush=True)
		directory = output / f'{index:02d}-{cell["task"]}-{cell["arm"]}'
		task = next(t for t in tasks if t.id == cell['task'])
		result = (
			await harness.execute_trial(
				task, cell['arm'], directory, deadline_seconds=240, parent_turn_limit=12, max_decisions=60
			)
		).model_dump(mode='json')
		result['burden'] = burden(directory, result)
		result['delivery_seconds'] = result['timings']['wall_seconds'] - result['settlement_seconds']
		(directory / 'result.json').write_text(json.dumps({k: v for k, v in result.items() if k != 'trace'}, indent=2))
		results.append(result)
		print(
			json.dumps({'event': 'executed', 'index': index, 'status': result['status'], 'seconds': result['delivery_seconds']}),
			flush=True,
		)
	# Grade only when the frozen execution batch has ended.
	sys.path.insert(0, str(args.upstream))
	evaluation = importlib.import_module('evaluation')
	judge_requests = []
	judge_http = httpx.AsyncClient(transport=AuditedTransport('judge', judge_requests, output / 'judge-http-attempts.jsonl'))
	judge = ResponsesJudge(evaluation.create_judge(), judge_http)
	try:
		for index, result in enumerate(results):
			assert_frozen(protocol, sources)
			task = next(t for t in tasks if t.id == result['task'])
			grading_task = {
				'task_id': task.id,
				'confirmed_task': task.instruction,
				'website': task.website,
				'weights': task.weights,
				'rubric': task.rubric,
			}
			print(json.dumps({'event': 'grade', 'index': index}), flush=True)
			try:
				judgement = await judge_collected_trace(
					grading_task,
					Trace.model_validate(result['trace']),
					judge,
					evaluation.judge_trace,
					output / 'judge' / str(index),
				)
				result['judgement'] = judgement
				if judgement.get('assessment_status') == 'collection_failure':
					result['verdict'] = {'complete': False, 'assessable': False, 'score': None, 'reasons': ['collection_gap']}
				else:
					result['verdict'] = validate_judgement(
						judgement, task.weights, oracle=None, collection_gaps=result['trace']['collection_gaps']
					).model_dump()
				if result['status'] != 'finished':
					result['verdict']['complete'] = False
			except Exception as error:
				result['verdict'] = {
					'complete': False,
					'assessable': False,
					'score': None,
					'reasons': [f'{type(error).__name__}: {error}'],
				}
			(output / f'graded-{index:02d}.json').write_text(
				json.dumps({k: v for k, v in result.items() if k != 'trace'}, indent=2)
			)
	finally:
		await judge.client.close()
	assert_frozen(protocol, sources)
	(output / 'summary.json').write_text(
		json.dumps(
			{
				'trials': [{k: v for k, v in r.items() if k != 'trace'} for r in results],
				'judge_requests': [r.model_dump() for r in judge_requests],
				'generalization_or_adoption_supported': False,
			},
			indent=2,
		)
	)
	print(json.dumps({'event': 'done', 'output': str(output)}), flush=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument(
		'--tasks', type=Path, default=REPO / '.bu-fast/evaluations/decisions-map-20261010-phase2/live/tasks-frozen.json'
	)
	parser.add_argument('--upstream', type=Path, default=Path('/private/tmp/bu-fast-upstream-bench-v2.1.1'))
	asyncio.run(main(parser.parse_args()))
