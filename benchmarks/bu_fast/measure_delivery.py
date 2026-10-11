"""Prepare, then explicitly run the baseline versus structured-delivery development comparison."""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.bu_fast.evaluation_audit import assert_frozen, freeze_protocol
from benchmarks.bu_fast.measure_architectures import runtime_fingerprint
from benchmarks.bu_fast.measure_goal_loop import ExperimentSettings, execute_trial, grade, grading_task, load_tasks, tools
from benchmarks.bu_fast.validate_evaluation import CHROME, REPO, configure_project_environment
from browser_use.fast.session import POLICY

BASE_COMMIT = '90fb6fa85ab51cd1f2c6cff97e2d6381ed23fcf4'


class Cell(BaseModel):
	"""A fresh isolated run; recovery stays inside this cell's original budget."""

	variant: Literal['baseline', 'structured-delivery']
	task: str
	repeat: int


class DeliveryProtocol(BaseModel):
	"""Conditions frozen locally without credentials or model requests."""

	model_config = ConfigDict(extra='forbid')
	base_commit: str = BASE_COMMIT
	phase: Literal['development-screening'] = 'development-screening'
	checkout: Path
	upstream: Path
	env_file: Path
	tasks: list[dict]
	order: list[Cell]
	runtime: dict
	conditions: dict = Field(
		default_factory=lambda: {
			'parent_model': 'gpt-6.1-sol',
			'effort': 'xhigh',
			'decisions_model': 'gpt-6-luna',
			'service_tier': 'default',
			'max_output_tokens': 8192,
			'deadline_seconds': 240,
			'parent_calls': 12,
			'decisions_calls': 60,
			'viewport': [1280, 900],
			'headless': True,
			'extensions': False,
			'fresh_profile': True,
			'normal_A_calls': 'Evidence selection then structured final delivery; extra retrieval and repair are counted without forced success.',
			'initial_evidence': 'Same baseline run summary and index, no additional body evidence automatically attached.',
			'quality': 'Separate operation, delivery, initial delegation and final delegation grading. finished/done never implies success.',
			'judge_evidence': {
				'text_limit': sys.maxsize,
				'images': 500,
				'encoded_image_bytes': 100_000_000,
				'source_indices': 'all',
				'overflow': 'unassessable, complete=null',
				'override': 'local modules restored in finally',
			},
			'cost': 'Unknown until pinned prices are supplied; request usage and latency are retained.',
			'evidence_tokens': 'Unknown: provider does not attribute input tokens to individual tool returns. Full parent token usage retained.',
			'timing': 'wall_seconds from execute_trial entry before initialization through answer/artifacts/tool journal; post-run collection and cleanup excluded. seconds excludes settlement; delivery_seconds starts at first navigation/delegation. Decisions-loop time uses finally on timeout/error.',
			'repair': 'Done-run Decisions restart or native mutation batch after done/finish; read/extract/screenshot-only batches excluded. Requests and extra evidence retrieval retained.',
			'interpretation': 'Four exposed development tasks; no held-out or generic-site claim. Keep errors, difficult multi-call cases, incomplete and unassessable cases.',
			'peek_policy': 'Complete all frozen execution before grading; no tuning or hidden retries during a batch.',
		}
	)
	operation_policy: str = POLICY
	grading_specs: dict
	parent_tools: dict


def prepare(output: Path, upstream: Path, env_file: Path, repeats: int) -> Path:
	"""Freeze source and task bytes; neither browser nor API client is started."""
	if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != BASE_COMMIT:
		raise ValueError('This worktree must retain the specified baseline HEAD')
	if not Path(CHROME).is_file():
		raise ValueError('Chrome executable missing')
	if not 1 <= repeats <= 10:
		raise ValueError('Use 1..10 repeats')
	output.mkdir(parents=True, exist_ok=False)
	tasks = load_tasks()
	order = []
	for repeat in range(repeats):
		for index, task in enumerate(tasks):
			variants: list[Literal['baseline', 'structured-delivery']] = ['baseline', 'structured-delivery']
			if (repeat + index) % 2:
				variants.reverse()
			order.extend(Cell(variant=variant, task=task['id'], repeat=repeat) for variant in variants)
	protocol = DeliveryProtocol(
		checkout=REPO,
		upstream=upstream.resolve(),
		env_file=env_file.resolve(),
		tasks=tasks,
		order=order,
		runtime=runtime_fingerprint(),
		grading_specs={task['id']: {'operation': grading_task(task, True), 'delivery': grading_task(task)} for task in tasks},
		parent_tools={
			variant: tools('goal', ExperimentSettings(variant=variant)) for variant in ('baseline', 'structured-delivery')
		},
	)
	sources = [
		*sorted((REPO / 'browser_use/fast').glob('*.py')),
		*sorted((REPO / 'benchmarks/bu_fast').glob('*.py')),
		REPO / 'benchmarks/bu_fast/goal-tasks.json',
		REPO / 'browser_use/actor/page.py',
		REPO / 'browser_use/actor/element.py',
		*[upstream / name for name in ('evaluation.py', 'findings_judge.py', 'screenshot_evidence.py', 'judge.py')],
	]
	path = output / 'protocol.json'
	freeze_protocol(path, protocol.model_dump(mode='json'), sources)
	for source in sources:
		target = output / 'source-snapshot' / str(source.resolve()).lstrip('/')
		target.parent.mkdir(parents=True, exist_ok=True)
		target.write_bytes(source.read_bytes())
	return path


def check(path: Path) -> DeliveryProtocol:
	"""Validate frozen bytes and runtime without requesting models."""
	data = json.loads(path.read_text())['protocol']
	sources = [Path(name) for name in data.pop('source_hashes')]
	assert_frozen(path, sources)
	protocol = DeliveryProtocol.model_validate(data)
	if protocol.checkout != REPO or protocol.runtime != runtime_fingerprint():
		raise ValueError('Run from the frozen checkout and interpreter')
	return protocol


async def run(path: Path, grade_only: bool, trial_index: int | None = None):
	protocol = check(path)
	if trial_index is not None and not 0 <= trial_index < len(protocol.order):
		raise ValueError('Trial index outside frozen order')
	if grade_only:
		for index, cell in enumerate(protocol.order):
			directory = path.parent / f'{index:03d}-{cell.task}-{cell.variant}-r{cell.repeat}'
			if not (directory / 'result.json').is_file():
				raise ValueError('Finish the full frozen batch before grading; missing trial result')
		configure_project_environment(protocol.env_file)
		return await grade(path.parent, protocol.upstream)
	if trial_index is None and any(path.parent.glob('*/result.json')):
		raise ValueError('Batch already has results; use a new frozen output, never silently retry')
	indices = [trial_index] if trial_index is not None else list(range(len(protocol.order)))
	for index in indices:
		cell = protocol.order[index]
		check(path)
		task = next(task for task in protocol.tasks if task['id'] == cell.task)
		directory = path.parent / f'{index:03d}-{cell.task}-{cell.variant}-r{cell.repeat}'
		if directory.exists():
			raise ValueError('Trial directory already exists; never overwrite or retry')
		configure_project_environment(protocol.env_file)
		result = await execute_trial(task, 'goal', cell.repeat, directory, ExperimentSettings(variant=cell.variant))
		print(result.model_dump_json(), flush=True)
	check(path)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--output', type=Path)
	parser.add_argument('--protocol', type=Path)
	parser.add_argument('--upstream', type=Path, default=Path('/private/tmp/bu-fast-upstream-bench-v2.1.1'))
	parser.add_argument('--env-file', type=Path, default=Path('/Users/yuyoshimuta/dev/bu-fast/.env.local'))
	parser.add_argument('--repeats', type=int, default=3)
	parser.add_argument('--trial', type=int)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--run', action='store_true')
	mode.add_argument('--grade-only', action='store_true')
	args = parser.parse_args()
	if args.trial is not None and args.grade_only:
		parser.error('--trial cannot be combined with --grade-only')
	if not args.protocol and not args.output:
		parser.error('Specify --output to prepare or --protocol to check/run')
	path = (
		args.protocol.resolve() if args.protocol else prepare(args.output.resolve(), args.upstream, args.env_file, args.repeats)
	)
	protocol = check(path)
	if args.trial is not None and not 0 <= args.trial < len(protocol.order):
		parser.error('Trial index outside frozen order')
	if args.run or args.grade_only:
		asyncio.run(run(path, args.grade_only, args.trial))
	else:
		print(
			json.dumps(
				{
					'protocol': str(path),
					'trials': len(protocol.order),
					'selected_trial': args.trial,
					'selected_cell': protocol.order[args.trial].model_dump() if args.trial is not None else None,
					'model_requests': 0,
					'credentials_checked': False,
					'python': sys.executable,
					'ready': True,
				}
			)
		)


if __name__ == '__main__':
	main()
