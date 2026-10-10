"""Freeze and run exposed development tasks across isolated architecture worktrees."""

import argparse
import asyncio
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.bu_fast.measure_goal_loop import ExperimentSettings, execute_trial, grade, grading_task, load_tasks
from benchmarks.bu_fast.validate_evaluation import CHROME, REPO, configure_project_environment

VARIANTS = ('baseline', 'advice', 'experience', 'parallel', 'parallel-serial')
DEFAULT_WORKTREES = {
	'baseline': REPO,
	'advice': Path('/Users/yuyoshimuta/.codex/worktrees/2a4f/bu-fast'),
	'experience': Path('/Users/yuyoshimuta/.codex/worktrees/65e8/bu-fast'),
	'parallel': Path('/Users/yuyoshimuta/.codex/worktrees/30a1/bu-fast'),
	'parallel-serial': Path('/Users/yuyoshimuta/.codex/worktrees/30a1/bu-fast'),
}


class Cell(BaseModel):
	"""One fresh browser trial in a named source checkout."""

	variant: Literal['baseline', 'advice', 'experience', 'parallel', 'parallel-serial']
	task: str
	repeat: int = Field(ge=0)
	checkout: Path
	settings: ExperimentSettings


class Protocol(BaseModel):
	"""Source, task, settings and memo bytes frozen before the first request."""

	model_config = ConfigDict(extra='forbid')
	phase: Literal['development-screening'] = 'development-screening'
	tasks: list[dict]
	grading_specs: dict
	order: list[Cell]
	sources: dict[str, str]
	commits: dict[str, str]
	env_file: Path
	upstream: Path
	memo_preparation: list[dict]
	limits: dict = Field(
		default_factory=lambda: {
			'task_seconds': 240,
			'parent_requests_total': 12,
			'decisions_requests': 60,
			'parallel_parent_requests': 4,
			'max_advice_requests': 2,
			'models': {'parent': 'gpt-6.1-sol', 'effort': 'xhigh', 'decisions': 'gpt-6-luna'},
			'interpretation': 'API surrogate; exposed tasks, not held-out adoption evidence. Parallel vs serial is the concurrency control; queue admission can differ and must be reported.',
		}
	)


def digest(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(checkout: Path):
	return (
		sorted((checkout / 'browser_use/fast').glob('*.py'))
		+ sorted((checkout / 'benchmarks/bu_fast').glob('*.py'))
		+ [checkout / 'benchmarks/bu_fast/goal-tasks.json']
	)


def validate_frozen(protocol: Protocol):
	"""Fail before requesting models if any frozen source or memo changed."""
	for path, expected in protocol.sources.items():
		if digest(Path(path)) != expected:
			raise ValueError(f'Frozen bytes changed: {path}')


def prepare(args) -> Path:
	"""Preflight local runtimes and write a no-network development protocol."""
	output = args.output.resolve()
	output.mkdir(parents=True, exist_ok=True)
	path = output / 'protocol.json'
	if path.exists():
		raise ValueError('Use a fresh output directory; existing protocol cannot be overwritten')
	upstream = args.upstream.resolve()
	sources = {}
	commits = {}
	for variant, checkout in DEFAULT_WORKTREES.items():
		result = subprocess.run(
			[
				'uv',
				'run',
				'--no-sync',
				'python',
				'-c',
				'from pathlib import Path; import browser_use.fast.session as s; print(Path(s.__file__).resolve())',
			],
			cwd=checkout,
			capture_output=True,
			text=True,
			check=True,
		)
		if Path(result.stdout.strip()) != checkout / 'browser_use/fast/session.py':
			raise ValueError(f'Editable source mismatch for {variant}')
		if not (checkout / 'benchmarks/bu_fast/measure_architectures.py').exists():
			raise ValueError(f'Common measurement code missing in {checkout}')
		commits[variant] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
		for source in source_files(checkout):
			sources[str(source)] = digest(source)
	for name in ('evaluation.py', 'findings_judge.py', 'screenshot_evidence.py'):
		source = upstream / name
		sources[str(source)] = digest(source)
	if not Path(CHROME).is_file():
		raise ValueError('Configured Chrome executable missing')
	if not args.env_file.is_file():
		raise ValueError('Explicit environment file missing')
	configure_project_environment(args.env_file)
	sources[str(args.memo_selection.resolve())] = digest(args.memo_selection)
	selection = json.loads(args.memo_selection.read_text())
	if set(selection) != {t['id'] for t in load_tasks()}:
		raise ValueError('Freeze explicit memo IDs for every development task')
	# Validate through the experience checkout, without opening a browser or model request.
	subprocess.run(
		[
			'uv',
			'run',
			'--no-sync',
			'python',
			'-c',
			'import json,sys; from pathlib import Path; from browser_use.fast.experience import load_memos; [load_memos(Path(sys.argv[1]), ids) for ids in json.loads(Path(sys.argv[2]).read_text()).values()]',
			str(args.memo_store.resolve()),
			str(args.memo_selection.resolve()),
		],
		cwd=DEFAULT_WORKTREES['experience'],
		check=True,
	)
	memo_preparation = []
	for ids in selection.values():
		if not ids or len(ids) > 5 or len(set(ids)) != len(ids):
			raise ValueError('Each experience trial requires 1..5 distinct selected memo IDs')
		for memo_id in ids:
			if len(memo_id) != 32 or any(c not in '0123456789abcdef' for c in memo_id):
				raise ValueError('Invalid memo ID')
			memo_path = args.memo_store.resolve() / f'{memo_id}.json'
			sources[str(memo_path)] = digest(memo_path)
			memo = json.loads(memo_path.read_text())
			if memo['source_run_id'] in args.excluded_run_ids:
				raise ValueError('Evaluation-derived memo rejected')
			if memo['id'] != memo_id or memo.get('confirmed_by') != 'Codex':
				raise ValueError('Unconfirmed or mismatched memo')
			if memo_id not in {m['id'] for m in memo_preparation}:
				memo_preparation.append(
					{'id': memo_id, 'source_run_id': memo['source_run_id'], 'preparation': memo['preparation']}
				)
	order = []
	for repeat in range(args.repeats):
		for task_index, task in enumerate(load_tasks()):
			shift = (repeat + task_index) % len(VARIANTS)
			for variant in VARIANTS[shift:] + VARIANTS[:shift]:
				settings = ExperimentSettings(
					variant=variant,
					memo_store=args.memo_store.resolve() if variant == 'experience' else None,
					memo_ids=selection[task['id']] if variant == 'experience' else [],
				)
				order.append(
					Cell(variant=variant, task=task['id'], repeat=repeat, checkout=DEFAULT_WORKTREES[variant], settings=settings)
				)
	protocol = Protocol(
		tasks=load_tasks(),
		grading_specs={t['id']: {'operation': grading_task(t, True), 'delivery': grading_task(t)} for t in load_tasks()},
		order=order,
		sources=sources,
		commits=commits,
		env_file=args.env_file.resolve(),
		upstream=upstream,
		memo_preparation=memo_preparation,
	)
	for source in sources:
		target = output / 'source-snapshot' / source.lstrip('/')
		target.parent.mkdir(parents=True, exist_ok=True)
		target.write_bytes(Path(source).read_bytes())
	path.write_text(protocol.model_dump_json(indent=2))
	return path


async def trial(path: Path, index: int):
	protocol = Protocol.model_validate_json(path.read_text())
	validate_frozen(protocol)
	cell = protocol.order[index]
	if REPO != cell.checkout:
		raise ValueError('Trial must run from its own frozen worktree')
	configure_project_environment(protocol.env_file)
	task = next(t for t in protocol.tasks if t['id'] == cell.task)
	directory = path.parent / f'{index:03d}-{cell.task}-{cell.variant}-r{cell.repeat}'
	if directory.exists():
		raise ValueError('Trial already has evidence; never overwrite or silently retry')
	result = await execute_trial(task, 'goal', cell.repeat, directory, cell.settings)
	validate_frozen(protocol)
	print(result.model_dump_json(), flush=True)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--output', type=Path)
	parser.add_argument('--protocol', type=Path)
	parser.add_argument('--upstream', type=Path, default=Path('/private/tmp/bu-fast-upstream-bench-v2.1.1'))
	parser.add_argument('--env-file', type=Path, default=REPO / '.env.local')
	parser.add_argument('--memo-store', type=Path)
	parser.add_argument('--memo-selection', type=Path)
	parser.add_argument('--excluded-run-ids', nargs='*', default=[])
	parser.add_argument('--repeats', type=int, choices=range(1, 11), default=3)
	parser.add_argument('--trial', type=int)
	parser.add_argument('--run', action='store_true')
	parser.add_argument('--grade-only', action='store_true')
	args = parser.parse_args()
	if args.protocol:
		path = args.protocol.resolve()
	else:
		if not args.output or not args.memo_store or not args.memo_selection:
			parser.error('Preparation requires --output, --memo-store and --memo-selection')
		path = prepare(args)
	protocol = Protocol.model_validate_json(path.read_text())
	validate_frozen(protocol)
	if args.trial is not None:
		asyncio.run(trial(path, args.trial))
	elif args.grade_only:
		configure_project_environment(protocol.env_file)
		asyncio.run(grade(path.parent, protocol.upstream))
	elif args.run:
		for index, cell in enumerate(protocol.order):
			validate_frozen(protocol)
			subprocess.run(
				[
					'uv',
					'run',
					'--no-sync',
					'python',
					'-m',
					'benchmarks.bu_fast.measure_architectures',
					'--protocol',
					str(path),
					'--trial',
					str(index),
				],
				cwd=cell.checkout,
				check=True,
			)
	else:
		print(json.dumps({'protocol': str(path), 'trials': len(protocol.order), 'model_requests': 0}))


if __name__ == '__main__':
	main()
