"""Run a browser goal through exploration, evidence recovery and cited delivery."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError


def chrome_cdp_url() -> str:
	"""Discover the running system Chrome's permission-controlled CDP endpoint."""
	from browser_use.browser.chrome import get_chrome_profile_path

	profile = get_chrome_profile_path(None)
	if profile is None:
		raise ValueError('Cannot locate Chrome profile; provide --cdp-url')
	try:
		lines = (Path(profile) / 'DevToolsActivePort').read_text().splitlines()
	except OSError:
		raise ValueError(
			'Chrome connection unavailable; enable chrome://inspect/#remote-debugging or provide --cdp-url'
		) from None
	if (
		len(lines) != 2
		or not lines[0].isdigit()
		or not 1 <= int(lines[0]) <= 65535
		or not lines[1].startswith('/devtools/browser/')
	):
		raise ValueError('Invalid Chrome discovery file; provide --cdp-url')
	return f'ws://127.0.0.1:{lines[0]}{lines[1]}'


def main():
	from browser_use.fast.exploration import PurposeCondition, PurposePlan
	from browser_use.fast.workflow import WorkflowSettings, WorkflowTask, run_workflow

	defaults = WorkflowSettings()
	parser = argparse.ArgumentParser(description='Complete a browser goal with cited answers and requested files')
	parser.add_argument('--task', type=Path, help='WorkflowTask JSON with the whole goal and required fields')
	parser.add_argument('--url')
	parser.add_argument('--goal')
	parser.add_argument('--plan', type=Path, help='Optional PurposePlan with required fields')
	parser.add_argument('--input', action='append', default=[])
	parser.add_argument('--env-file', type=Path, default=Path('.env.local'))
	parser.add_argument('--timeout', type=int, default=defaults.deadline_seconds)
	parser.add_argument('--max-decisions', type=int, default=defaults.max_decisions)
	parser.add_argument('--parent-turns', type=int, default=defaults.parent_turn_budget)
	parser.add_argument('--output-dir', type=Path, default=Path('.bu-fast/workflows'))
	parser.add_argument('--log-dir', type=Path, help='Save detailed diagnostic logs under this directory')
	parser.add_argument('--executable-path', default=defaults.executable_path)
	connection = parser.add_mutually_exclusive_group()
	connection.add_argument('--connect-chrome', action='store_true', help='Connect to running system Chrome')
	connection.add_argument('--cdp-url', help='Existing browser CDP HTTP or WebSocket URL')
	parser.add_argument('--tab-id', help='Exact target ID when multiple tabs have the supplied --url')
	display = parser.add_mutually_exclusive_group()
	display.add_argument('--visible', dest='headless', action='store_const', const=False, default=defaults.headless)
	display.add_argument('--headless', dest='headless', action='store_const', const=True)
	args = parser.parse_args()

	try:
		if args.task:
			if args.url or args.goal or args.plan or args.input:
				parser.error('--task cannot be combined with --url, --goal, --plan or --input')
			task = WorkflowTask.model_validate_json(args.task.read_text())
		else:
			if not args.url or not args.goal:
				parser.error('Provide --task or both --url and --goal')
			plan = (
				PurposePlan.model_validate_json(args.plan.read_text())
				if args.plan
				else PurposePlan(
					conditions=[
						PurposeCondition(
							id='outcome',
							requirement='Fulfil the entire original goal, including every requested output and constraint.',
						)
					]
				)
			)
			task = WorkflowTask(
				id='goal',
				start_url=args.url,
				goal=args.goal,
				exact_input_values=list(dict.fromkeys([*args.input, *plan.inputs])),
				required_fields=plan.conditions,
			)
		settings = WorkflowSettings(
			deadline_seconds=args.timeout,
			parent_turn_budget=args.parent_turns,
			max_decisions=args.max_decisions,
			log_dir=args.log_dir,
			executable_path=args.executable_path,
			headless=args.headless,
			cdp_url=chrome_cdp_url() if args.connect_chrome else args.cdp_url,
			tab_id=args.tab_id,
		)
	except ValidationError as error:
		parser.error('; '.join(item['msg'] for item in error.errors(include_input=False, include_url=False)))
	except OSError as error:
		parser.error(str(error))
	except ValueError as error:
		parser.error(str(error))
	load_dotenv(args.env_file, override=True)
	directory = args.output_dir / uuid.uuid4().hex
	try:
		result = asyncio.run(run_workflow(task, directory, settings))
	except KeyboardInterrupt:
		from browser_use.fast.workflow import WorkflowResult

		result = WorkflowResult(task=task.id, directory=str(directory.resolve()), status='interrupted', error='Task interrupted')
	print(
		json.dumps(
			{
				'status': result.status,
				'directory': result.directory,
				'final_answer': result.delivery.final_answer if result.delivery else None,
				'error': result.error,
			},
			ensure_ascii=False,
		)
	)
	raise SystemExit(0 if result.status == 'complete' else 130 if result.status == 'interrupted' else 2)


if __name__ == '__main__':
	main()
