"""Run a browser goal through exploration, evidence recovery and cited delivery."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError


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
	parser.add_argument('--executable-path', default=defaults.executable_path)
	parser.add_argument('--visible', action='store_true')
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
			executable_path=args.executable_path,
			headless=not args.visible,
		)
	except ValidationError as error:
		parser.error('; '.join(item['msg'] for item in error.errors(include_input=False, include_url=False)))
	except OSError as error:
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
