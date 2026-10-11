"""Start a goal or supply text to the saved run on its existing browser tab."""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.fast import BrowserTask, DecisionRuntime, RunResult
from browser_use.fast.delivery import complete_run


async def execute(args):
	load_dotenv(args.env_file, override=True)
	resume = RunResult.model_validate_json(args.result.read_text()) if args.command == 'resume' else None
	if resume:
		if not resume.cdp_url or not resume.target_id:
			raise ValueError('Saved run has no browser connection')
		browser = BrowserSession(
			cdp_url=resume.cdp_url,
			browser_profile=BrowserProfile(keep_alive=True, enable_default_extensions=False, captcha_solver=False),
		)
	else:
		browser = None
	runtime = DecisionRuntime(
		browser,
		target_id=resume.target_id if resume else None,
		timeout=args.timeout,
		max_decisions=args.max_decisions,
		output_dir=args.output_dir,
		headless=not args.visible,
		executable_path=args.executable_path,
	)
	try:
		return await runtime.run(
			None if resume else BrowserTask(url=args.url, goal=args.goal, inputs=args.input),
			resume=resume,
			inputs=args.input if resume else None,
		)
	finally:
		await runtime.close()


def main():
	parser = argparse.ArgumentParser(description='Goal-driven Decisions browser loop')
	commands = parser.add_subparsers(dest='command', required=True)
	complete = commands.add_parser('complete', help='Save a structured Codex delivery using recorded evidence; no model calls')
	complete.add_argument('result', type=Path)
	complete.add_argument('--delivery', type=Path, required=True)
	complete.add_argument('--required-field', action='append', required=True)
	for name in ['start', 'resume']:
		command = commands.add_parser(name)
		if name == 'start':
			command.add_argument('--url', required=True)
			command.add_argument('--goal', required=True)
		else:
			command.add_argument('result', type=Path)
		command.add_argument('--input', action='append', default=[])
		command.add_argument('--env-file', type=Path, default=Path('.env.local'))
		command.add_argument('--timeout', type=float, default=180)
		command.add_argument('--max-decisions', type=int, default=60)
		command.add_argument('--output-dir', type=Path, default=Path('.bu-fast/runs'))
		command.add_argument('--visible', action='store_true')
		command.add_argument('--executable-path')
	args = parser.parse_args()
	if args.command == 'complete':
		receipt = complete_run(args.result, args.delivery, args.required_field)
		print(receipt.model_dump_json())
		raise SystemExit(0 if receipt.disposition == 'complete' else 2)
	result = asyncio.run(execute(args))

	print(json.dumps(result.summary(), ensure_ascii=False))
	raise SystemExit(0 if result.status == 'done' else 2)


if __name__ == '__main__':
	main()
