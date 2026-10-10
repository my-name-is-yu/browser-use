"""The declarative subscription benchmark contract used by the production CLI."""

import argparse
from pathlib import Path

from browser_use.fast import CompletionCheck, Job, Plan


def fixture_plan(url: str) -> Plan:
	"""Require saved application state rather than a click acknowledgement."""
	return Plan(
		goal='Save Alice in Tokyo with the cheapest plan supporting at least 3 people.',
		jobs=[
			Job(
				id='subscription',
				goal='Fill Name and City, choose the cheapest plan for at least 3 people, then save once.',
				start_url=url,
				inputs=['Alice', 'Tokyo'],
				checks=[
					CompletionCheck(
						id='saved',
						kind='javascript',
						value="window.saved?.name === 'Alice' && window.saved?.city === 'Tokyo' && window.saved?.plan === 'team'",
					)
				],
			)
		],
	)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--url', default='http://127.0.0.1:8765/run/fast')
	parser.add_argument('--output', type=Path, required=True)
	args = parser.parse_args()
	args.output.write_text(fixture_plan(args.url).model_dump_json(indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
	main()
