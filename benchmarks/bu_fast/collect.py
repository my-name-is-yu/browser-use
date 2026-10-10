"""Collect independent oracle results from real Codex and bu-fast runs."""

import argparse
import json
import statistics
from pathlib import Path
from urllib.request import urlopen


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--origin', default='http://127.0.0.1:8857')
	parser.add_argument('--fast-logs', nargs='+', type=Path, required=True)
	parser.add_argument('--standard-runs', nargs='+', required=True)
	parser.add_argument('--output', type=Path, default=Path('benchmarks/bu_fast/results/comparison.json'))
	args = parser.parse_args()
	rows = []
	for name in args.standard_runs:
		with urlopen(args.origin + '/run/' + name + '/state') as response:
			oracle = json.load(response)
		rows.append({'backend': 'codex_standard_browser_use', 'case': name, 'oracle': oracle})
	for log in args.fast_logs:
		result = next(json.loads(line) for line in reversed(log.read_text().splitlines()) if line.startswith('{"status"'))
		url = result['plan']['jobs'][0]['start_url']
		with urlopen(url + '/state') as response:
			oracle = json.load(response)
		rows.append(
			{
				'backend': 'bu_fast',
				'case': url.rsplit('/', 1)[-1],
				'status': result['status'],
				'run_id': result['run_id'],
				'journal': str(Path(result['directory']) / 'journal.jsonl'),
				'decisions': result['decisions'],
				'oracle': oracle,
			}
		)
	for row in rows:
		row['success'] = (
			row['oracle']['state'] == {'name': 'Alice', 'city': 'Tokyo', 'plan': 'team'} and row['oracle']['saves'] == 1
		)
	summary = {
		backend: {
			'runs': len(group),
			'successes': sum(r['success'] for r in group),
			'median_page_to_save_ms': statistics.median(r['oracle']['elapsed_ms'] for r in group),
		}
		for backend in ('codex_standard_browser_use', 'bu_fast')
		if (group := [r for r in rows if r['backend'] == backend])
	}
	artifact = {
		'task': 'Save Alice / Tokyo with cheapest plan for at least 3 people',
		'metric': 'Server page GET to first save POST',
		'baseline': 'Actual Codex standard Browser Use via cua_repl, batched known actions',
		'limitations': [
			'Small repeated fixture smoke test, not an adoption benchmark',
			'Same Codex operator in one conversation; repeated-task learning is not randomized',
			'Fast uses dedicated Chrome and standard uses Codex IAB; viewport and backend differ',
			'Browser startup and final verification are excluded from this server metric',
			'No controlled outer-model token/cost or failure-distribution measurement',
		],
		'summary': summary,
		'runs': rows,
	}
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
	print(json.dumps(summary))


if __name__ == '__main__':
	main()
