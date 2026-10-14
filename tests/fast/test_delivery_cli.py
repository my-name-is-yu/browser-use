"""The default CLI preserves the goal and reports validated completion."""

import json
import sys

import pytest

from browser_use.fast import cli, workflow
from browser_use.fast.delivery import CompleteDelivery


@pytest.fixture
def invoke(tmp_path, monkeypatch, capsys):
	calls = []

	async def execute(task, directory, settings):
		calls.append((task, settings))
		directory.mkdir(parents=True)
		return workflow.WorkflowResult(
			task=task.id,
			directory=str(directory.resolve()),
			status=status[0],
			delivery=CompleteDelivery(
				disposition=status[0], browser_summary='', records=[], issues=[], final_answer='Observed result'
			)
			if status[0] in ('complete', 'incomplete')
			else None,
		)

	status = ['complete']

	monkeypatch.setattr(workflow, 'run_workflow', execute)
	monkeypatch.setattr(cli, 'load_dotenv', lambda *a, **kw: None)

	def run(arguments):
		monkeypatch.setattr(sys, 'argv', ['bu-fast', *arguments, '--output-dir', str(tmp_path / 'outputs')])
		with pytest.raises(SystemExit) as result:
			cli.main()
		output = capsys.readouterr()
		return result.value.code, json.loads(output.out) if output.out else None

	return run, calls, status


def test_run_preserves_whole_goal_and_configured_budget(invoke):
	run, calls, _ = invoke
	goal = 'Keep every requested explanation and example. ' * 100
	code, result = run(['--url', 'https://example.test/', '--goal', goal, '--input', 'known', '--parent-turns', '3'])
	assert code == 0 and result['final_answer'] == 'Observed result'
	assert calls[0][0].goal == goal and calls[0][0].exact_input_values == ['known']
	assert calls[0][1].parent_turn_budget == 3
	assert calls[0][1].deadline_seconds == 180 and calls[0][1].max_decisions == 90


def test_run_accepts_task_and_rejects_conflicting_arguments(tmp_path, invoke):
	run, calls, _ = invoke
	task = workflow.WorkflowTask(
		id='task',
		start_url='https://example.test/',
		goal='Read',
		required_fields=[{'id': 'answer', 'requirement': 'Observed answer'}],
	)
	path = tmp_path / 'task.json'
	path.write_text(task.model_dump_json())
	assert run(['--task', str(path)])[0] == 0
	assert calls[0][0] == task
	assert run(['--task', str(path), '--goal', 'replacement'])[0] == 2
	assert len(calls) == 1


def test_run_plan_keeps_requirements_and_deduplicates_inputs(tmp_path, invoke):
	run, calls, _ = invoke
	path = tmp_path / 'plan.json'
	path.write_text(json.dumps({'conditions': [{'id': 'a', 'requirement': 'Exact source'}], 'inputs': ['known', 'other']}))
	assert run(['--url', 'https://example.test/', '--goal', 'Read', '--plan', str(path), '--input', 'known'])[0] == 0
	assert calls[0][0].exact_input_values == ['known', 'other']
	assert calls[0][0].required_fields[0].id == 'a'


@pytest.mark.parametrize('outcome', ['incomplete', 'error', 'timeout', 'interrupted'])
def test_only_complete_outcome_exits_successfully(invoke, outcome):
	run, _, status = invoke
	status[0] = outcome
	code, result = run(['--url', 'https://example.test/', '--goal', 'Read'])
	assert code == (130 if outcome == 'interrupted' else 2) and result['status'] == outcome


@pytest.mark.parametrize(
	'command', ['run', 'start', 'resume', 'complete', 'evidence', 'review-delivery', 'confirm-fields', 'cited-draft']
)
def test_removed_commands_are_rejected(invoke, command):
	run, calls, _ = invoke
	assert run([command])[0] == 2
	assert not calls


def test_cli_and_public_python_api_share_defaults(invoke):
	import browser_use.fast as public

	run, calls, _ = invoke
	assert run(['--url', 'https://example.test/', '--goal', 'Read'])[0] == 0
	assert calls[0][1] == public.WorkflowSettings()


def test_cli_enables_diagnostic_logs_only_with_explicit_directory(invoke, tmp_path):
	run, calls, _ = invoke
	assert run(['--url', 'https://example.test/', '--goal', 'Read'])[0] == 0
	assert calls[0][1].log_dir is None
	assert run(['--url', 'https://example.test/', '--goal', 'Read', '--log-dir', str(tmp_path / 'logs')])[0] == 0
	assert calls[1][1].log_dir == tmp_path / 'logs'


@pytest.mark.parametrize('flag,headless', [('--visible', False), ('--headless', True)])
def test_cli_explicit_browser_display(invoke, flag, headless):
	run, calls, _ = invoke
	assert run(['--url', 'https://example.test/', '--goal', 'Read', flag])[0] == 0
	assert calls[0][1].headless is headless


def test_cli_rejects_conflicting_display_flags(invoke):
	run, calls, _ = invoke
	assert run(['--url', 'https://example.test/', '--goal', 'Read', '--visible', '--headless'])[0] == 2
	assert not calls


def test_cli_connects_to_running_chrome(invoke, monkeypatch):
	run, calls, _ = invoke
	monkeypatch.setattr(cli, 'chrome_cdp_url', lambda: 'ws://127.0.0.1:9222/devtools/browser/test')
	assert run(['--connect-chrome', '--url', 'https://example.test/', '--goal', 'Read'])[0] == 0
	assert calls[0][1].cdp_url == 'ws://127.0.0.1:9222/devtools/browser/test'
	assert calls[0][1].headless is None


def test_cli_accepts_explicit_cdp_endpoint_and_target_id(invoke):
	run, calls, _ = invoke
	assert (
		run(['--cdp-url', 'http://127.0.0.1:9222', '--tab-id', 'existing', '--url', 'https://example.test/', '--goal', 'Read'])[0]
		== 0
	)
	assert calls[0][1].cdp_url == 'http://127.0.0.1:9222' and calls[0][1].tab_id == 'existing'


@pytest.mark.parametrize(
	'options',
	[
		['--tab-id', 'tab'],
		['--cdp-url', 'file:///tmp/browser'],
		['--cdp-url', 'http://localhost:9222', '--visible'],
		['--cdp-url', 'http://localhost:9222', '--executable-path', '/usr/bin/chrome'],
		['--cdp-url', 'http://localhost:9222', '--connect-chrome'],
	],
)
def test_cli_rejects_invalid_connection_settings(invoke, options):
	run, calls, _ = invoke
	assert run(['--url', 'https://example.test/', '--goal', 'Read', *options])[0] == 2
	assert not calls


@pytest.mark.parametrize('contents', [None, '', '9222\n', '99999\n/devtools/browser/test\n', '9222\nnot-a-browser-endpoint\n'])
def test_chrome_discovery_rejects_missing_or_invalid_file(tmp_path, monkeypatch, contents):
	from browser_use.browser import chrome

	monkeypatch.setattr(chrome, 'get_chrome_profile_path', lambda _: str(tmp_path))
	if contents is not None:
		(tmp_path / 'DevToolsActivePort').write_text(contents)
	with pytest.raises(ValueError):
		cli.chrome_cdp_url()


def test_chrome_discovery_uses_websocket_endpoint(tmp_path, monkeypatch):
	from browser_use.browser import chrome

	monkeypatch.setattr(chrome, 'get_chrome_profile_path', lambda _: str(tmp_path))
	(tmp_path / 'DevToolsActivePort').write_text('9222\n/devtools/browser/test\n')
	assert cli.chrome_cdp_url() == 'ws://127.0.0.1:9222/devtools/browser/test'


def test_public_api_exposes_the_adopted_workflow():
	import browser_use.fast as public

	assert public.run_workflow is workflow.run_workflow
	assert public.WorkflowTask is workflow.WorkflowTask
	assert 'DecisionRuntime' not in public.__all__


@pytest.mark.parametrize(
	'arguments',
	[['--url', 'file:///tmp/data', '--goal', 'Read'], ['--url', 'https://example.test/', '--goal', 'Read', '--timeout', '0']],
)
def test_bad_task_or_budget_exits_without_browser_call(invoke, arguments):
	run, calls, _ = invoke
	assert run(arguments)[0] == 2
	assert not calls


def test_ctrl_c_reports_interrupted_without_traceback(invoke, monkeypatch):
	def interrupt(coroutine):
		coroutine.close()
		raise KeyboardInterrupt

	monkeypatch.setattr(cli.asyncio, 'run', interrupt)
	run, _, _ = invoke
	code, result = run(['--url', 'https://example.test/', '--goal', 'Read'])
	assert code == 130 and result['status'] == 'interrupted'
