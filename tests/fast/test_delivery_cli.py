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
