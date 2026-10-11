"""The adopted CLI saves cited outputs without reopening the browser or calling a model."""

import json
import sys

import pytest

from browser_use.fast import cli
from browser_use.fast.delivery import CompleteDelivery, ResultRecord, complete_run
from browser_use.fast.views import BrowserTask, Observation, RunBudget, RunResult


@pytest.fixture
def saved_delivery(tmp_path):
	observation = Observation(
		id='observed',
		document_id='document',
		target_id='tab',
		url='https://example.test/',
		title='Result',
		text='Applied result',
		viewport_text='Applied result',
		targets=[],
		total_targets=0,
	)
	run = RunResult(
		run_id='run',
		directory=str(tmp_path),
		task=BrowserTask(url=observation.url, goal='Read the result'),
		budget=RunBudget(max_decisions=5, decisions_used=2, deadline_at=1),
		status='done',
		reason='model_done',
		observation=observation,
	)
	result_path = tmp_path / 'result.json'
	result_path.write_text(run.model_dump_json())
	(tmp_path / 'journal.jsonl').write_text(json.dumps({'event': 'observed', 'observation': observation.model_dump()}) + '\n')
	delivery = CompleteDelivery(
		disposition='complete',
		browser_summary='Observed result',
		issues=[],
		records=[ResultRecord(id='result', value='Applied', quote='Applied result', source_url=observation.url)],
		final_answer='Applied [source](https://example.test/)',
	)
	delivery_path = tmp_path / 'input-delivery.json'
	delivery_path.write_text(delivery.model_dump_json())
	return result_path, delivery_path, delivery


def test_complete_cli_renders_all_outputs_without_browser_calls(saved_delivery, monkeypatch, capsys):
	result_path, delivery_path, delivery = saved_delivery

	def forbidden(*args):
		raise AssertionError('Completion must not execute the browser or call a model')

	monkeypatch.setattr(cli, 'execute', forbidden)
	monkeypatch.setattr(
		sys, 'argv', ['bu-fast', 'complete', str(result_path), '--delivery', str(delivery_path), '--required-field', 'result']
	)
	before = result_path.read_bytes()
	with pytest.raises(SystemExit) as exit_info:
		cli.main()
	assert exit_info.value.code == 0
	receipt = json.loads(capsys.readouterr().out)
	assert receipt['independent_quality_check_pending'] is True
	assert len(receipt['files']) == 4
	output = result_path.parent / 'delivery'
	assert json.loads((output / 'results.json').read_text()) == [r.model_dump() for r in delivery.records]
	assert (output / 'report.md').read_text().endswith(delivery.final_answer + '\n')
	assert (output / 'final.txt').read_text() == delivery.final_answer
	assert result_path.read_bytes() == before


def test_cli_rejects_invented_sources_and_preserves_missing_fields(saved_delivery):
	result_path, delivery_path, delivery = saved_delivery
	delivery.records[0].quote = 'Invented evidence'
	delivery_path.write_text(delivery.model_dump_json())
	with pytest.raises(ValueError, match='Quote/source'):
		complete_run(result_path, delivery_path, ['result'])
	assert not (result_path.parent / 'delivery').exists()
	delivery.disposition = 'incomplete'
	delivery.records[0] = ResultRecord(id='result', value=None, quote=None, source_url=None)
	delivery.issues = ['Result missing']
	delivery.final_answer = 'The requested result is missing.'
	delivery_path.write_text(delivery.model_dump_json())
	receipt = complete_run(result_path, delivery_path, ['result'])
	assert receipt.disposition == 'incomplete'
	assert json.loads((result_path.parent / 'delivery/results.json').read_text())[0]['value'] is None
