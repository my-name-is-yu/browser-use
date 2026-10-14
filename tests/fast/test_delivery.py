"""Structured delivery must preserve evidence, missing fields and recovery accounting."""

import json
from typing import Literal

import pytest

from browser_use.fast.delivery import CompleteDelivery, ResultRecord, save_delivery, validate_delivery
from browser_use.fast.views import BrowserTask
from tests.fast.test_goal_loop import Choices, runtime


def delivery(
	disposition: Literal['complete', 'incomplete'] = 'complete',
	value: str | None = 'Applied',
	quote: str | None = 'Applied result',
):
	return CompleteDelivery(
		disposition=disposition,
		browser_summary='Observed applied result',
		records=[
			ResultRecord(id='result', value=value, quote=quote, source_url='https://example.test/' if value is not None else None)
		],
		issues=[] if value else ['Missing result'],
		final_answer='Applied [source](https://example.test/)',
	)


def evidence():
	return {
		'id': 'o',
		'url': 'https://example.test/',
		'title': 'Page',
		'text': 'Applied result',
		'fragment_text': '',
		'headings': [],
		'links': [],
		'targets': [],
		'text_truncated': False,
	}


def test_provenance_missing_and_artifacts(tmp_path):
	data = delivery()
	validate_delivery(data, ['result'], [evidence()])
	receipt = save_delivery(data, tmp_path)
	assert receipt.files == ['results.json', 'report.md']
	assert json.loads((tmp_path / 'results.json').read_text())[0]['quote'] == 'Applied result'
	assert '[source](https://example.test/)' in (tmp_path / 'report.md').read_text()
	assert (tmp_path / 'report.md').read_text().endswith(data.final_answer + '\n')
	with pytest.raises(ValueError, match='Quote/source'):
		validate_delivery(delivery(quote='Invented'), ['result'], [evidence()])
	with pytest.raises(ValueError, match='Missing field'):
		validate_delivery(delivery(value=None), ['result'], [])
	missing = delivery('incomplete', None, None)
	validate_delivery(missing, ['result'], [])
	save_delivery(missing, tmp_path)
	assert json.loads((tmp_path / 'results.json').read_text())[0] == {
		'id': 'result',
		'value': None,
		'quote': None,
		'source_url': None,
		'supporting_quotes': [],
	}


def test_quote_fragment_and_whitespace_keep_observed_url_identity():
	data = delivery(quote='Applied\n  result')
	data.records[0].source_url = 'https://example.test/#observed-section'
	validate_delivery(data, ['result'], [evidence()])
	data.records[0].source_url = 'https://example.test/?different=1#observed-section'
	with pytest.raises(ValueError, match='Quote/source'):
		validate_delivery(data, ['result'], [evidence()])
	data.records[0].source_url = 'https://other.test/#observed-section'
	with pytest.raises(ValueError, match='Quote/source'):
		validate_delivery(data, ['result'], [evidence()])
	linked = evidence()
	linked['links'] = [{'href': 'https://other.test/'}]
	with pytest.raises(ValueError, match='Quote/source'):
		validate_delivery(data, ['result'], [linked])


async def test_done_repair_keeps_goal_identity_deadline_and_budget(tmp_path):
	r = runtime(tmp_path, Choices('every requested', 'every requested'))
	first = await r.run(BrowserTask(url='https://example.test/', goal='Observe the applied result'))
	assert first.status == 'done'
	final = await r.run(resume=first, repair='The submitted result was not observed')
	assert final.run_id == first.run_id and final.task == first.task
	assert final.budget.deadline_at == first.budget.deadline_at
	assert final.budget.decisions_used == first.budget.decisions_used + 1
	assert r.driver.page.goto.await_count == 1
	assert r.decisions.calls[-1][0]['completion_deficiency'] == 'The submitted result was not observed'
	final.budget.max_decisions = final.budget.decisions_used
	blocked = await r.run(resume=final, repair='Still missing source')
	assert blocked.status == 'blocked' and blocked.reason == 'decision_budget_exhausted'
	assert len(r.decisions.calls) == 2
