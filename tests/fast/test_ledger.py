"""Retained evidence must survive presentation limits and citation-based output."""

import csv
import io
import json
from types import SimpleNamespace

import pytest

from browser_use.fast.exploration import PurposeCondition, PurposePlan
from browser_use.fast.ledger import (
	CitedDelivery,
	CitedRecord,
	CitedTable,
	EvidenceLedger,
	EvidenceRequirement,
	LedgerExplorer,
	save_tables,
)
from tests.fast.test_goal_loop import Choices, observation, runtime


def test_ledger_retains_tail_and_distinct_states_without_fabricating_quotes():
	ledger = EvidenceLedger()
	text = 'Navigation\n' * 2000 + 'The frobnicate flag preserves the exact submitted values.'
	ledger.add({'id': 'state1', 'url': 'https://example.test/a', 'text': text})
	ledger.add({'id': 'state2', 'url': 'https://example.test/a', 'text': text.replace('submitted', 'changed')})
	packet = ledger.packet('frobnicate', max_characters=2400)
	assert packet['presentation_truncated']
	assert any('frobnicate' in c['text'] for c in packet['citations'])
	assert {c.state for c in ledger.citations.values()} == {'state1', 'state2'}
	for citation in ledger.citations.values():
		original = text if citation.state == 'state1' else text.replace('submitted', 'changed')
		assert citation.text == original[citation.start : citation.end]


def test_compilation_preserves_arbitrary_markdown_and_copies_only_offered_citations():
	ledger = EvidenceLedger()
	ledger.add({'url': 'https://example.test/', 'text': 'Observed result: 42'})
	cid = ledger.requirement_packet([EvidenceRequirement(id='field', requirement='result')]).citations[0].id
	value = '| result |\n| --- |\n| 42 |\n\n![image](https://example.test/image.png)'
	draft = CitedDelivery(disposition='complete', records=[CitedRecord(id='ANSWER', value=value, citation_ids=[cid])])
	result = ledger.compile(draft, ['ANSWER'])
	assert result.records[0].quote == 'Observed result: 42'
	assert value in result.final_answer
	assert result.final_answer.count(value) == 1
	additional = ledger.compile(draft.model_copy(update={'final_answer': 'Requested additional analysis'}), ['ANSWER'])
	assert value in additional.final_answer
	assert additional.final_answer.endswith('Requested additional analysis')
	with pytest.raises(ValueError, match='not offered'):
		ledger.compile(
			CitedDelivery(disposition='complete', records=[CitedRecord(id='ANSWER', value=value, citation_ids=['invented'])]),
			['ANSWER'],
		)
	with pytest.raises(ValueError, match='each required'):
		ledger.compile(draft, ['ANSWER', 'missing'])


def test_short_final_answer_cannot_hide_the_requested_example_and_result():
	ledger = EvidenceLedger()
	ledger.add({'url': 'https://example.test/', 'text': 'Example: [1, 2].map(x => x * 2) => [2, 4]'})
	cid = ledger.requirement_packet([EvidenceRequirement(id='field', requirement='example')]).citations[0].id
	example = '`[1, 2].map(x => x * 2)` produces `[2, 4]`.'
	draft = CitedDelivery(
		disposition='complete',
		records=[CitedRecord(id='example', value=example, citation_ids=[cid])],
		final_answer='The examples above explain the behavior.',
	)
	compiled = ledger.compile(draft, ['example'])
	assert example in compiled.final_answer
	assert '[source](https://example.test/)' in compiled.final_answer
	assert draft.final_answer in compiled.final_answer


def test_missing_field_does_not_create_a_citation_or_claim_completeness():
	ledger = EvidenceLedger()
	draft = CitedDelivery(disposition='incomplete', records=[CitedRecord(id='missing', value=None)])
	assert ledger.compile(draft, ['missing']).records[0].quote is None
	with pytest.raises(ValueError, match='Missing field'):
		ledger.compile(draft.model_copy(update={'disposition': 'complete'}), ['missing'])


@pytest.mark.parametrize('missing_first', [False, True])
def test_missing_field_sharing_a_cited_table_saves_incomplete_without_retry(tmp_path, missing_first):
	from browser_use.fast.delivery import save_delivery

	ledger = EvidenceLedger()
	ledger.add({'url': 'https://example.test/', 'text': 'For documentation examples.'})
	cid = ledger.requirement_packet([EvidenceRequirement(id='field', requirement='documentation')]).citations[0].id
	table = CitedTable(
		id='contact',
		title='Contact information',
		columns=['purpose', 'support_phone'],
		rows=[['For documentation examples.', '']],
		csv_path='contact.csv',
	)
	records = [
		CitedRecord(id='purpose', value=None, table_ids=['contact'], citation_ids=[cid]),
		CitedRecord(id='support_phone', value=None, table_ids=['contact']),
	]
	if missing_first:
		records.reverse()
	draft = CitedDelivery(
		disposition='incomplete', records=records, tables=[table], issues=['No support telephone number is printed.']
	)
	delivery = ledger.compile(draft, ['purpose', 'support_phone'])
	save_delivery(delivery, tmp_path)
	save_tables(draft, tmp_path)
	saved = {r['id']: r for r in json.loads((tmp_path / 'results.json').read_text())}
	assert delivery.disposition == 'incomplete'
	assert delivery.issues == draft.issues
	assert saved['support_phone'] == {
		'id': 'support_phone',
		'value': None,
		'quote': None,
		'source_url': None,
		'supporting_quotes': [],
	}
	assert saved['purpose']['quote'] == 'For documentation examples.'
	assert delivery.final_answer.count('Contact information') == 1
	assert 'support_phone: evidence is missing.' in delivery.final_answer
	assert list(csv.DictReader(io.StringIO((tmp_path / 'contact.csv').read_text())))[0]['support_phone'] == ''
	with pytest.raises(ValueError, match='Missing field: support_phone'):
		ledger.compile(draft.model_copy(update={'disposition': 'complete'}), ['purpose', 'support_phone'])
	with pytest.raises(ValueError, match='referenced by a cited record'):
		ledger.compile(
			draft.model_copy(update={'records': [CitedRecord(id='support_phone', value=None, table_ids=['contact'])]}),
			['support_phone'],
		)
	with pytest.raises(ValueError, match='not offered'):
		ledger.compile(
			draft.model_copy(
				update={
					'records': [CitedRecord(id='support_phone', value=None, table_ids=['contact'], citation_ids=['invented'])]
				}
			),
			['support_phone'],
		)


async def test_completion_checks_multiple_sources_without_rechecking_each_observation(tmp_path):
	r = runtime(tmp_path, Choices())
	r.__class__ = LedgerExplorer
	r.plan = PurposePlan(conditions=[PurposeCondition(id='facts', requirement='Both source facts')])
	r.sources, r.missing_condition, r.ledger = {}, None, EvidenceLedger()
	r.result = SimpleNamespace(task=SimpleNamespace(goal='Both source facts'))
	first = observation().model_copy(update={'text': 'First required fact'})
	second = observation().model_copy(update={'text': 'Second required fact', 'url': 'https://example.test/second'})
	await r._decision_context(first)
	context = await r._decision_context(second)
	assert context['purpose_conditions'] == [condition.model_dump() for condition in r.plan.conditions]
	assert 'requirements' not in context
	assert context['retained_sources'] and context['observed_input_states']
	assert len(r.ledger.evidence) == 2
	assert not r.decisions.calls


def test_shared_table_preserves_csv_cells_and_narrative_without_repeating_table(tmp_path):
	ledger = EvidenceLedger()
	ledger.add({'url': 'https://example.test/', 'text': 'Observed comma, quote " Unicode 日本語 and two states'})
	cid = ledger.requirement_packet([EvidenceRequirement(id='field', requirement='Observed')]).citations[0].id
	rows = [['12', '53.6', 'comma, quote " 日本語\nline | <tag>'], ['31', '87.8', 'changed']]
	table = CitedTable(
		id='conversions',
		title='Observed conversions',
		columns=['input', 'output', 'note'],
		rows=rows,
		csv_path='nested/conversions.csv',
	)
	draft = CitedDelivery(
		disposition='complete',
		records=[
			CitedRecord(id=key, value=None, table_ids=['conversions'], citation_ids=[cid]) for key in ['baseline', 'changed']
		],
		tables=[table],
		final_answer='Additional requested analysis. ![diagram](https://example.test/image.png)',
	)
	delivery = ledger.compile(draft, ['baseline', 'changed'])
	assert delivery.final_answer.count('Observed conversions') == 1
	assert all(r.value is not None and table.markdown() in r.value for r in delivery.records)
	assert draft.final_answer in delivery.final_answer
	from browser_use.fast.delivery import save_delivery

	save_delivery(delivery, tmp_path)
	assert (tmp_path / 'report.md').read_text().count('Observed conversions') == 1
	assert json.loads((tmp_path / 'results.json').read_text()) == [r.model_dump() for r in delivery.records]
	assert '[source](https://example.test/)' in delivery.final_answer
	paths = save_tables(draft, tmp_path)
	assert list(csv.reader(io.StringIO(paths[0].read_text()))) == [table.columns, *rows]
	assert '&#124;' in delivery.final_answer and '&lt;tag&gt;' in delivery.final_answer
	with pytest.raises(ValueError, match='no unknown'):
		ledger.compile(draft.model_copy(update={'tables': []}), ['baseline', 'changed'])
	with pytest.raises(ValueError, match='relative'):
		CitedTable(id='x', title='x', columns=['x'], rows=[['x']], csv_path='../escape.csv')
	with pytest.raises(ValueError, match='same width'):
		CitedTable(id='x', title='x', columns=['x'], rows=[['x', 'y']])


def test_table_csv_cannot_escape_output_directory_through_symlink(tmp_path):
	(tmp_path / 'outside').mkdir()
	(tmp_path / 'outputs').mkdir()
	(tmp_path / 'outputs/link').symlink_to(tmp_path / 'outside', target_is_directory=True)
	table = CitedTable(id='x', title='x', columns=['x'], rows=[['x']], csv_path='link/escape.csv')
	with pytest.raises(ValueError, match='inside outputs'):
		save_tables(CitedDelivery(disposition='complete', records=[], tables=[table]), tmp_path / 'outputs')
	assert not (tmp_path / 'outside/escape.csv').exists()


def test_requirement_packet_shares_text_without_merging_citations_or_changed_links():
	from browser_use.fast.ledger import EvidenceRequirement

	ledger = EvidenceLedger()
	for state, value, href in [('before', '12', 'https://example.test/a'), ('after', '31', 'https://example.test/b')]:
		ledger.add(
			{
				'id': state,
				'url': 'https://example.test/form',
				'text': f'Applied input {value}. Displayed result for this state.',
				'viewport_text': 'Shared original text',
				'links': [{'text': 'Details', 'href': href, 'context': 'Same visible label'}],
			}
		)
	packet = ledger.requirement_packet(
		[EvidenceRequirement(id='values', requirement='Both applied input values and displayed results')]
	)
	assert list(packet.texts.values()).count('Shared original text') == 1
	shared = [c for c in packet.citations if packet.texts[c.text_ref] == 'Shared original text']
	assert {c.state for c in shared} == {'before', 'after'}
	assert len({c.id for c in shared}) == 2
	for c in packet.citations:
		original = ledger.citations[c.id]
		assert (c.state, c.url, c.start, c.end, packet.texts[c.text_ref]) == (
			original.state,
			original.url,
			original.start,
			original.end,
			original.text,
		)
	assert {(tuple(g.states), g.links[0].href) for g in packet.link_groups} == {
		(('before',), 'https://example.test/a'),
		(('after',), 'https://example.test/b'),
	}
	assert all(packet.texts[g.context_ref] == 'Same visible label' for g in packet.link_groups)
	assert ledger.offered == {c.id for c in packet.citations}


def test_requirement_packet_balances_original_requirements_and_keeps_whole_link_groups():
	from browser_use.fast.ledger import EvidenceRequirement

	ledger = EvidenceLedger()
	ledger.add({'id': 'long', 'url': 'https://example.test/long', 'text': 'common common common ' * 1000})
	ledger.add(
		{
			'id': 'rare',
			'url': 'https://example.test/rare',
			'text': 'Rare fixed-length restriction and its actual example output.',
			'links': [
				{'text': f'item {i}', 'href': f'https://example.test/item/{i}', 'context': 'One observed row'} for i in range(100)
			],
		}
	)
	requirements = [
		EvidenceRequirement(id='common', requirement='common'),
		EvidenceRequirement(id='rare', requirement='Rare fixed-length restriction example output'),
	]
	packet = ledger.requirement_packet(requirements, max_characters=2400)
	assert any('Rare fixed-length' in text for text in packet.texts.values())
	assert [r.requirement for r in packet.requirements] == [r.requirement for r in requirements]
	assert all(r.citation_ids for r in packet.requirements)
	assert len(packet.link_groups) == 1 and len(packet.link_groups[0].links) == 100
	assert packet.presentation_truncated
	assert ledger.offered == {c.id for c in packet.citations}


@pytest.mark.parametrize('changed', [False, True])
async def test_denied_completion_hands_off_only_when_evidence_is_unchanged(tmp_path, changed):
	from unittest.mock import AsyncMock

	from browser_use.fast.session import DecisionRuntime
	from browser_use.fast.views import BrowserTask

	choices = Choices('every requested', 'required answer', 'every requested', 'all browser requirements')
	r = runtime(tmp_path, choices)
	r.__class__ = LedgerExplorer
	r.plan = PurposePlan(conditions=[PurposeCondition(id='answer', requirement='Required answer')])
	r.sources, r.missing_condition, r.ledger = {}, None, EvidenceLedger()
	r.start = AsyncMock(return_value=r)
	r._build_actions = lambda o, inputs: DecisionRuntime._build_actions(r, o, inputs)
	first = observation('12')
	second = first.model_copy(deep=True, update={'id': 'new-observation'})
	first.targets[0].ref = 'o-0'
	second.targets[0].ref = 'new-observation-0'
	if changed:
		second.text = 'Required answer after applying 31'
		second.targets[0].value = '31'
	r.driver.observe.side_effect = [first, second]
	result = await r.run(BrowserTask(url='http://example.test/', goal='Read the required answer'))
	if changed:
		assert result.status == 'done'
		assert result.completion_missing_condition is None
		assert len(choices.calls) == 4
	else:
		assert result.status == 'blocked' and result.reason == 'completion_evidence_unchanged'
		assert result.summary()['completion_missing_condition'] == 'Required answer'
		assert len(choices.calls) == 3
	assert result.history == []


def test_input_states_keep_pending_and_applied_results_separate():
	ledger = EvidenceLedger()
	for id, value, output, url in [
		('applied12', '12', '53.6 °F', 'https://example.test/?val=12'),
		('pending31', '31', '53.6 °F', 'https://example.test/?val=12'),
		('applied31', '31', '87.8 °F', 'https://example.test/?val=31'),
		('same31', '31', '87.8 °F', 'https://example.test/?val=31'),
	]:
		o = observation(value).model_copy(update={'id': id, 'url': url, 'viewport_text': output})
		o.targets[0].ref = id + '-0'
		ledger.add(o.model_dump(exclude={'screenshot'}))
	states = ledger.input_states()
	assert not states.presentation_truncated
	assert [(s.state, s.controls[0].value, s.viewport_text, s.url) for s in states.states] == [
		('applied12', '12', '53.6 °F', 'https://example.test/?val=12'),
		('pending31', '31', '53.6 °F', 'https://example.test/?val=12'),
		('same31', '31', '87.8 °F', 'https://example.test/?val=31'),
	]
	assert len(ledger.evidence) == 4
