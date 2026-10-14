"""Goal-loop handoffs preserve browser identity, input scope and request budgets."""

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from browser_use.fast.driver import StaleTarget, build_actions
from browser_use.fast.session import DecisionRuntime
from browser_use.fast.views import Action, Answer, BrowserTask, Decision, HistoryItem, Observation, RunResult, Target


def observation(value=''):
	return Observation(
		id='o',
		document_id='d',
		target_id='tab',
		url='http://example.test/',
		title='Form',
		text='Form',
		viewport_text='Form',
		targets=[Target(ref='field', tag='input', label='Search', value=value)],
		total_targets=1,
		screenshot='aW1hZ2U=',
	)


class Choices:
	def __init__(self, *ops):
		self.ops = iter(ops)
		self.calls = []

	async def choose(self, state, question, **kwargs):
		self.calls.append((state, question))
		op = next(self.ops)
		choice = next(c for c in question.choices if op in c.description.lower())
		return Decision(answer=Answer(type='choice', name='action', choice=choice.value), latency_ms=1)


def runtime(tmp_path, choices, *, navigate_to_start=True, save_logs=True):
	browser = SimpleNamespace(
		cdp_url='http://browser', agent_focus_target_id='tab', start=AsyncMock(), stop=AsyncMock(), kill=AsyncMock()
	)
	r = DecisionRuntime(
		browser,
		target_id='tab',
		decisions=choices,
		timeout=180,
		max_decisions=90,
		output_dir=tmp_path,
		log_dir=tmp_path if save_logs else None,
		navigate_to_start=navigate_to_start,
	)

	async def no_display_change(_):
		await asyncio.Event().wait()

	r.driver = cast(
		Any,
		SimpleNamespace(
			page=SimpleNamespace(goto=AsyncMock()),
			observe=AsyncMock(return_value=observation()),
			execute=AsyncMock(),
			watch_display_change=AsyncMock(side_effect=no_display_change),
			stop_display_watch=AsyncMock(),
		),
	)
	return cast(Any, r)


async def test_existing_tab_runs_from_current_state_without_reloading(tmp_path):
	r = runtime(tmp_path, Choices('every requested'), navigate_to_start=False)
	result = await r.run(BrowserTask(url='http://example.test/', goal='Read the current page'))
	assert result.status == 'done'
	r.driver.page.goto.assert_not_awaited()
	assert result.observation.document_id == 'd'


async def test_runtime_without_diagnostic_logs_keeps_images_state_and_input_resume(tmp_path):
	r = runtime(tmp_path, Choices('ask codex', 'every requested'), save_logs=False)
	first = await r.run(BrowserTask(url='http://example.test', goal='Search'))
	final = await r.run(resume=first, inputs=['Python'])
	assert final.status == 'done' and final.run_id == first.run_id
	assert final.budget.deadline_at == first.budget.deadline_at
	directory = Path(final.directory)
	assert (directory / 'o.png').read_bytes() == b'image'
	saved = RunResult.model_validate_json((directory / 'result.json').read_text())
	assert saved.observation is not None and saved.observation.screenshot == 'aW1hZ2U='
	assert not (directory / 'journal.jsonl').exists()
	assert 'evidence_path' not in final.summary()
	assert Path(final.summary()['result_path']).is_file()


@pytest.mark.parametrize('save_logs', [False, True])
async def test_input_request_resume_retains_identity_and_budget(tmp_path, save_logs):
	r = runtime(tmp_path, Choices('ask codex'), save_logs=save_logs)
	first = await r.run(BrowserTask(url='http://example.test', goal='Search for Python'))
	assert first.status == 'needs_input' and first.input_target.ref == 'field'
	before = first.budget.deadline_at
	r.decisions = Choices('fill', 'every requested')
	r.driver.observe.side_effect = [observation(), observation('Python')]
	final = await r.run(resume=first, inputs=['Python'])
	assert final.status == 'done' and final.run_id == first.run_id
	assert final.budget.deadline_at == before and final.budget.decisions_used == 3
	assert r.driver.page.goto.await_count == 1 and r.driver.execute.await_args.args[0].text == 'Python'
	assert r.decisions.calls[0][0]['available_inputs'] == ['Python']


async def test_unknown_effect_is_not_replayed(tmp_path):
	r = runtime(tmp_path, Choices('fill'))
	r.driver.execute.side_effect = ConnectionError('lost')
	result = await r.run(BrowserTask(url='http://example.test', goal='Search', inputs=['value']))
	assert result.status == 'unknown' and result.history[0].status == 'unknown'
	with pytest.raises(ValueError):
		await r.run(resume=result, inputs=['value'])
	assert r.driver.execute.await_count == 1


async def test_decision_receives_offscreen_source_and_coverage_limit(tmp_path):
	"""An article title must not hide source facts that were already captured offscreen."""
	r = runtime(tmp_path, Choices('every requested'))
	o = observation()
	o.title = 'Article'
	o.viewport_text = 'Article navigation'
	o.text = 'Navigation\n' + 'Python is a high-level programming language.\n' + 'x' * 13000
	r.driver.observe.return_value = o
	await r.run(BrowserTask(url='http://example.test', goal='Observe the article definition'))
	page = r.decisions.calls[0][0]['page']
	assert 'Python is a high-level programming language.' in page['document_text']
	assert 'Python' not in page['text']
	assert len(page['document_text']) == 12000
	assert page['document_text_truncated'] is True
	assert 'matching title or URL alone' in r.decisions.calls[0][1].instructions


async def test_vision_is_sent_saved_and_retained_across_input_resume(tmp_path):
	class ImageChoices(Choices):
		async def choose(self, state, question, **kwargs):
			assert kwargs['screenshot'] == 'aW1hZ2U='
			return await super().choose(state, question, **kwargs)

	r = runtime(tmp_path, ImageChoices('ask codex', 'every requested'))
	o = observation()
	o.screenshot = 'aW1hZ2U='
	r.driver.observe.return_value = o
	first = await r.run(BrowserTask(url='http://example.test', goal='Search'))
	final = await r.run(resume=first, inputs=['Python'])
	assert final.status == 'done'
	assert r.driver.observe.await_args.kwargs == {'screenshot': True}
	events = [json.loads(line) for line in (Path(final.directory) / 'journal.jsonl').read_text().splitlines()]
	images = [e['image'] for e in events if e['event'] == 'decision']
	assert len(images) == 2
	assert all(Path(image['path']).read_bytes() == b'image' for image in images)
	assert all(image['bytes'] == 5 for image in images)
	assert final.budget.deadline_at == first.budget.deadline_at


async def test_missing_vision_image_does_not_silently_send_text_only(tmp_path):
	r = runtime(tmp_path, Choices('every requested'))
	r.driver.observe.return_value.screenshot = None
	result = await r.run(BrowserTask(url='http://example.test', goal='Read'))
	assert result.status == 'error' and 'missing its screenshot' in result.reason
	assert result.budget.decisions_used == 0 and not r.decisions.calls


@pytest.mark.parametrize('source_truncated', [False, True])
def test_decision_page_preserves_source_collection_truncation(source_truncated):
	o = observation()
	o.text_truncated = source_truncated
	page = o.decision_page()
	assert page.document_text == o.text
	assert page.document_text_truncated is source_truncated


@pytest.mark.parametrize('save_logs', [False, True])
async def test_interrupted_dispatch_is_saved_and_cannot_resume(tmp_path, save_logs):
	r = runtime(tmp_path, Choices('fill'), save_logs=save_logs)

	async def interrupt(action, observed):
		assert r.result
		saved = RunResult.model_validate_json((Path(r.result.directory) / 'result.json').read_text())
		assert saved.history[-1].status == 'started'
		assert saved.budget.decisions_used == 1
		raise asyncio.CancelledError

	r.driver.execute.side_effect = interrupt
	with pytest.raises(asyncio.CancelledError):
		await r.run(BrowserTask(url='http://example.test', goal='Search', inputs=['value']))
	assert r.result
	saved = RunResult.model_validate_json((Path(r.result.directory) / 'result.json').read_text())
	assert saved.status == 'unknown' and saved.history[-1].status == 'unknown'
	if save_logs:
		events = [json.loads(line) for line in (Path(saved.directory) / 'journal.jsonl').read_text().splitlines()]
		assert next(e for e in events if e['event'] == 'observed')['observation']['text'] == 'Form'
		assert events[-1]['event'] == 'receipt'
	else:
		assert not (Path(saved.directory) / 'journal.jsonl').exists()
	with pytest.raises(ValueError):
		await r.run(resume=saved, inputs=['value'])
	r.driver.execute.assert_awaited_once()


@pytest.mark.parametrize('status', ['started', 'unknown'])
async def test_resume_rejects_unresolved_saved_dispatch(tmp_path, status):
	r = runtime(tmp_path, Choices('ask codex'))
	result = await r.run(BrowserTask(url='http://example.test', goal='Search'))
	result.history.append(
		HistoryItem(
			action=Action(id='fill', observation_id='o', op='fill', description='Fill search'),
			status=status,
			url_before='http://example.test/',
		)
	)
	with pytest.raises(ValueError, match='unresolved operation'):
		await r.run(resume=result, inputs=['Python'])
	r.driver.execute.assert_not_awaited()


async def test_stale_target_is_reobserved(tmp_path):
	r = runtime(tmp_path, Choices('fill', 'every requested'))
	r.driver.execute.side_effect = StaleTarget('changed')
	result = await r.run(BrowserTask(url='http://example.test', goal='Search', inputs=['value']))
	assert result.history[0].status == 'not_started' and r.driver.observe.await_count == 2


async def test_resume_does_not_reset_expired_budget(tmp_path):
	r = runtime(tmp_path, Choices('ask codex'))
	result = await r.run(BrowserTask(url='http://example.test', goal='Search'))
	result.budget.deadline_at = time.time() - 1
	final = await r.run(resume=result, inputs=['Python'])
	assert final.status == 'timeout' and final.budget.decisions_used == 1


async def test_resume_rejects_another_tab(tmp_path):
	r = runtime(tmp_path, Choices('ask codex'))
	result = await r.run(BrowserTask(url='http://example.test', goal='Search'))
	r.target_id = 'other'
	with pytest.raises(ValueError):
		await r.run(resume=result, inputs=['Python'])


def test_menu_has_target_bound_values_and_input_request():
	actions = build_actions(observation(), ['Python'])
	assert sum(a.op == 'fill' for a in actions) == 1
	assert next(a for a in actions if a.op == 'fill').text == 'Python'
	assert next(a for a in actions if a.op == 'needs_input').ref == 'field'
	assert all(a.op not in ('extract', 'replan') for a in actions)


async def test_budget_exhaustion_prevents_extra_request(tmp_path):
	r = runtime(tmp_path, Choices('fill'))
	r.max_decisions = 1
	result = await r.run(BrowserTask(url='http://example.test', goal='Search', inputs=['value']))
	assert result.reason == 'decision_budget_exhausted' and len(r.decisions.calls) == 1


def recovery_receipt(result, after=None):
	from browser_use.fast.views import RecoveryAction, RecoveryReceipt

	return RecoveryReceipt(
		run_id=result.run_id,
		cdp_url=result.cdp_url,
		target_id=result.target_id,
		before=result.observation,
		after=after or result.observation,
		actions=[RecoveryAction(op='goto', status='returned', value='http://example.test/')],
		completed_at=time.time(),
	)


@pytest.mark.parametrize('save_logs', [False, True])
async def test_blocked_recovery_reobserves_and_preserves_budget(tmp_path, save_logs):
	r = runtime(tmp_path, Choices('no supported operation'), save_logs=save_logs)
	first = await r.run(BrowserTask(url='http://example.test', goal='Search'))
	after = observation('recovered')
	after.id, after.document_id = 'recovered', 'new-document'
	r.driver.observe.return_value = after
	r.decisions = Choices('every requested')
	final = await r.run(resume=first, recovery=recovery_receipt(first, after))
	assert final.status == 'done' and final.run_id == first.run_id
	assert final.task == first.task and final.budget.deadline_at == first.budget.deadline_at
	assert final.budget.decisions_used == first.budget.decisions_used + 1
	assert len(final.recovery_receipts) == 1 and r.driver.page.goto.await_count == 1
	assert r.decisions.calls[0][0]['targets'][0]['value'] == 'recovered'
	assert 'native_recovery' in r.decisions.calls[0][0]
	if save_logs:
		events = [json.loads(line) for line in (Path(final.directory) / 'journal.jsonl').read_text().splitlines()]
		assert any(e['event'] == 'recovery_receipt' for e in events)
	else:
		assert not (Path(final.directory) / 'journal.jsonl').exists()


async def test_recovery_rejects_changed_document_without_new_decision(tmp_path):
	r = runtime(tmp_path, Choices('no supported operation'))
	first = await r.run(BrowserTask(url='http://example.test', goal='Search'))
	receipt = recovery_receipt(first)
	changed = observation()
	changed.document_id = 'unexpected-document'
	r.driver.observe.return_value = changed
	final = await r.run(resume=first, recovery=receipt)
	assert final.reason == 'recovery_observation_changed'
	assert final.budget.decisions_used == first.budget.decisions_used
	assert len(r.decisions.calls) == 1


@pytest.mark.parametrize('change', ['run', 'tab', 'before', 'unknown', 'budget'])
async def test_recovery_requires_known_receipts_and_same_run(tmp_path, change):
	r = runtime(tmp_path, Choices('no supported operation'))
	first = await r.run(BrowserTask(url='http://example.test', goal='Search'))
	receipt = recovery_receipt(first)
	if change == 'run':
		receipt.run_id = 'another-run'
	elif change == 'tab':
		receipt.target_id = 'another-tab'
	elif change == 'before':
		receipt.before = observation('unrecorded')
	elif change == 'unknown':
		receipt.actions[0].status = 'unknown'
	elif change == 'budget':
		first.budget.decisions_used = first.budget.max_decisions
		final = await r.run(resume=first, recovery=receipt)
		assert final.reason == 'decision_budget_exhausted'
		assert len(r.decisions.calls) == 1
		return
	with pytest.raises(ValueError):
		await r.run(resume=first, recovery=receipt)
	assert len(r.decisions.calls) == 1


async def test_recovery_does_not_reset_expired_deadline(tmp_path):
	r = runtime(tmp_path, Choices('no supported operation'))
	first = await r.run(BrowserTask(url='http://example.test', goal='Search'))
	receipt = recovery_receipt(first)
	first.budget.deadline_at = time.time() - 1
	final = await r.run(resume=first, recovery=receipt)
	assert final.status == 'timeout'
	assert final.budget.deadline_at == first.budget.deadline_at
	assert final.budget.decisions_used == first.budget.decisions_used
	assert len(r.decisions.calls) == 1


async def test_recovery_rejects_unknown_internal_dispatch(tmp_path):
	r = runtime(tmp_path, Choices('no supported operation'))
	first = await r.run(BrowserTask(url='http://example.test', goal='Search'))
	first.history.append(
		HistoryItem(
			action=Action(id='click', observation_id='o', op='click', description='Click'),
			status='unknown',
			url_before=first.observation.url,
		)
	)
	with pytest.raises(ValueError, match='unresolved operation'):
		await r.run(resume=first, recovery=recovery_receipt(first))
	assert len(r.decisions.calls) == 1


@pytest.mark.parametrize('op', ['read', 'extract', 'wait'])
async def test_read_only_receipts_cannot_reenter_blocked_loop(tmp_path, op):
	from browser_use.fast.views import RecoveryReceipt

	r = runtime(tmp_path, Choices('no supported operation'))
	first = await r.run(BrowserTask(url='http://example.test', goal='Search'))
	data = recovery_receipt(first).model_dump()
	data['actions'][0]['op'] = op
	with pytest.raises(ValueError, match='can advance'):
		RecoveryReceipt.model_validate(data)
	assert len(r.decisions.calls) == 1


async def test_default_runtime_sends_text_and_image_without_read_action(tmp_path):
	class ImageChoices(Choices):
		async def choose(self, state, question, **kwargs):
			assert kwargs['screenshot'] == 'aW1hZ2U='
			assert state['page']['document_text'] == 'Exact source'
			assert not any('Read current page' in c.description for c in question.choices)
			return await super().choose(state, question, **kwargs)

	r = runtime(tmp_path, ImageChoices('every requested'))
	default_runtime = DecisionRuntime(
		r.browser, target_id='tab', decisions=r.decisions, timeout=r.timeout, max_decisions=r.max_decisions, output_dir=tmp_path
	)
	default_runtime.driver = r.driver
	r = cast(Any, default_runtime)
	o = observation()
	o.text = 'Exact source'
	o.screenshot = 'aW1hZ2U='
	r.driver.observe.return_value = o
	result = await r.run(BrowserTask(url='http://example.test', goal='Read exact source'))
	assert result.status == 'done'


async def test_new_handoff_replaces_generated_alternatives_but_keeps_known_values(tmp_path):
	"""Search revisions cannot reactivate every query generated in earlier handoffs."""
	r = runtime(tmp_path, Choices('ask codex'))
	first = await r.run(BrowserTask(url='http://example.test', goal='Explore', inputs=['known']))
	r.decisions = Choices('ask codex')
	second = await r.run(resume=first, inputs=['old query'])
	r.decisions = Choices('every requested')
	final = await r.run(resume=second, inputs=['new query'])
	assert final.task.inputs == ['known', 'new query']
	assert final.initial_inputs == ['known']
	assert final.run_id == first.run_id and final.budget.decisions_used == 3
	assert r.decisions.calls[0][0]['available_inputs'] == ['known', 'new query']
	assert not any('old query' in c.description for c in r.decisions.calls[0][1].choices)


async def test_alternating_states_stop_before_request_budget(tmp_path):
	"""Value changes must not bypass the bound on a two-state operation cycle."""
	r = runtime(tmp_path, Choices(*['fill'] * 8))
	r.driver.observe.side_effect = [observation(value) for value in ['a', 'b'] * 4]
	result = await r.run(BrowserTask(url='http://example.test', goal='Search', inputs=['a', 'b']))
	assert result.status == 'blocked' and result.reason == 'repeated_state_cycle'
	assert result.budget.decisions_used == 5
	assert r.driver.execute.await_count == 5


async def test_distinct_applied_states_do_not_trigger_cycle_guard(tmp_path):
	"""Sequential conversions and automatically applied inputs remain supported."""
	r = runtime(tmp_path, Choices('fill', 'fill', 'every requested'))
	r.driver.observe.side_effect = [observation(value) for value in ['', '1', '10']]
	result = await r.run(BrowserTask(url='http://example.test', goal='Compare', inputs=['1', '10']))
	assert result.status == 'done' and result.task.inputs == ['1', '10']
	assert [h.action.op for h in result.history] == ['fill', 'fill']


async def test_returned_fill_context_rebinds_to_current_observation_refs(tmp_path):
	"""Observation-local target IDs change even when the actual field survives."""
	r = runtime(tmp_path, Choices('fill', 'every requested'))
	before, after = observation(), observation('query')
	after.id = 'next'
	after.targets[0].ref = 'new-field-ref'
	after.focused = 'new-field-ref'
	r.driver.observe.side_effect = [before, after]
	await r.run(BrowserTask(url='http://example.test', goal='Search', inputs=['query']))
	state = r.decisions.calls[1][0]
	assert state['last_filled_field'] == {'target': 'new-field-ref', 'value': 'query'}
	assert state['focused_target'] == 'new-field-ref'


@pytest.mark.parametrize('reason', ['changed', 'observation_replaced', 'monitor_failed'])
async def test_display_change_cancels_old_choice_before_dispatch(tmp_path, reason):
	from browser_use.fast.views import DisplayChange

	started, canceled = asyncio.Event(), asyncio.Event()

	class UpdatingChoices(Choices):
		async def choose(self, state, question, **kwargs):
			if state['page']['text'] == 'Old result':
				self.calls.append((state, question))
				started.set()
				try:
					await asyncio.Event().wait()
				finally:
					canceled.set()
			return await super().choose(state, question, **kwargs)

	async def change(_):
		await started.wait()
		return DisplayChange(reason=reason)

	choices = UpdatingChoices('fill', 'every requested')
	r = runtime(tmp_path, choices)
	r.driver.watch_display_change.side_effect = change
	r.driver.observe.side_effect = [
		observation(),
		observation('12').model_copy(update={'viewport_text': 'Old result'}),
		observation('12').model_copy(update={'viewport_text': 'Updated result'}),
	]
	result = await r.run(BrowserTask(url='http://example.test/', goal='Observe the applied input', inputs=['12']))
	assert result.status == 'done'
	assert canceled.is_set() and result.budget.decisions_used == 3
	assert len(choices.calls) == 3
	assert len(result.history) == 1 and result.history[0].status == 'returned'
	r.driver.execute.assert_awaited_once()
	r.driver.stop_display_watch.assert_awaited_once()
	assert 'decision_invalidated' in (Path(result.directory) / 'journal.jsonl').read_text()


async def test_unchanged_display_does_not_hold_the_next_decision(tmp_path):
	choices = Choices('fill', 'every requested')
	r = runtime(tmp_path, choices)
	r.driver.observe.side_effect = [observation(), observation('12')]
	result = await asyncio.wait_for(
		r.run(BrowserTask(url='http://example.test/', goal='Fill the field', inputs=['12'])), timeout=1
	)
	assert result.status == 'done' and result.budget.decisions_used == 2
	r.driver.watch_display_change.assert_awaited_once()
	r.driver.stop_display_watch.assert_awaited_once()
