"""Goal-loop handoffs preserve browser identity, input scope and request budgets."""

import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from browser_use.fast import BrowserTask, DecisionRuntime
from browser_use.fast.driver import StaleTarget, build_actions
from browser_use.fast.views import Action, Answer, Decision, HistoryItem, Observation, Target


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


def runtime(tmp_path, choices):
	browser = SimpleNamespace(
		cdp_url='http://browser', agent_focus_target_id='tab', start=AsyncMock(), stop=AsyncMock(), kill=AsyncMock()
	)
	r = DecisionRuntime(browser, target_id='tab', decisions=choices, output_dir=tmp_path)
	r.driver = cast(
		Any,
		SimpleNamespace(
			page=SimpleNamespace(goto=AsyncMock()), observe=AsyncMock(return_value=observation()), execute=AsyncMock()
		),
	)
	return cast(Any, r)


async def test_input_request_resume_retains_identity_and_budget(tmp_path):
	r = runtime(tmp_path, Choices('ask codex'))
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


def test_grader_projection_preserves_exact_events():
	import json

	from benchmarks.bu_fast.measure_goal_loop import compact_trace
	from benchmarks.bu_fast.validate_evaluation import Trace

	source = {
		'url': 'https://example.test/source',
		'text': 'source text ' * 100,
		'links': [{'text': 'row', 'href': 'https://example.test/item'}] * 20,
	}
	steps = [json.dumps({'result': source}), json.dumps({'result': source, 'input': '10'})]
	original = Trace(final_result='', agent_steps=steps, screenshots_b64=[], screenshot_steps=[])
	projected = compact_trace(original)
	values = {}

	def restore(value):
		if isinstance(value, dict):
			if 'evidence_value_id' in value:
				content = restore(value['value'])
				values[value['evidence_value_id']] = content
				return content
			if 'evidence_value_ref' in value:
				return values[value['evidence_value_ref']]
			return {k: restore(v) for k, v in value.items()}
		if isinstance(value, list):
			return [restore(v) for v in value]
		return value

	assert [restore(json.loads(s)) for s in projected.agent_steps] == [json.loads(s) for s in steps]


def test_measurement_weights_and_resume_interface():
	from benchmarks.bu_fast.measure_goal_loop import ResumeBrowser, grading_task, load_tasks

	for task in load_tasks():
		assert sum(grading_task(task)['weights'].values()) == 100
		assert sum(grading_task(task, True)['weights'].values()) == 100
	assert set(ResumeBrowser.model_fields) == {'inputs'}
