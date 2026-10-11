"""Judge construction and frozen per-cell dispatch must work without model requests."""

import ast
import base64
import importlib
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from benchmarks.bu_fast import judge_evidence as evidence
from benchmarks.bu_fast import measure_delivery as protocol
from benchmarks.bu_fast import measure_goal_loop as measure

UPSTREAM = Path('/private/tmp/bu-fast-upstream-bench-v2.1.1')


def test_grading_matches_baseline_with_only_delivery_answer_suffix():
	"""Freeze the same dictionaries for all variants without extra format requirements."""
	source = subprocess.check_output(
		['git', 'show', f'{protocol.BASE_COMMIT}:benchmarks/bu_fast/measure_goal_loop.py'], text=True
	)
	tree = ast.parse(source)
	function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'grading_task')
	rubrics = next(
		node
		for node in tree.body
		if isinstance(node, ast.Assign)
		and any(isinstance(target, ast.Name) and target.id == 'OPERATION_RUBRICS' for target in node.targets)
	)
	namespace: dict[str, Any] = {'OPERATION_RUBRICS': ast.literal_eval(rubrics.value)}
	exec(compile(ast.Module(body=[function], type_ignores=[]), '<baseline-grading-task>', 'exec'), namespace)
	suffix = ' The final answer must accurately summarize the same records, include supporting source links and explicitly indicate missing fields.'
	for task in measure.load_tasks():
		for operation in (True, False):
			expected = namespace['grading_task'](task, operation)
			if not operation:
				expected['rubric'] += suffix
			actual = measure.grading_task(task, operation)
			assert actual == expected
			assert actual['rubric'].count(suffix.strip()) == (0 if operation else 1)


@pytest.fixture
def evaluation(monkeypatch):
	monkeypatch.syspath_prepend(str(UPSTREAM))
	return importlib.import_module('evaluation')


def png():
	stream = BytesIO()
	Image.new('RGB', (2, 2), 'white').save(stream, format='PNG')
	return base64.b64encode(stream.getvalue()).decode('ascii')


async def test_real_judge_input_preserves_tail_images_artifacts_and_restores(evaluation):
	findings = importlib.import_module('findings_judge')
	original = {
		(module.__name__, name): getattr(module, name) for module in (evaluation, findings) for name in evidence.TEXT_LIMITS
	}
	selector = evaluation.prepare_screenshots
	task = measure.grading_task(measure.load_tasks()[0])
	trace = measure.Trace(
		final_result='a' * 110_000 + 'ANSWER_TAIL',
		agent_steps=[
			json.dumps({'text': 'b' * 710_000 + 'SOURCE_TAIL', 'url': task['website']}),
			json.dumps({'end': 'FINAL_EVENT'}),
		],
		screenshots_b64=[png()] * 52,
		screenshot_steps=[1] * 51 + [2],
		output_files_text='c' * 610_000 + 'ARTIFACT_TAIL',
	)
	projected = measure.compact_trace(trace)
	assert [json.loads(step) for step in measure.restore_trace(projected).agent_steps] == [
		json.loads(step) for step in trace.agent_steps
	]
	preview = await evidence.preview_judge_input(evaluation, task, projected.model_dump())
	encoded = json.dumps(preview.messages)
	assert all(marker in encoded for marker in ('SOURCE_TAIL', 'FINAL_EVENT', 'ANSWER_TAIL', 'ARTIFACT_TAIL'))
	images = [
		part
		for message in preview.messages
		for part in message.get('content', [])
		if isinstance(part, dict) and part.get('type') == 'image_url'
	]
	assert len(images) == 52 and preview.screenshot_source_indices == list(range(52))
	assert evaluation.prepare_screenshots is selector
	assert all(
		getattr(module, name) == original[module.__name__, name]
		for module in (evaluation, findings)
		for name in evidence.TEXT_LIMITS
	)
	with pytest.raises(RuntimeError):
		with evidence.full_judge_evidence(evaluation):
			assert evaluation.TRAJECTORY_MAX_CHARS == sys.maxsize
			raise RuntimeError('interrupted batch')
	assert evaluation.prepare_screenshots is selector
	assert evaluation.TRAJECTORY_MAX_CHARS == original['evaluation', 'TRAJECTORY_MAX_CHARS']


def test_image_overflow_does_not_sample_or_shrink():
	with pytest.raises(ValueError, match='limits'):
		evidence.all_screenshots([png()] * 501, 500, 100_000_000)
	with pytest.raises(ValueError, match='limits'):
		evidence.all_screenshots([png()], 500, 1)
	images, metadata = evidence.all_screenshots([png()] * 3, 500, 100_000_000)
	assert metadata['source_indices'] == [0, 1, 2] and not metadata['omitted']
	assert images == [png()] * 3


@pytest.mark.parametrize('overflow', ['images', 'context'])
async def test_grading_overflow_is_unassessable_for_all_four_traces(tmp_path, monkeypatch, evaluation, overflow):
	directory = tmp_path / 'trial'
	directory.mkdir()
	(directory / 'result.json').write_text(json.dumps({'task': measure.load_tasks()[0]['id']}))
	trace = measure.Trace(
		final_result='answer',
		agent_steps=[json.dumps({'source': 'tail'})],
		screenshots_b64=[png()] * (501 if overflow == 'images' else 1),
		screenshot_steps=[1] * (501 if overflow == 'images' else 1),
	)
	for name in ('trace', 'operation-trace', 'delegation-trace', 'final-delegation-trace'):
		(directory / (name + '.json')).write_text(trace.model_dump_json())
	monkeypatch.setattr(evaluation, 'create_judge', lambda: None)
	monkeypatch.setattr(
		measure, 'ResponsesJudge', lambda pinned, client: SimpleNamespace(client=SimpleNamespace(close=client.aclose))
	)
	calls = []

	async def rejected(*args, **kwargs):
		calls.append(args[0])
		raise RuntimeError('mock context overflow')

	monkeypatch.setattr(measure, 'judge_collected_trace', rejected)
	before = evaluation.TRAJECTORY_MAX_CHARS
	await measure.grade(tmp_path, UPSTREAM)
	result = json.loads((directory / 'graded.json').read_text())
	for name in ('operation', 'delivery', 'delegation', 'final_delegation'):
		assert result[name]['verdict']['assessable'] is False
		assert result[name]['verdict']['complete'] is None
	assert len(calls) == (0 if overflow == 'images' else 4)
	assert evaluation.TRAJECTORY_MAX_CHARS == before


def test_protocol_tampering_and_judge_source_freeze(tmp_path):
	path = protocol.prepare(tmp_path / 'freeze', UPSTREAM, tmp_path / 'env', 1)
	data = json.loads(path.read_text())
	assert str(UPSTREAM / 'judge.py') in data['protocol']['source_hashes']
	assert len(protocol.check(path).order) == 8
	data['protocol']['conditions']['deadline_seconds'] = 241
	path.write_text(json.dumps(data))
	with pytest.raises(ValueError, match='Frozen protocol'):
		protocol.check(path)


async def test_single_trial_dispatch_and_existing_directory_rejection(tmp_path, monkeypatch):
	path = protocol.prepare(tmp_path / 'freeze', UPSTREAM, tmp_path / 'env', 1)
	calls = []
	monkeypatch.setattr(protocol, 'configure_project_environment', lambda path: None)

	async def execute(task, arm, repeat, directory, settings):
		calls.append((task['id'], settings.variant))
		directory.mkdir()
		(directory / 'result.json').write_text('{}')
		return measure.TrialMetrics(task=task['id'], arm=arm, repeat=repeat)

	monkeypatch.setattr(protocol, 'execute_trial', execute)
	await protocol.run(path, False, 1)
	await protocol.run(path, False, 0)
	assert calls == [(protocol.check(path).order[index].task, protocol.check(path).order[index].variant) for index in (1, 0)]
	with pytest.raises(ValueError, match='already exists'):
		await protocol.run(path, False, 1)
	with pytest.raises(ValueError, match='outside'):
		await protocol.run(path, False, -1)
