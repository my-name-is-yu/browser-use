"""Common lossless judge-input policy, applied locally and restored after use."""

import base64
import importlib
import sys
from contextlib import contextmanager
from io import BytesIO
from typing import Any

from PIL import Image
from pydantic import BaseModel

MAX_IMAGES = 500
MAX_IMAGE_BYTES = 100_000_000
TEXT_LIMITS = ('WEBSITE_MAX_CHARS', 'TRAJECTORY_MAX_CHARS', 'FINAL_RESULT_MAX_CHARS', 'FILES_MAX_CHARS')


def all_screenshots(images: list[str], max_images: int, max_bytes: int) -> tuple[list[str], dict[str, Any]]:
	"""Keep every source index, including repeats; reject overflow instead of sampling."""
	if len(images) > max_images or sum(len(image.encode('ascii')) for image in images) > max_bytes:
		raise ValueError('Judge image evidence exceeds frozen count/encoded-byte limits')
	mimes = []
	for encoded in images:
		with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
			image.load()
			if image.format is None:
				raise ValueError('Unrecognized screenshot format')
			mimes.append(Image.MIME[image.format])
	return images, {
		'input_count': len(images),
		'selected_count': len(images),
		'source_indices': list(range(len(images))),
		'mime_types': mimes,
		'encoded_bytes': sum(len(image) for image in images),
		'resized': [],
		'reencoded': [],
		'omitted': [],
		'notes': ['All source screenshot indices retained, including repeated states.'],
	}


def validate_image_budget(trace: dict[str, Any]) -> None:
	"""Include agent-created image deliverables in the shared admission budget."""
	images = [*trace.get('screenshots_b64', []), *[item['data'] for item in trace.get('output_images', [])]]
	if len(images) > MAX_IMAGES or sum(len(image.encode('ascii')) for image in images) > MAX_IMAGE_BYTES:
		raise ValueError('Judge image evidence exceeds frozen count/encoded-byte limits')


@contextmanager
def full_judge_evidence(evaluation):
	"""Override both prompt builder and auditor during one batch, then restore originals."""
	findings = importlib.import_module('findings_judge')
	changes = [(module, name, sys.maxsize) for module in (evaluation, findings) for name in TEXT_LIMITS]
	changes += [
		(evaluation, 'MAX_IMAGES', MAX_IMAGES),
		(evaluation, 'MAX_SCREENSHOT_BYTES', MAX_IMAGE_BYTES),
		(evaluation, 'prepare_screenshots', all_screenshots),
	]
	originals = [(module, name, getattr(module, name)) for module, name, value in changes]
	try:
		for module, name, value in changes:
			setattr(module, name, value)
		yield
	finally:
		for module, name, value in reversed(originals):
			setattr(module, name, value)


class JudgeInputPreview(BaseModel):
	"""Exactly the messages assembled by the real upstream adapter, without an API call."""

	messages: list[dict[str, Any]]
	screenshot_source_indices: list[int]


class _InputBuilt(Exception):
	pass


async def preview_judge_input(evaluation, task: dict[str, Any], trace: dict[str, Any]) -> JudgeInputPreview:
	"""Stop the actual judge path at model invocation and return its complete input."""
	validate_image_budget(trace)
	assembled = []

	class PreviewJudge:
		async def ainvoke(self, messages, **kwargs):
			assembled.extend(message.model_dump(mode='json') for message in messages)
			raise _InputBuilt

	with full_judge_evidence(evaluation):
		try:
			await evaluation.judge_trace(task, trace, PreviewJudge())
		except _InputBuilt:
			return JudgeInputPreview(messages=assembled, screenshot_source_indices=list(range(len(trace['screenshots_b64']))))
	raise RuntimeError('Judge adapter did not reach the input preview boundary')
