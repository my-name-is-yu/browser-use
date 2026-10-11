"""Save a parent's evidence-backed delivery without further model calls."""

import json
from pathlib import Path
from typing import Literal
from urllib.parse import urldefrag

from pydantic import Field

from browser_use.fast.views import Contract, Observation, RunResult


class ResultRecord(Contract):
	"""One requested field; null values preserve explicitly missing evidence."""

	id: str = Field(min_length=1)
	value: str | None
	quote: str | None
	source_url: str | None


class CompleteDelivery(Contract):
	"""A single structured judgement, result set and user-facing answer."""

	disposition: Literal['complete', 'continue', 'incomplete']
	browser_summary: str
	records: list[ResultRecord]
	issues: list[str]
	final_answer: str


class DeliveryReceipt(Contract):
	"""Saved state is a claim pending independent semantic and state grading."""

	disposition: str
	files: list[str]
	independent_quality_check_pending: bool = True


def validate_delivery(delivery: CompleteDelivery, required_ids: list[str], evidence: list[dict]) -> None:
	"""Check field coverage and quote provenance, leaving semantic truth to grading."""
	if delivery.disposition == 'continue':
		return
	ids = [record.id for record in delivery.records]
	if len(set(ids)) != len(ids) or set(ids) != set(required_ids):
		raise ValueError('Return each required field exactly once, including missing fields')

	def strings(value):
		if isinstance(value, str):
			yield value
		elif isinstance(value, dict):
			for child in value.values():
				yield from strings(child)
		elif isinstance(value, list):
			for child in value:
				yield from strings(child)

	for record in delivery.records:
		if record.value is not None and not record.value.strip():
			raise ValueError(f'Use null for missing values: {record.id}')
		if record.value is None:
			if delivery.disposition == 'complete':
				raise ValueError(f'Missing field: {record.id}')
			if record.quote is not None or record.source_url is not None:
				raise ValueError(f'Missing fields require null quote/source_url: {record.id}')
			continue
		if not record.quote or not record.quote.strip() or not record.source_url:
			raise ValueError(f'Quote and source URL required: {record.id}')
		quote = ' '.join(record.quote.split())
		if not any(
			any(
				urldefrag(record.source_url)[0] == urldefrag(url)[0]
				for url in [item.get('url'), *[link.get('href') for link in item.get('links', [])]]
				if isinstance(url, str)
			)
			and any(quote in ' '.join(text.split()) for text in strings(item))
			for item in evidence
		):
			raise ValueError(f'Quote/source not present in retrieved evidence: {record.id}')
	if delivery.disposition != 'continue' and not delivery.final_answer.strip():
		raise ValueError('A final answer is required for terminal delivery')


def save_delivery(delivery: CompleteDelivery, directory: Path) -> DeliveryReceipt:
	"""Render the common Markdown format and save both artifacts locally."""
	directory.mkdir(parents=True, exist_ok=True)
	(directory / 'results.json').write_text(
		json.dumps([record.model_dump() for record in delivery.records], ensure_ascii=False, indent=2), encoding='utf-8'
	)

	def cell(value: str | None) -> str:
		return (
			(value if value is not None else 'Missing')
			.replace('&', '&amp;')
			.replace('<', '&lt;')
			.replace('>', '&gt;')
			.replace('|', '&#124;')
			.replace('\n', '<br>')
		)

	lines = [f'Status: {delivery.disposition}', '', '| id | value | quote | source_url |', '| --- | --- | --- | --- |']
	for record in delivery.records:
		source = '[source](' + record.source_url.replace('(', '%28').replace(')', '%29') + ')' if record.source_url else None
		lines.append('| ' + ' | '.join(cell(value) for value in (record.id, record.value, record.quote, source)) + ' |')
	lines += ['', delivery.final_answer]
	(directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
	return DeliveryReceipt(disposition=delivery.disposition, files=['results.json', 'report.md'])


def complete_run(result_path: Path, delivery_path: Path, required_ids: list[str]) -> DeliveryReceipt:
	"""Validate a Codex delivery against saved observations and render it without model calls."""
	run = RunResult.model_validate_json(result_path.read_text())
	delivery = CompleteDelivery.model_validate_json(delivery_path.read_text())
	if delivery.disposition == 'continue':
		raise ValueError('Continue requires more evidence or browser work before saving a terminal delivery')
	if not required_ids or len(set(required_ids)) != len(required_ids):
		raise ValueError('Supply each required field ID exactly once')
	evidence = []
	for line in (Path(run.directory) / 'journal.jsonl').read_text().splitlines():
		event = json.loads(line)
		if event['event'] == 'observed':
			evidence.append(Observation.model_validate(event['observation']).model_dump(exclude={'screenshot'}))
	validate_delivery(delivery, required_ids, evidence)
	directory = Path(run.directory) / 'delivery'
	receipt = save_delivery(delivery, directory)
	(directory / 'final.txt').write_text(delivery.final_answer, encoding='utf-8')
	(directory / 'delivery.json').write_text(delivery.model_dump_json(indent=2), encoding='utf-8')
	return receipt.model_copy(
		update={'files': [str(directory / name) for name in [*receipt.files, 'final.txt', 'delivery.json']]}
	)
