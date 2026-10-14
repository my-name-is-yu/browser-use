"""Save a parent's evidence-backed delivery without further model calls."""

import json
from pathlib import Path
from typing import Literal
from urllib.parse import urldefrag

from pydantic import Field

from browser_use.fast.views import Contract


class SupportingQuote(Contract):
	"""One additional exact source quote for a separately requested part of a field."""

	quote: str = Field(min_length=1)
	source_url: str = Field(min_length=1)


class ResultRecord(Contract):
	"""One requested field; null values preserve explicitly missing evidence."""

	id: str = Field(min_length=1)
	value: str | None
	quote: str | None
	source_url: str | None
	supporting_quotes: list[SupportingQuote] = Field(default_factory=list)


class CompleteDelivery(Contract):
	"""A single structured judgement, result set and user-facing answer."""

	disposition: Literal['complete', 'incomplete']
	browser_summary: str
	records: list[ResultRecord]
	issues: list[str]
	final_answer: str


class DeliveryReceipt(Contract):
	"""Paths and terminal disposition of the saved delivery."""

	disposition: str
	files: list[str]


def validate_delivery(delivery: CompleteDelivery, required_ids: list[str], evidence: list[dict]) -> None:
	"""Check field coverage and source quotes without inferring semantic correctness."""
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
			if record.quote is not None or record.source_url is not None or record.supporting_quotes:
				raise ValueError(f'Missing fields require null quote/source_url: {record.id}')
			continue
		if not record.quote or not record.quote.strip() or not record.source_url:
			raise ValueError(f'Quote and source URL required: {record.id}')
		for quote, source_url in [
			(record.quote, record.source_url),
			*[(q.quote, q.source_url) for q in record.supporting_quotes],
		]:
			if not any(
				urldefrag(source_url)[0] == urldefrag(item.get('url', ''))[0]
				and any(' '.join(quote.split()) in ' '.join(text.split()) for text in strings(item))
				for item in evidence
			):
				raise ValueError(f'Quote/source not present in retrieved evidence: {record.id}')
	if not delivery.final_answer.strip():
		raise ValueError('A final answer is required for terminal delivery')


def save_delivery(delivery: CompleteDelivery, directory: Path) -> DeliveryReceipt:
	"""Save source records and the already-rendered answer without a second table."""
	directory.mkdir(parents=True, exist_ok=True)
	(directory / 'results.json').write_text(
		json.dumps([record.model_dump() for record in delivery.records], ensure_ascii=False, indent=2), encoding='utf-8'
	)
	(directory / 'report.md').write_text(f'Status: {delivery.disposition}\n\n{delivery.final_answer}\n', encoding='utf-8')
	return DeliveryReceipt(disposition=delivery.disposition, files=['results.json', 'report.md'])
