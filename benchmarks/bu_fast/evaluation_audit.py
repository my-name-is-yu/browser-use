"""Measurement and evidence gates shared by development and frozen evaluation.

Missing usage is unknown, partial credit is not completion, and an evaluation
must reject changed code or exposed task groups before dispatching an agent.
"""

import hashlib
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Span(BaseModel):
	"""One monotonic interval; children are included in their parent's duration."""

	id: int
	parent: int | None
	category: str
	start: float
	end: float


class Timings(BaseModel):
	"""Exclusive wall time with an explicit uninstrumented remainder."""

	wall_seconds: float
	exclusive_seconds: dict[str, float]
	unattributed_seconds: float


def summarize_timings(spans: list[Span], start: float, end: float) -> Timings:
	"""Reject double-counted sibling intervals rather than calling the residual inference."""
	if end < start or len({s.id for s in spans}) != len(spans):
		raise ValueError('Invalid trial interval or duplicate span')
	by_id = {s.id: s for s in spans}
	exclusive: dict[str, float] = {}
	for span in spans:
		if not start <= span.start <= span.end <= end:
			raise ValueError('Span is outside the trial')
		if span.parent is not None:
			parent = by_id.get(span.parent)
			if parent is None or not parent.start <= span.start <= span.end <= parent.end:
				raise ValueError('Child is outside its parent')
			seen = {span.id}
			while parent:
				if parent.id in seen:
					raise ValueError('Cyclic span hierarchy')
				seen.add(parent.id)
				parent = by_id.get(parent.parent) if parent.parent is not None else None
		children = [s for s in spans if s.parent == span.id]
		exclusive[span.category] = (
			exclusive.get(span.category, 0) + span.end - span.start - sum(s.end - s.start for s in children)
		)
	for parent_id in {s.parent for s in spans}:
		siblings = sorted((s for s in spans if s.parent == parent_id), key=lambda s: s.start)
		if any(a.end > b.start for a, b in zip(siblings, siblings[1:])):
			raise ValueError('Overlapping siblings require a separate critical-path analysis')
	covered = sum(s.end - s.start for s in spans if s.parent is None)
	return Timings(wall_seconds=end - start, exclusive_seconds=exclusive, unattributed_seconds=end - start - covered)


class Recorder:
	"""Record sequential nested spans, including operations that raise."""

	def __init__(self):
		self.spans: list[Span] = []
		self._stack: list[int] = []
		self._next = 0

	@contextmanager
	def span(self, category: str) -> Iterator[None]:
		identifier = self._next
		self._next += 1
		parent = self._stack[-1] if self._stack else None
		start = time.monotonic()
		self._stack.append(identifier)
		try:
			yield
		finally:
			self._stack.pop()
			self.spans.append(Span(id=identifier, parent=parent, category=category, start=start, end=time.monotonic()))


class TokenUsage(BaseModel):
	"""Observed billing fields; None must never become a zero token count."""

	model_config = ConfigDict(strict=True)
	input_tokens: int | None = Field(default=None, ge=0)
	output_tokens: int | None = Field(default=None, ge=0)
	cached_tokens: int | None = Field(default=None, ge=0)
	cache_write_tokens: int | None = Field(default=None, ge=0)

	@model_validator(mode='after')
	def validate_cached_counts(self):
		if (
			self.input_tokens is not None
			and sum(n or 0 for n in (self.cached_tokens, self.cache_write_tokens)) > self.input_tokens
		):
			raise ValueError('Cache counts exceed input tokens')
		return self

	@classmethod
	def from_api(cls, raw: dict[str, Any]) -> 'TokenUsage':
		"""Normalize Responses, Decisions and Chat Completions without dropping the raw record."""
		details = raw.get('input_tokens_details', raw.get('prompt_tokens_details')) or {}
		return cls(
			input_tokens=raw.get('input_tokens', raw.get('prompt_tokens')),
			output_tokens=raw.get('output_tokens', raw.get('completion_tokens')),
			cached_tokens=details.get('cached_tokens'),
			cache_write_tokens=details.get('cache_write_tokens', raw.get('cache_creation_input_tokens')),
		)


class RequestUsage(BaseModel):
	"""Every HTTP attempt, including errors and cancellations, belongs in the denominator."""

	role: Literal['parent', 'decisions', 'judge']
	model: str
	endpoint: Literal['decisions', 'responses', 'chat/completions', 'codex_cli']
	status: Literal['completed', 'error', 'timeout']
	http_status: int | None = None
	latency_seconds: float = Field(ge=0)
	usage: TokenUsage = Field(default_factory=TokenUsage)
	raw_usage: dict[str, Any] | None = None
	request_id: str | None = None
	service_tier: str | None = None
	response_status: str | None = None
	incomplete_reason: str | None = None


class Price(BaseModel):
	"""Pinned USD per million token rates, supplied with the experiment protocol."""

	model: str
	endpoint: str
	service_tier: str = 'default'
	input: float = Field(ge=0)
	cached: float = Field(ge=0)
	cache_write: float = Field(ge=0)
	output: float = Field(ge=0)
	long_context_boundary: int = 272000
	long_input_multiplier: float = 2
	long_output_multiplier: float = 1.5
	processing_multiplier: float = 1
	regional_multiplier: float = 1


class Cost(BaseModel):
	"""A known token estimate is a lower bound when billing fields are missing."""

	known_usd: float
	complete: bool
	unknown_attempts: int
	missing_fields: list[str]


def summarize_cost(requests: list[RequestUsage], prices: list[Price]) -> Cost:
	"""Account for retries, cached reads, cache writes and reasoning within output tokens."""
	known = 0.0
	unknown = 0
	missing = []
	for index, request in enumerate(requests):
		price = next((p for p in prices if p.model == request.model and p.endpoint == request.endpoint), None)
		usage = request.usage
		fields = []
		if price is None or request.endpoint == 'codex_cli':
			fields.append('pricing_or_billing_basis')
		elif usage.input_tokens is None:
			fields.append('input_tokens')
		else:
			if request.endpoint != 'decisions' and request.service_tier != price.service_tier:
				fields.append('service_tier')
			long = usage.input_tokens > price.long_context_boundary
			input_multiplier = price.long_input_multiplier if long else 1
			output_multiplier = price.long_output_multiplier if long else 1
			if request.endpoint == 'decisions':
				# This endpoint bills all input at one rate, with no cache/output charges.
				amount = usage.input_tokens * price.input * input_multiplier
			else:
				if usage.output_tokens is None:
					fields.append('output_tokens')
				if usage.cached_tokens is None:
					fields.append('cached_tokens')
				if usage.cache_write_tokens is None and price.cache_write != price.input:
					fields.append('cache_write_tokens')
				cached, writes = usage.cached_tokens or 0, usage.cache_write_tokens or 0
				amount = (usage.input_tokens - cached - writes) * price.input + cached * price.cached + writes * price.cache_write
				amount *= input_multiplier
				amount += (usage.output_tokens or 0) * price.output * output_multiplier
			if not fields:
				known += amount * price.processing_multiplier * price.regional_multiplier / 1_000_000
		if fields:
			unknown += 1
			missing.extend(f'{index}:{field}' for field in fields)
	return Cost(known_usd=known, complete=not unknown, unknown_attempts=unknown, missing_fields=missing)


class EvidenceVerdict(BaseModel):
	"""Semantic credit and independently verified completion are separate outcomes."""

	score: float = Field(ge=0, le=1)
	complete: bool
	assessable: bool
	reasons: list[str]


class ArtifactReceipt(BaseModel):
	"""Harness-owned receipt of a delivered file, independent of an agent's claim."""

	path: str
	sha256: str
	size: int = Field(ge=0)


class CollectionGap(BaseModel):
	"""A verified collection failure; absence of agent work is not a collection failure."""

	artifact: str
	reason: Literal['unreadable', 'content_changed']


def artifact_receipt(path: Path) -> ArtifactReceipt:
	"""Record bytes that the harness actually observed at delivery time."""
	content = path.read_bytes()
	return ArtifactReceipt(path=str(path), sha256=hashlib.sha256(content).hexdigest(), size=len(content))


def collect_artifact(receipt: ArtifactReceipt) -> tuple[str | None, CollectionGap | None]:
	"""Verify collection against the earlier receipt without trusting final response text."""
	try:
		content = Path(receipt.path).read_bytes()
	except OSError:
		return None, CollectionGap(artifact=receipt.path, reason='unreadable')
	if len(content) != receipt.size or hashlib.sha256(content).hexdigest() != receipt.sha256:
		return None, CollectionGap(artifact=receipt.path, reason='content_changed')
	try:
		return content.decode('utf-8'), None
	except UnicodeError:
		return None, CollectionGap(artifact=receipt.path, reason='unreadable')


def validate_judgement(
	result: dict[str, Any], weights: dict[str, int], *, oracle: bool | None, collection_gaps: list[CollectionGap] | None = None
) -> EvidenceVerdict:
	"""Check exact rubric coverage, arithmetic, collection gaps and an optional saved-state oracle."""
	if not weights or any(type(w) is not int or w <= 0 for w in weights.values()) or sum(weights.values()) != 100:
		raise ValueError('Weights must be positive integers totaling 100')
	findings = result['judgement']['findings']
	items = [f['item'] for f in findings]
	if len(items) != len(weights) or set(items) != set(weights):
		raise ValueError('Missing, duplicate or unknown finding')
	if any(f['status'] not in ('met', 'violated', 'not_assessable') for f in findings):
		raise ValueError('Unknown finding status')
	for finding in findings:
		reason = finding.get('not_assessable_reason')
		if (finding['status'] == 'not_assessable' and reason not in ('missing_evidence', 'absent_scope')) or (
			finding['status'] != 'not_assessable' and reason is not None
		):
			raise ValueError('Invalid assessment reason')
	calculated = sum(weights[f['item']] for f in findings if f['status'] == 'met') / 100
	if result.get('rh_zeroed'):
		calculated = 0
	if abs(calculated - result['score']) > 1e-9:
		raise ValueError('Incorrect score arithmetic')
	missing = [f['item'] for f in findings if f.get('not_assessable_reason') == 'missing_evidence']
	reasons = (['missing_evidence'] if missing else []) + (
		['clipped_evidence'] if result.get('evidence_clipped_sections') else []
	)
	if collection_gaps:
		reasons.append('collection_gap')
	if oracle is False:
		reasons.append('oracle_rejected')
	assessable = not missing and not result.get('evidence_clipped_sections') and not collection_gaps
	return EvidenceVerdict(
		score=calculated,
		complete=assessable and calculated == 1 and oracle is not False,
		assessable=assessable,
		reasons=reasons,
	)


class TaskGroup(BaseModel):
	"""Split by domain, UI template and task family, not only task ID."""

	id: str
	domain: str
	template: str
	family: str


def assert_unused(tasks: list[TaskGroup], exposed: list[TaskGroup]) -> None:
	"""Reject overlap before reading task bodies or running acceptance trials."""
	for field in ('id', 'domain', 'template', 'family'):
		seen = {getattr(t, field) for t in exposed}
		if any(getattr(t, field) in seen for t in tasks):
			raise ValueError(f'Acceptance overlaps exposed {field}')


def source_hashes(paths: list[Path]) -> dict[str, str]:
	"""Fingerprint only code/config files, never credentials."""
	return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def assert_frozen(path: Path, paths: list[Path]) -> dict[str, Any]:
	"""An immutable protocol must still match the source before every evaluation dispatch."""
	envelope = json.loads(path.read_text())
	payload = envelope['protocol']
	encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
	if hashlib.sha256(encoded).hexdigest() != envelope['sha256'] or payload['source_hashes'] != source_hashes(paths):
		raise ValueError('Frozen protocol or implementation changed')
	return payload


def freeze_protocol(path: Path, payload: dict[str, Any], paths: list[Path]) -> str:
	"""Create a new experiment revision; existing freezes cannot be overwritten."""
	payload = {**payload, 'source_hashes': source_hashes(paths)}
	digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
	with path.open('x') as stream:
		json.dump({'sha256': digest, 'protocol': payload}, stream, indent=2, ensure_ascii=False)
		stream.write('\n')
	return digest
