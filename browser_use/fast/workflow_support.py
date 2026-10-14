"""Execution records, browser actions and artifact I/O."""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TokenUsage(BaseModel):
	"""Observed billing fields; None must never become a zero token count."""

	model_config = ConfigDict(strict=True)
	input_tokens: int | None = Field(default=None, ge=0)
	output_tokens: int | None = Field(default=None, ge=0)
	cached_tokens: int | None = Field(default=None, ge=0)

	@model_validator(mode='after')
	def validate_cached_counts(self):
		if self.input_tokens is not None and (self.cached_tokens or 0) > self.input_tokens:
			raise ValueError('Cache counts exceed input tokens')
		return self

	@classmethod
	def from_api(cls, raw: dict[str, Any]) -> 'TokenUsage':
		"""Normalize Responses and Decisions without dropping the raw record."""
		details = raw.get('input_tokens_details') or {}
		return cls(
			input_tokens=raw.get('input_tokens'),
			output_tokens=raw.get('output_tokens'),
			cached_tokens=details.get('cached_tokens'),
		)


class RequestUsage(BaseModel):
	"""Every HTTP attempt, including errors and cancellations, belongs in the denominator."""

	role: Literal['parent', 'decisions']
	model: str
	endpoint: Literal['decisions', 'responses']
	status: Literal['completed', 'error', 'timeout', 'cancelled']
	http_status: int | None = None
	latency_seconds: float = Field(ge=0)
	usage: TokenUsage = Field(default_factory=TokenUsage)
	raw_usage: dict[str, Any] | None = None
	request_id: str | None = None
	service_tier: str | None = None
	response_status: str | None = None
	incomplete_reason: str | None = None


class RequestTransport(httpx.AsyncBaseTransport):
	"""Capture every physical HTTP attempt without storing authorization or request bodies."""

	def __init__(self, role: Literal['parent', 'decisions'], requests: list[RequestUsage], ledger: Path, limit: int):
		self.limit = limit
		self.admitted_requests = 0
		self.role = role
		self.requests = requests
		self.ledger = ledger
		self.inner: httpx.AsyncBaseTransport = httpx.AsyncHTTPTransport()

	async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
		if self.admitted_requests >= self.limit:
			raise ValueError(f'{self.role} request budget exhausted')
		self.admitted_requests += 1
		start = time.monotonic()
		payload = json.loads(request.content)
		row = {
			'role': self.role,
			'model': payload.get('model', 'unknown'),
			'endpoint': request.url.path.removeprefix('/v1/'),
			'status': 'error',
		}
		try:
			response = await self.inner.handle_async_request(request)
			await response.aread()
			row['http_status'] = response.status_code
			row['request_id'] = response.headers.get('x-request-id')
			if response.status_code == 200:
				body = response.json()
				row.update(status='completed', raw_usage=body.get('usage'), service_tier=body.get('service_tier'))
				row['response_status'] = body.get('status')
				row['incomplete_reason'] = (body.get('incomplete_details') or {}).get('reason')
				row['usage'] = TokenUsage.from_api(body.get('usage') or {})
			return response
		except asyncio.CancelledError:
			row['status'] = 'cancelled'
			raise
		except (TimeoutError, httpx.TimeoutException):
			row['status'] = 'timeout'
			raise
		finally:
			row['latency_seconds'] = time.monotonic() - start
			event = RequestUsage.model_validate(row)
			self.requests.append(event)
			with self.ledger.open('a') as stream:
				stream.write(event.model_dump_json() + '\n')
				stream.flush()
				os.fsync(stream.fileno())

	async def aclose(self) -> None:
		await self.inner.aclose()


class ToolExchange(BaseModel):
	"""Durable parent-tool response, including exceptions that have no browser event."""

	call_id: str
	name: str
	arguments: str
	status: Literal['completed', 'error', 'timeout']
	result: dict[str, Any] | list[dict[str, Any]]


def record_tool_exchange(directory: Path, exchange: ToolExchange) -> None:
	"""Persist the exact response returned to the parent before another model call."""
	with (directory / 'tool-exchanges.jsonl').open('a') as stream:
		stream.write(exchange.model_dump_json() + '\n')
		stream.flush()
		os.fsync(stream.fileno())


class NativeAction(BaseModel):
	"""One native browser operation; a batch avoids forced per-click model calls."""

	model_config = ConfigDict(extra='forbid')
	op: Literal['goto', 'read', 'extract', 'click', 'fill', 'select', 'press', 'back', 'scroll', 'wait']
	selector: str = ''
	value: str = ''
	index: int = Field(default=0, ge=0)
	seconds: float = Field(default=0, ge=0, le=5)

	@model_validator(mode='after')
	def valid_arguments(self):
		if self.op == 'goto':
			url = urlparse(self.value)
			if url.scheme not in ('http', 'https') or not url.netloc:
				raise ValueError('An absolute HTTP(S) navigation URL is required')
		if self.op in ('extract', 'click', 'fill', 'select') and not self.selector.strip():
			raise ValueError('This operation requires a CSS selector')
		if self.op == 'press' and not self.value:
			raise ValueError('Press requires a key')
		if self.op == 'scroll':
			int(self.value)
		return self


class NativeBatch(BaseModel):
	"""Execute already-known actions together, stopping on the first error."""

	model_config = ConfigDict(extra='forbid')
	actions: list[NativeAction] = Field(min_length=1, max_length=50)


class WriteArtifact(BaseModel):
	"""Write a requested UTF-8 deliverable within this run's outputs directory."""

	model_config = ConfigDict(extra='forbid')
	path: str
	content: str


def delivered_path(directory: Path, name: str) -> Path:
	"""Reject absolute paths, traversal and symlinks outside this run."""
	root = directory.resolve()
	path = root / name
	if not name or Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root):
		raise ValueError('Artifact path must stay inside outputs')
	if path.suffix.lower() not in ('.txt', '.md', '.csv', '.json'):
		raise ValueError('Supported artifacts are UTF-8 text files')
	return path


async def native_selector_element(page: Any, selector: str, index: int):
	"""Bind a fresh CSS match across open shadow roots to the same native Actor element API."""
	session_id = await page.session_id
	client = page.dom_service.browser_session.cdp_client
	await client.send.DOM.getDocument({'depth': 0, 'pierce': True}, session_id=session_id)
	expression = (
		"""((c) => {
const roots=[document]; for(let i=0;i<roots.length;i++) for(const e of roots[i].querySelectorAll('*')) if(e.shadowRoot) roots.push(e.shadowRoot);
return [...new Set(roots.flatMap(r=>[...r.querySelectorAll(c.selector)]))][c.index];
})"""
		+ '('
		+ json.dumps({'selector': selector, 'index': index})
		+ ')'
	)
	result = await client.send.Runtime.evaluate({'expression': expression, 'returnByValue': False}, session_id=session_id)
	object_id = result.get('result', {}).get('objectId')
	if not object_id:
		raise ValueError('No matching native target')
	try:
		description = await client.send.DOM.describeNode({'objectId': object_id}, session_id=session_id)
		return await page.get_element(description['node']['backendNodeId'])
	finally:
		await client.send.Runtime.releaseObject({'objectId': object_id}, session_id=session_id)


def browser_tools():
	"""Expose browser operations and requested files without host-code execution."""
	from browser_use.llm.schema import SchemaOptimizer

	return [
		{
			'type': 'function',
			'name': name,
			'description': description,
			'strict': True,
			'parameters': SchemaOptimizer.create_optimized_json_schema(schema),
		}
		for name, description, schema in [
			(
				'browser_batch',
				'Run native browser actions together. read gives rendered text and target refs; extract reads rendered text matching a CSS selector. Other actions use native Actor methods. Relative hrefs may be resolved from observed pages.',
				NativeBatch,
			),
			(
				'write_artifact',
				'Write a requested deliverable from observed evidence into outputs. Paths must be relative .txt, .md, .csv or .json.',
				WriteArtifact,
			),
		]
	]
