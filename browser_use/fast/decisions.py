"""Pooled finite choices. Confidence is recorded, never a universal action gate."""

import asyncio
import json
import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

from browser_use.fast.views import Answer, ChoiceQuestion, Decision


class DecisionError(RuntimeError):
	"""A request failed without exposing HTTP bodies or credentials."""


class _Response(BaseModel):
	answers: list[Answer]
	usage: dict[str, Any] = Field(default_factory=dict)


class DecisionClient:
	"""Validate choices against a supplied menu without generating arguments."""

	def __init__(
		self, api_key: str | None = None, *, model: str = 'gpt-6-luna', transport: httpx.AsyncBaseTransport | None = None
	):
		key = api_key or os.environ.get('OPENAI_API_KEY')
		if not key:
			raise DecisionError('OPENAI_API_KEY is not configured')
		self.model = model
		self._client = httpx.AsyncClient(
			base_url='https://api.openai.com/v1/', headers={'Authorization': f'Bearer {key}'}, transport=transport, timeout=30
		)

	async def close(self) -> None:
		await self._client.aclose()

	async def choose(
		self, state: dict[str, Any], question: ChoiceQuestion, *, deadline: float, screenshot: str | None = None
	) -> Decision:
		"""Dependent choices use subsequent requests; low confidence is not refusal."""
		ids = [c.value for c in question.choices]
		if len(ids) != len(set(ids)):
			raise ValueError('Choice values must be distinct')
		remaining = deadline - time.monotonic()
		if remaining <= 0:
			raise DecisionError('Decision deadline expired')
		payload: dict[str, Any] = {
			'model': self.model,
			'input': json.dumps(state, ensure_ascii=False),
			'questions': [question.model_dump()],
		}
		if screenshot:
			payload['input'] = [
				{
					'role': 'user',
					'content': [
						{'type': 'input_text', 'text': payload['input']},
						{'type': 'input_image', 'image_url': f'data:image/png;base64,{screenshot}'},
					],
				}
			]
		start = time.monotonic()
		try:
			async with asyncio.timeout(remaining):
				response = await self._client.post('decisions', json=payload)
			if response.status_code != 200:
				raise DecisionError(f'Decisions API returned HTTP {response.status_code}')
			body = _Response.model_validate(response.json())
			if len(body.answers) != 1 or body.answers[0].name != question.name:
				raise ValueError('Unexpected answer name or count')
			answer = body.answers[0]
			if answer.type == 'choice':
				if answer.choice not in ids:
					raise ValueError('Answer is outside the supplied menu')
				if answer.probabilities is not None:
					values = [p.value for p in answer.probabilities]
					if (
						set(values) != set(ids)
						or len(values) != len(ids)
						or abs(sum(p.probability for p in answer.probabilities) - 1) > 0.02
					):
						raise ValueError('Invalid choice distribution')
			if time.monotonic() >= deadline:
				raise DecisionError('Decision arrived after deadline')
			return Decision(answer=answer, latency_ms=(time.monotonic() - start) * 1000, usage=body.usage)
		except DecisionError:
			raise
		except (httpx.HTTPError, TimeoutError, ValueError, TypeError):
			raise DecisionError('Decisions request or response validation failed') from None
