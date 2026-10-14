"""The finite-choice transport rejects invalid responses without gating on confidence."""

import json
import time

import httpx
import pytest

from browser_use.fast.decisions import DecisionClient, DecisionError
from browser_use.fast.views import Choice, ChoiceQuestion

QUESTION = ChoiceQuestion(
	name='operation',
	instructions='Choose',
	choices=[
		Choice(value='click', description='Click'),
		Choice(value='wait', description='Wait'),
	],
)


async def test_low_confidence_choice_and_image_are_preserved():
	requests = []

	def handle(request):
		requests.append(json.loads(request.content))
		return httpx.Response(
			200,
			json={
				'answers': [
					{
						'type': 'choice',
						'name': 'operation',
						'choice': 'click',
						'confidence': 0.12,
						'probabilities': [{'value': 'click', 'probability': 0.12}, {'value': 'wait', 'probability': 0.88}],
					}
				],
				'usage': {'input_tokens': 42},
			},
		)

	client = DecisionClient('test', transport=httpx.MockTransport(handle))
	try:
		answer = await client.choose({'job': 'test'}, QUESTION, deadline=time.monotonic() + 10, screenshot='aW1hZ2U=')
		assert answer.answer.choice == 'click'
		assert answer.answer.confidence == 0.12
		assert answer.usage['input_tokens'] == 42
		assert requests[0]['model'] == 'gpt-6-luna'
		assert requests[0]['input'][0]['content'][1]['image_url'].startswith('data:image/png;base64,')
	finally:
		await client.close()


@pytest.mark.parametrize(
	'answer',
	[
		{'type': 'choice', 'name': 'operation', 'choice': 'invented'},
		{'type': 'choice', 'name': 'other', 'choice': 'click'},
		{'type': 'choice', 'name': 'operation', 'choice': 'click', 'confidence': 1.1},
		{
			'type': 'choice',
			'name': 'operation',
			'choice': 'click',
			'probabilities': [
				{'value': 'click', 'probability': 0.6},
				{'value': 'click', 'probability': 0.4},
			],
		},
		{
			'type': 'choice',
			'name': 'operation',
			'choice': 'click',
			'probabilities': [
				{'value': 'click', 'probability': 0.7},
				{'value': 'wait', 'probability': 0.7},
			],
		},
	],
)
async def test_invalid_answer_cannot_dispatch(answer):
	client = DecisionClient('test', transport=httpx.MockTransport(lambda _: httpx.Response(200, json={'answers': [answer]})))
	try:
		with pytest.raises(DecisionError):
			await client.choose({}, QUESTION, deadline=time.monotonic() + 10)
	finally:
		await client.close()


async def test_refusal_is_handoff():
	client = DecisionClient(
		'test',
		transport=httpx.MockTransport(
			lambda _: httpx.Response(200, json={'answers': [{'type': 'refusal', 'name': 'operation'}]})
		),
	)
	try:
		answer = await client.choose({}, QUESTION, deadline=time.monotonic() + 10)
		assert answer.answer.type == 'refusal'
	finally:
		await client.close()


async def test_http_error_does_not_expose_body_or_key():
	client = DecisionClient('secret-value', transport=httpx.MockTransport(lambda _: httpx.Response(401, text='secret-value')))
	try:
		with pytest.raises(DecisionError) as error:
			await client.choose({}, QUESTION, deadline=time.monotonic() + 10)
		assert 'secret-value' not in str(error.value)
	finally:
		await client.close()


async def test_deadline_prevents_network_request():
	def handle(_):
		pytest.fail('Expired request was dispatched')

	client = DecisionClient('test', transport=httpx.MockTransport(handle))
	try:
		with pytest.raises(DecisionError):
			await client.choose({}, QUESTION, deadline=time.monotonic() - 1)
	finally:
		await client.close()
