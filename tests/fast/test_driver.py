"""Observed DOM targets and forms must be revalidated before dispatch."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from browser_use.dom.views import NodeType
from browser_use.fast.driver import BrowserDriver, StaleTarget
from browser_use.fast.views import Action, Target
from tests.fast.test_goal_loop import observation


@pytest.mark.parametrize('op', ['click', 'fill', 'select'])
async def test_target_operations_use_the_observed_backend_node_and_reject_changed_meaning(op):
	o = observation()
	element = SimpleNamespace(click=AsyncMock(), fill=AsyncMock(), select_option=AsyncMock())
	page = SimpleNamespace(evaluate=AsyncMock(return_value='{"passed":true}'), get_element=AsyncMock(return_value=element))
	driver = BrowserDriver(cast(Any, page), 'tab')
	driver._target_nodes = {'field': 73}
	action = Action(id='1', observation_id=o.id, op=op, ref='field', text='exact value', description='Apply')
	await driver.execute(action, o)
	page.get_element.assert_awaited_once_with(73)
	if op == 'click':
		element.click.assert_awaited_once_with()
	else:
		operation = element.fill if op == 'fill' else element.select_option
		operation.assert_awaited_once_with('exact value')
	page.get_element.reset_mock()
	page.evaluate.side_effect = ['{"passed":true}', '{"passed":false}']
	with pytest.raises(StaleTarget):
		await driver.execute(action, o)
	page.get_element.assert_not_awaited()


@pytest.mark.parametrize('capture_fails', [False, True])
async def test_observation_uses_oss_candidates_for_the_requested_tab_and_releases_objects(monkeypatch, capture_fails):
	tree = SimpleNamespace(parent_node=None, node_type=NodeType.DOCUMENT_NODE)
	node = SimpleNamespace(backend_node_id=73, parent_node=tree, node_type=NodeType.ELEMENT_NODE)
	frame_document = SimpleNamespace(parent_node=tree, node_type=NodeType.DOCUMENT_NODE)
	frame_node = SimpleNamespace(backend_node_id=99, parent_node=frame_document, node_type=NodeType.ELEMENT_NODE)
	serializer = SimpleNamespace(
		serialize_accessible_elements=lambda: (SimpleNamespace(selector_map={9: node, 10: frame_node}), {})
	)
	monkeypatch.setattr('browser_use.fast.driver.DOMTreeSerializer', lambda *args, **kwargs: serializer)

	async def snapshot(params, **kwargs):
		if capture_fails:
			raise RuntimeError('Document navigated')
		config = params['arguments'][0]['value']
		assert params['arguments'][1:] == [{'objectId': 'oss-node'}]
		o = observation()
		o.id = config['id']
		o.targets[0].ref = config['id'] + '-0'
		return {'result': {'value': o.model_dump()}}

	resolve = AsyncMock(return_value={'object': {'objectId': 'oss-node'}})
	release = AsyncMock()
	client = SimpleNamespace(
		send=SimpleNamespace(
			DOM=SimpleNamespace(resolveNode=resolve, getDocument=AsyncMock()),
			Runtime=SimpleNamespace(
				evaluate=AsyncMock(return_value={'result': {'objectId': 'window'}}),
				callFunctionOn=AsyncMock(side_effect=snapshot),
				releaseObjectGroup=release,
			),
		)
	)
	dom = SimpleNamespace(
		get_dom_tree=AsyncMock(return_value=(tree, {})), browser_session=SimpleNamespace(id='b', cdp_client=client)
	)

	async def session_id():
		return 'session'

	page = SimpleNamespace(dom_service=dom, session_id=session_id())
	driver = BrowserDriver(cast(Any, page), 'tab')
	if capture_fails:
		with pytest.raises(RuntimeError, match='Document navigated'):
			await driver.observe()
	else:
		o = await driver.observe()
		assert driver._target_nodes[o.targets[0].ref] == 73
	dom.get_dom_tree.assert_awaited_once_with(target_id='tab')
	client.send.DOM.getDocument.assert_awaited_once_with({'depth': 0, 'pierce': True}, session_id='session')
	assert resolve.await_args is not None
	assert resolve.await_args.args[0]['backendNodeId'] == 73
	assert resolve.await_count == 1
	release.assert_awaited_once_with({'objectGroup': resolve.await_args.args[0]['objectGroup']}, session_id='session')


async def test_new_tab_link_revalidates_before_same_tab_navigation():
	o = observation()
	o.targets = [Target(ref='link', tag='a', href='/body', label='Body', new_tab=True)]
	page = SimpleNamespace(evaluate=AsyncMock(side_effect=['{"passed":true}', '{"passed":true}']), goto=AsyncMock())
	driver = BrowserDriver(cast(Any, page), 'tab')
	action = Action(id='1', observation_id=o.id, op='click', ref='link', description='Read Body')
	await driver.execute(action, o)
	page.goto.assert_awaited_once_with('http://example.test/body')
	page.goto.reset_mock()
	page.evaluate.side_effect = ['{"passed":true}', '{"passed":false}']
	with pytest.raises(StaleTarget):
		await driver.execute(action, o)
	page.goto.assert_not_awaited()


async def test_enter_checks_the_observed_input_before_pressing_key():
	o = observation('needle')
	o.focused = 'field'
	page = SimpleNamespace(evaluate=AsyncMock(side_effect=['{"passed":true}', '{"passed":true}']), press=AsyncMock())
	driver = BrowserDriver(cast(Any, page), 'tab')
	await driver.execute(Action(id='1', observation_id=o.id, op='enter', description='Submit'), o)
	assert page.evaluate.await_args.args[1] == {'ref': 'field', 'value': 'needle'}
	page.press.assert_awaited_once_with('Enter')


async def test_navigation_rejects_unobserved_or_changed_source_link():
	o = observation()
	o.links = [{'href': 'https://example.test/body', 'text': 'Body', 'context': 'Read Body'}]
	page = SimpleNamespace(evaluate=AsyncMock(side_effect=['{"passed":true}', '{"passed":false}']), goto=AsyncMock())
	driver = BrowserDriver(cast(Any, page), 'tab')
	with pytest.raises(StaleTarget):
		await driver.execute(Action(id='1', observation_id=o.id, op='navigate', text=o.links[0]['href'], description='Read'), o)
	page.goto.assert_not_awaited()
	page.evaluate.side_effect = ['{"passed":true}']
	with pytest.raises(StaleTarget):
		await driver.execute(
			Action(id='2', observation_id=o.id, op='navigate', text='https://example.test/invented', description='Read'), o
		)
	page.goto.assert_not_awaited()


@pytest.mark.parametrize('new_tab', [False, True])
async def test_observed_link_uses_native_click_unless_it_requests_a_new_tab(monkeypatch, new_tab):
	o = observation()
	o.links = [{'href': 'https://example.test/body', 'text': 'Body', 'context': 'Read Body'}]
	page = SimpleNamespace(
		evaluate=AsyncMock(side_effect=['{"passed":true}', '{"passed":true,"index":2,"new_tab":' + str(new_tab).lower() + '}']),
		goto=AsyncMock(),
	)
	element = SimpleNamespace(click=AsyncMock())
	bind = AsyncMock(return_value=element)
	monkeypatch.setattr('browser_use.fast.driver.native_selector_element', bind)
	await BrowserDriver(cast(Any, page), 'tab').execute(
		Action(id='1', observation_id=o.id, op='navigate', text=o.links[0]['href'], description='Read'), o
	)
	if new_tab:
		page.goto.assert_awaited_once_with(o.links[0]['href'])
		bind.assert_not_awaited()
	else:
		bind.assert_awaited_once_with(page, 'a[href]', 2)
		element.click.assert_awaited_once()
		page.goto.assert_not_awaited()
