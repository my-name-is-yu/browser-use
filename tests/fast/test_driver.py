"""Observed DOM targets and forms must be revalidated before dispatch."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from browser_use.fast.driver import BrowserDriver, StaleTarget
from browser_use.fast.views import Action, Target
from tests.fast.test_goal_loop import observation


async def test_new_tab_link_revalidates_before_same_tab_navigation():
	o = observation()
	o.targets = [Target(ref='link', tag='a', href='/body', label='Body', new_tab=True)]
	page = SimpleNamespace(evaluate=AsyncMock(side_effect=['{"passed":true}', '{"passed":true}']), goto=AsyncMock())
	driver = BrowserDriver(page, 'tab')
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
	driver = BrowserDriver(page, 'tab')
	await driver.execute(Action(id='1', observation_id=o.id, op='enter', description='Submit'), o)
	assert page.evaluate.await_args.args[1] == {'ref': 'field', 'value': 'needle'}
	page.press.assert_awaited_once_with('Enter')


async def test_navigation_rejects_unobserved_or_changed_source_link():
	o = observation()
	o.links = [{'href': 'https://example.test/body', 'text': 'Body', 'context': 'Read Body'}]
	page = SimpleNamespace(evaluate=AsyncMock(side_effect=['{"passed":true}', '{"passed":false}']), goto=AsyncMock())
	driver = BrowserDriver(page, 'tab')
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
	await BrowserDriver(page, 'tab').execute(
		Action(id='1', observation_id=o.id, op='navigate', text=o.links[0]['href'], description='Read'), o
	)
	if new_tab:
		page.goto.assert_awaited_once_with(o.links[0]['href'])
		bind.assert_not_awaited()
	else:
		bind.assert_awaited_once_with(page, 'a[href]', 2)
		element.click.assert_awaited_once()
		page.goto.assert_not_awaited()
