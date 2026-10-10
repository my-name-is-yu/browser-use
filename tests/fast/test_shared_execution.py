"""The existing Agent uses the same kernel and retains a dispatched prefix."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from browser_use.agent.service import Agent
from browser_use.agent.views import ActionResult
from browser_use.tools.service import Tools


async def test_agent_retains_prefix_on_connection_loss():
	tools = Tools()
	Action = tools.registry.create_action_model()
	actions = [Action.model_validate({'input': {'index': i, 'text': 'Value'}}) for i in (1, 2, 3)]
	tools.act = AsyncMock(
		side_effect=[ActionResult(extracted_content='First input applied'), ConnectionError('connection closed')]
	)
	agent = Agent.__new__(Agent)
	agent.browser_session = SimpleNamespace(
		browser_profile=SimpleNamespace(wait_between_actions=0),
		agent_focus_target_id='tab',
		get_current_page_url=AsyncMock(return_value='http://fixture/'),
	)
	agent.tools = tools
	agent.state = SimpleNamespace(last_result=None, n_steps=1)
	agent.settings = SimpleNamespace(page_extraction_llm=None)
	agent.file_system = None
	agent.sensitive_data = None
	agent.available_file_paths = []
	agent.extraction_schema = None
	agent._check_stop_or_pause = AsyncMock()
	agent._log_action = AsyncMock()
	agent._demo_mode_log = AsyncMock()
	with pytest.raises(ConnectionError):
		await agent.multi_act(actions)
	assert tools.act.await_count == 2
	assert agent.state.last_result[0].extracted_content == 'First input applied'
	assert agent.state.last_result[1].error == 'ConnectionError: connection closed'
