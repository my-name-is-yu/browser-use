"""Browser execution shared by the model-driven Agent and external Python code."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from browser_use.agent.views import ActionModel, ActionResult, AgentHistory, AgentOutput, StepMetadata
from browser_use.browser.session import BrowserSession
from browser_use.browser.views import BrowserStateHistory, BrowserStateSummary
from browser_use.dom.views import DOMInteractedElement
from browser_use.screenshots.service import ScreenshotService
from browser_use.tools.service import Tools


class ActionReceipt(BaseModel):
	model_config = ConfigDict(arbitrary_types_allowed=True)
	index: int
	action: dict[str, Any]
	status: Literal['not_started', 'started', 'returned', 'verified', 'unknown'] = 'not_started'
	result: ActionResult | None = None
	error: str | None = None
	started_at: float | None = None
	returned_at: float | None = None


class BatchExecution(BaseModel):
	receipts: list[ActionReceipt] = Field(default_factory=list)
	stop_reason: str = 'complete'

	@property
	def results(self) -> list[ActionResult]:
		return [receipt.result for receipt in self.receipts if receipt.result is not None]


def is_connection_error(error: Exception) -> bool:
	message = str(error).lower()
	return isinstance(error, ConnectionError) or any(
		text in message
		for text in (
			'websocket connection closed',
			'connection closed',
			'browser has been closed',
			'browser closed',
			'no browser',
		)
	)


async def capture_browser_state(
	browser: BrowserSession, *, include_screenshot: bool = False, include_recent_events: bool = False
) -> BrowserStateSummary:
	return await browser.get_browser_state_summary(
		include_screenshot=include_screenshot, cached=False, include_recent_events=include_recent_events
	)


async def execute_action_batch(
	actions: list[ActionModel],
	browser: BrowserSession,
	tools: Tools,
	*,
	tool_kwargs: dict[str, Any] | None = None,
	check_stop: Callable[[], Awaitable[None]] | None = None,
	before_action: Callable[[int, ActionModel], Awaitable[None]] | None = None,
	progress: Callable[[ActionReceipt], None] | None = None,
	log_action: Callable[[ActionModel, str, int, int], Awaitable[None]] | None = None,
	on_result: Callable[[str, ActionResult], Awaitable[None]] | None = None,
	raise_connection_errors: bool = False,
	report: BatchExecution | None = None,
) -> BatchExecution:
	"""Run an action prefix; record dispatch intent before awaiting browser I/O.

	A returned ActionResult is not evidence that the application's postcondition holds.
	Cancellation preserves the supplied report and propagates to the caller.
	"""
	if report is None:
		report = BatchExecution()
	report.receipts = [ActionReceipt(index=i, action=action.model_dump(exclude_unset=True)) for i, action in enumerate(actions)]
	for i, action in enumerate(actions):
		receipt = report.receipts[i]
		name = next(iter(receipt.action), 'unknown')
		if i > 0 and receipt.action.get('done') is not None:
			report.stop_reason = 'done_requires_single_action'
			break
		try:
			if i > 0:
				await asyncio.sleep(browser.browser_profile.wait_between_actions)
			if check_stop:
				await check_stop()
			if before_action:
				await before_action(i, action)
			if check_stop:
				await check_stop()
			if log_action:
				await log_action(action, name, i + 1, len(actions))
			before_url = await browser.get_current_page_url()
			before_target = browser.agent_focus_target_id
			if check_stop:
				await check_stop()
			receipt.status = 'started'
			receipt.started_at = time.time()
			if progress:
				progress(receipt)
			result = await tools.act(action=action, browser_session=browser, **(tool_kwargs or {}))
			receipt.result = result
			receipt.status = 'unknown' if result.error else 'returned'
			receipt.returned_at = time.time()
			if progress:
				progress(receipt)
			if on_result:
				await on_result(name, result)
			if result.error or result.is_done:
				report.stop_reason = 'action_error' if result.error else 'done'
				break
			if i == len(actions) - 1:
				break
			registered = tools.registry.registry.actions.get(name)
			if registered and registered.terminates_sequence:
				report.stop_reason = 'terminates_sequence'
				break
			if await browser.get_current_page_url() != before_url or browser.agent_focus_target_id != before_target:
				report.stop_reason = 'page_changed'
				break
		except asyncio.CancelledError:
			if receipt.status == 'started':
				receipt.status = 'unknown'
				receipt.result = ActionResult(error='Action dispatch interrupted; outcome unknown. Verify before retrying.')
			report.stop_reason = 'cancelled'
			if progress:
				progress(receipt)
			raise
		except Exception as error:
			receipt.error = f'{type(error).__name__}: {error}'
			if receipt.status == 'started':
				receipt.status = 'unknown'
			if receipt.result is None:
				receipt.result = ActionResult(error=receipt.error)
			report.stop_reason = (
				'interrupted'
				if isinstance(error, InterruptedError)
				else ('precondition_failed' if receipt.status == 'not_started' else 'action_error')
			)
			if progress:
				progress(receipt)
			if raise_connection_errors and (isinstance(error, InterruptedError) or is_connection_error(error)):
				raise
			break
	return report


async def make_execution_history(
	actions: list[ActionModel],
	state: BrowserStateSummary,
	results: list[ActionResult],
	screenshots: ScreenshotService,
	metadata: StepMetadata | None,
	*,
	model_output: AgentOutput | None = None,
	state_message: str | None = None,
	source: Literal['model', 'code', 'decisions'] = 'model',
	step_number: int = 0,
) -> AgentHistory:
	"""Record actions without manufacturing model thoughts for externally supplied code."""
	interacted = []
	for action in actions:
		index = action.get_index()
		node = state.dom_state.selector_map.get(index) if index is not None else None
		interacted.append(DOMInteractedElement.load_from_enhanced_dom_tree(node) if node else None)
	path = (
		await screenshots.store_screenshot(state.screenshot, metadata.step_number if metadata else step_number)
		if state.screenshot
		else None
	)
	return AgentHistory(
		model_output=model_output,
		result=results,
		state=BrowserStateHistory(
			url=state.url, title=state.title, tabs=state.tabs, interacted_element=interacted or [None], screenshot_path=path
		),
		metadata=metadata,
		state_message=state_message,
		action_source=source,
		supplied_actions=[action.model_dump(exclude_unset=True) for action in actions] if source != 'model' else None,
	)
