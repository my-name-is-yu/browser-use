"""Validated goals, native actions and resumable browser state."""

from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
	model_config = ConfigDict(extra='forbid')


Operation = Literal[
	'click',
	'navigate',
	'fill',
	'select',
	'enter',
	'escape',
	'back',
	'scroll_down',
	'scroll_up',
	'read_page',
	'wait',
	'done',
	'needs_input',
	'blocked',
]


class BrowserTask(Contract):
	"""A natural-language goal and existing exact values; no action recipe."""

	url: str
	goal: str = Field(min_length=1)
	inputs: list[str] = Field(default_factory=list, max_length=20)

	@model_validator(mode='after')
	def valid_url(self):
		if urlparse(self.url).scheme not in ('http', 'https') or not urlparse(self.url).netloc:
			raise ValueError('An absolute HTTP(S) start URL is required')
		return self


class Target(Contract):
	ref: str
	tag: str
	type: str = ''
	label: str = ''
	context: str = ''
	href: str = ''
	value: str = ''
	group: str = ''
	options: list[str] = Field(default_factory=list)
	option_labels: dict[str, str] = Field(default_factory=dict)
	state: str = ''
	new_tab: bool = False


class DecisionPage(Contract):
	"""Current viewport and bounded document evidence, with explicit coverage limits."""

	url: str
	title: str
	text: str
	section: str
	document_text: str
	document_text_truncated: bool


class Observation(Contract):
	id: str
	document_id: str
	target_id: str
	url: str
	title: str
	text: str
	viewport_text: str
	fragment_text: str = ''
	targets: list[Target]
	total_targets: int
	focused: str = ''
	scroll_y: float = 0
	modal: bool = False
	ready: str = 'complete'
	headings: list[dict[str, str]] = Field(default_factory=list)
	links: list[dict[str, str]] = Field(default_factory=list)
	text_truncated: bool = False
	screenshot: str | None = Field(default=None, repr=False)

	def decision_page(self) -> DecisionPage:
		"""Expose saved source text beyond the viewport without implying full coverage."""
		return DecisionPage(
			url=self.url,
			title=self.title,
			text=self.viewport_text,
			section=self.fragment_text,
			document_text=self.text[:12000],
			document_text_truncated=self.text_truncated or len(self.text) > 12000,
		)


class DisplayChange(Contract):
	"""A reason to reobserve, never a verdict that an input was applied successfully."""

	reason: Literal['changed', 'observation_replaced', 'stopped', 'monitor_failed']


class Action(Contract):
	id: str
	observation_id: str
	op: Operation
	description: str
	ref: str = ''
	text: str = ''


class Choice(Contract):
	value: str
	description: str


class ChoiceQuestion(Contract):
	type: Literal['choice'] = 'choice'
	name: str
	instructions: str
	choices: list[Choice] = Field(min_length=2, max_length=255)


class Probability(BaseModel):
	value: str
	probability: float = Field(ge=0, le=1, allow_inf_nan=False)


class Answer(BaseModel):
	type: Literal['choice', 'refusal']
	name: str
	choice: str | None = None
	confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
	probabilities: list[Probability] | None = None


class Decision(Contract):
	answer: Answer
	latency_ms: float
	usage: dict[str, Any] = Field(default_factory=dict)


class RunBudget(Contract):
	max_decisions: int = Field(ge=1)
	decisions_used: int = Field(default=0, ge=0)
	deadline_at: float = Field(gt=0, allow_inf_nan=False)


class HistoryItem(Contract):
	"""Actual dispatch receipt followed by an optional resulting observation."""

	action: Action
	target: Target | None = None
	document_id: str = ''
	status: Literal['started', 'returned', 'not_started', 'unknown']
	url_before: str
	url_after: str = ''
	evidence: str = ''
	error: str | None = None


class RecoveryAction(Contract):
	"""A parent's native dispatch acknowledgment, rather than recovery advice."""

	op: Literal['goto', 'read', 'extract', 'click', 'fill', 'select', 'press', 'back', 'scroll', 'wait']
	status: Literal['returned', 'started', 'unknown', 'not_started']
	selector: str = ''
	value: str = ''
	evidence: str = ''
	error: str | None = None


class RecoveryReceipt(Contract):
	"""Known native effects on the saved browser, bounded by observed documents."""

	run_id: str
	cdp_url: str
	target_id: str
	before: Observation
	after: Observation
	actions: list[RecoveryAction] = Field(min_length=1, max_length=50)
	completed_at: float = Field(gt=0, allow_inf_nan=False)

	@model_validator(mode='after')
	def acknowledged(self):
		if any(action.status != 'returned' or action.error is not None for action in self.actions):
			raise ValueError('Recovery requires returned native actions without unresolved effects')
		if not any(action.op in ('goto', 'click', 'fill', 'select', 'press', 'back', 'scroll') for action in self.actions):
			raise ValueError('Recovery requires an acknowledged operation that can advance the browser')
		if self.before.target_id != self.target_id or self.after.target_id != self.target_id:
			raise ValueError('Recovery observations must belong to the same tab')
		return self


class RunResult(Contract):
	"""A saved goal and budget with input or acknowledged native recovery handoffs."""

	run_id: str
	directory: str
	log_directory: str | None = None
	task: BrowserTask
	initial_inputs: list[str] = Field(default_factory=list, max_length=20)
	budget: RunBudget
	status: Literal['done', 'needs_input', 'blocked', 'unknown', 'error', 'timeout']
	reason: str
	completion_missing_condition: str | None = None
	observation: Observation | None = None
	input_target: Target | None = None
	history: list[HistoryItem] = Field(default_factory=list)
	cdp_url: str | None = None
	target_id: str | None = None
	recovery_receipts: list[RecoveryReceipt] = Field(default_factory=list)

	def summary(self) -> dict[str, Any]:
		o = self.observation
		result: dict[str, Any] = {
			'run_id': self.run_id,
			'status': self.status,
			'reason': self.reason,
			'input_target': self.input_target.model_dump() if self.input_target else None,
			'page': None if o is None else {'url': o.url, 'title': o.title, 'text': (o.fragment_text or o.viewport_text)[:6000]},
			'recent_actions': [h.model_dump() for h in self.history[-8:]],
			'budget': self.budget.model_dump(),
			'result_path': self.directory + '/result.json',
		}
		if self.log_directory is not None:
			result['evidence_path'] = self.log_directory + '/journal.jsonl'
		if self.completion_missing_condition is not None:
			result['completion_missing_condition'] = self.completion_missing_condition
		return result
