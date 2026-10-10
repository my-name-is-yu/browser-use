"""Validated goals, native actions and resumable browser state."""

from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
	model_config = ConfigDict(extra='forbid')


Operation = Literal[
	'click', 'fill', 'select', 'enter', 'escape', 'back', 'scroll_down', 'scroll_up', 'wait', 'done', 'needs_input', 'blocked'
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
	rect: dict[str, float] = Field(default_factory=dict)


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


class Action(Contract):
	id: str
	observation_id: str
	op: Operation
	description: str
	ref: str = ''
	text: str = ''
	url: str = ''


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
	status: Literal['started', 'returned', 'not_started', 'unknown']
	url_before: str
	url_after: str = ''
	evidence: str = ''
	error: str | None = None


class RunResult(Contract):
	"""A saved run supports only input supply, not task replanning."""

	run_id: str
	directory: str
	task: BrowserTask
	budget: RunBudget
	status: Literal['done', 'needs_input', 'blocked', 'unknown', 'error', 'timeout']
	reason: str
	observation: Observation | None = None
	input_target: Target | None = None
	history: list[HistoryItem] = Field(default_factory=list)
	cdp_url: str | None = None
	target_id: str | None = None
	managed_browser: bool = False

	def summary(self) -> dict[str, Any]:
		o = self.observation
		return {
			'run_id': self.run_id,
			'status': self.status,
			'reason': self.reason,
			'input_target': self.input_target.model_dump() if self.input_target else None,
			'page': None if o is None else {'url': o.url, 'title': o.title, 'text': (o.fragment_text or o.viewport_text)[:6000]},
			'recent_actions': [h.model_dump() for h in self.history[-8:]],
			'budget': self.budget.model_dump(),
			'result_path': self.directory + '/result.json',
			'evidence_path': self.directory + '/journal.jsonl',
		}
