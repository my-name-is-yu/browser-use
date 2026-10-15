"""Validated task requirements and literal ranking of retained source text."""

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from browser_use.fast.views import Contract


class PurposeCondition(Contract):
	"""An observable outcome, never a selector or a site-specific action recipe."""

	id: str = Field(min_length=1)
	requirement: str = Field(min_length=1, max_length=1200)
	kind: Literal['fact', 'artifact'] = 'fact'
	path: str | None = None

	@model_validator(mode='after')
	def validate_artifact(self):
		"""Only explicitly declared artifacts use file receipts instead of source citations."""
		if self.kind == 'artifact':
			if not self.path or not self.path.strip() or self.path == '.':
				raise ValueError('Artifact requirements need a relative file path')
			path = Path(self.path)
			if path.is_absolute() or '..' in path.parts:
				raise ValueError('Artifact paths must stay inside outputs')
			if path.as_posix().casefold() in ('report.md', 'results.json'):
				raise ValueError('Artifact path is reserved for internal outputs')
		elif self.path is not None:
			raise ValueError('Only artifact requirements may specify a path')
		return self


class PurposePlan(Contract):
	"""The parent's initial interpretation and candidate input text."""

	conditions: list[PurposeCondition] = Field(max_length=254)
	inputs: list[str] = Field(default_factory=list, max_length=20)

	@model_validator(mode='after')
	def unique_conditions(self):
		if len({condition.id for condition in self.conditions}) != len(self.conditions) or any(
			c.id == 'COMPLETE' for c in self.conditions
		):
			raise ValueError('Purpose conditions must have unique IDs')
		if any(not value.strip() for value in self.inputs):
			raise ValueError('Input candidates cannot be empty')
		return self


def relevance(query: str, text: str) -> float:
	"""Rank observed passages with literal identifiers and ordinary query words."""
	words = set(re.findall(r'[a-zA-Z][a-zA-Z0-9_]{2,}|\b\d+(?:\.\d+)?\b', query.lower())) - {
		'the',
		'and',
		'one',
		'with',
		'its',
		'for',
		'find',
		'observe',
		'documented',
		'documentation',
		'including',
		'example',
	}
	words |= {word[:-1] for word in words if len(word) > 4 and word.endswith('s')}
	lower = text.lower()
	return sum((4 if '_' in word else 1) * min(3, lower.count(word)) for word in words)
