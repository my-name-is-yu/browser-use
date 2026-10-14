"""Retain exact browser sources and compile citations without model-copied quotes."""

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from browser_use.fast.delivery import CompleteDelivery, ResultRecord, SupportingQuote, validate_delivery
from browser_use.fast.exploration import PurposePlan, relevance
from browser_use.fast.session import POLICY, DecisionRuntime
from browser_use.fast.views import Action, Choice, ChoiceQuestion, Contract, Observation, Target


class SourceCitation(Contract):
	"""An immutable original substring; different states remain separately attributable."""

	id: str
	state: str
	url: str
	text: str
	start: int
	end: int


class CitedRecord(Contract):
	id: str
	value: str | None
	citation_ids: list[str] = Field(default_factory=list)
	table_ids: list[str] = Field(default_factory=list)


class CitedTable(Contract):
	"""Generate cells once for both the requested CSV and the final Markdown table."""

	id: str
	title: str
	columns: list[str] = Field(min_length=1)
	rows: list[list[str]] = Field(min_length=1)
	csv_path: str | None = None

	@model_validator(mode='after')
	def validate_table(self):
		if len(set(self.columns)) != len(self.columns) or any(len(row) != len(self.columns) for row in self.rows):
			raise ValueError('Table columns must be distinct and every row must have the same width')
		if self.csv_path is not None:
			path = Path(self.csv_path)
			if path.is_absolute() or '..' in path.parts or path.suffix.lower() != '.csv':
				raise ValueError('Table CSV path must be relative and end in .csv')
		return self

	def markdown(self) -> str:
		"""Escape cell delimiters without modifying CSV cell values."""

		def row(values):
			return (
				'| '
				+ ' | '.join(
					v.replace('&', '&amp;')
					.replace('<', '&lt;')
					.replace('>', '&gt;')
					.replace('|', '&#124;')
					.replace('\r\n', '\n')
					.replace('\r', '\n')
					.replace('\n', '<br>')
					for v in values
				)
				+ ' |'
			)

		return '\n'.join([self.title, '', row(self.columns), row(['---'] * len(self.columns)), *[row(r) for r in self.rows]])

	def csv_text(self) -> str:
		"""Preserve commas, quotes, Unicode and multiline values with standard CSV escaping."""
		stream = io.StringIO(newline='')
		writer = csv.writer(stream)
		writer.writerow(self.columns)
		writer.writerows(self.rows)
		return stream.getvalue()


class ReadRetainedEvidence(Contract):
	"""Find original citations for a missing requirement without visiting the browser."""

	query: str = Field(min_length=1, max_length=2000)


class CitedDelivery(Contract):
	"""Generate values once; optional prose preserves non-tabular task deliverables."""

	disposition: Literal['complete', 'continue', 'incomplete']
	records: list[CitedRecord]
	issues: list[str] = Field(default_factory=list)
	final_answer: str = ''
	tables: list[CitedTable] = Field(default_factory=list)


class PackedCitation(Contract):
	"""An unchanged citation identity and state referencing one exact shared text."""

	id: str
	state: str
	url: str
	start: int
	end: int
	text_ref: str


class PacketLink(Contract):
	text: str
	href: str


class ObservedLinkGroup(Contract):
	"""Links observed together in one context; states share exactly this link set."""

	source_url: str
	states: list[str]
	context_ref: str
	links: list[PacketLink]


class EvidenceRequirement(Contract):
	"""An unchanged requirement or a targeted retained-source query."""

	id: str
	requirement: str = Field(min_length=1)


class RequirementEvidence(EvidenceRequirement):
	"""Retrieval candidates for an unchanged requirement, not a coverage verdict."""

	citation_ids: list[str]


class RequirementPacket(Contract):
	"""Exact text dictionary with separately attributable citations and link groups."""

	texts: dict[str, str]
	citations: list[PackedCitation]
	requirements: list[RequirementEvidence]
	link_groups: list[ObservedLinkGroup]
	retained_count: int
	retained_link_groups: int
	presentation_truncated: bool


class ObservedInputState(Contract):
	"""Co-observed controls and visible output; a filled control is not proof of application."""

	state: str
	url: str
	document_id: str
	ready: str
	controls: list[Target]
	viewport_text: str


class ObservedInputStates(Contract):
	states: list[ObservedInputState]
	presentation_truncated: bool


class EvidenceLedger:
	"""Keep all sources; bounded presentation is explicitly distinct from acquisition."""

	def __init__(self):
		self.evidence: list[dict[str, Any]] = []
		self.citations: dict[str, SourceCitation] = {}
		self.offered: set[str] = set()
		self.seen: set[str] = set()

	def input_states(self) -> ObservedInputStates:
		"""Keep entered values beside the exact displayed text from the same observation."""
		states = {}
		for source in self.evidence:
			controls = [
				Target.model_validate(t)
				for t in source.get('targets', [])
				if t.get('tag', '').lower() in ('input', 'textarea', 'select') and t.get('value')
			]
			if not controls:
				continue
			state = ObservedInputState(
				state=source['id'],
				url=source['url'],
				document_id=source.get('document_id', ''),
				ready=source.get('ready', ''),
				controls=controls,
				viewport_text=source.get('viewport_text', ''),
			)
			key = json.dumps(state.model_dump(exclude={'state': True, 'controls': {'__all__': {'ref'}}}), sort_keys=True)
			states.pop(key, None)
			states[key] = state
		return ObservedInputStates(states=list(states.values())[-12:], presentation_truncated=len(states) > 12)

	def add(self, evidence: dict[str, Any]) -> None:
		"""Index original source strings without inventing text or merging different states."""
		url = evidence.get('url')
		if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
			return
		fingerprint = hashlib.sha256(repr(evidence).encode()).hexdigest()
		if fingerprint in self.seen:
			return
		self.seen.add(fingerprint)
		self.evidence.append(evidence)
		state = str(evidence.get('id', fingerprint[:20]))

		def texts(value):
			if isinstance(value, dict):
				for key, child in value.items():
					if key in ('text', 'fragment_text', 'viewport_text', 'section', 'value') and isinstance(child, str):
						yield child
					elif isinstance(child, (dict, list)):
						yield from texts(child)
			elif isinstance(value, list):
				for child in value:
					yield from texts(child)

		for text in dict.fromkeys(texts(evidence)):
			for start in range(0, len(text), 1500):
				# Overlap preserves sentences and definitions across chunk boundaries.
				end = min(len(text), start + 2200)
				part = text[start:end]
				if not part.strip():
					continue
				identity = hashlib.sha256((state + '\0' + url + '\0' + part).encode()).hexdigest()[:20]
				self.citations[identity] = SourceCitation(id=identity, state=state, url=url, text=part, start=start, end=end)

	def packet(self, query: str, *, max_characters: int = 48000) -> dict:
		"""Offer relevance-ranked original chunks while retaining every omitted source."""
		selected, size, copies = [], 0, set()
		for citation in sorted(self.citations.values(), key=lambda c: relevance(query, c.text), reverse=True):
			key = (citation.url, citation.text)
			if key in copies or size + len(citation.text) > max_characters:
				continue
			copies.add(key)
			selected.append(citation.model_dump())
			size += len(citation.text)
		links = {}
		for source in self.evidence:
			for link in source.get('links', []):
				if isinstance(link, dict) and str(link.get('href', '')).startswith(('http://', 'https://')):
					links[(link.get('text', ''), link['href'])] = link
		return {
			'citations': selected,
			'observed_links': sorted(links.values(), key=lambda link: relevance(query, str(link)), reverse=True)[:80],
			'retained_count': len(self.citations),
			'presentation_truncated': len(selected) < len(self.citations),
		}

	def requirement_packet(self, requirements: list[EvidenceRequirement], *, max_characters: int = 48000) -> RequirementPacket:
		"""Balance retrieval across requirements and share exact text without merging states."""
		by_text: dict[str, list[SourceCitation]] = {}
		for citation in self.citations.values():
			by_text.setdefault(citation.text, []).append(citation)
		limit = max(6, 24 // max(1, len(requirements)))
		ranked = [sorted(by_text, key=lambda text: relevance(r.requirement, text), reverse=True)[:limit] for r in requirements]
		selected: dict[str, None] = {}
		size = 0
		# Round robin keeps a verbose requirement from consuming every original passage.
		for index in range(limit):
			for candidates in ranked:
				if index >= len(candidates):
					continue
				text = candidates[index]
				if text not in selected and size + len(text) <= max_characters:
					selected[text] = None
					size += len(text)
		texts: dict[str, str] = {}

		def text_ref(text: str) -> str:
			key = hashlib.sha256(text.encode()).hexdigest()[:20]
			texts[key] = text
			return key

		citations = [
			PackedCitation(**c.model_dump(exclude={'text'}), text_ref=text_ref(text)) for text in selected for c in by_text[text]
		]
		self.offered.update(c.id for c in citations)
		fields = [
			RequirementEvidence(
				id=requirement.id,
				requirement=requirement.requirement,
				citation_ids=[c.id for text in candidates if text in selected for c in by_text[text]],
			)
			for requirement, candidates in zip(requirements, ranked)
		]
		groups: dict[tuple, ObservedLinkGroup] = {}
		for source in self.evidence:
			contexts: dict[str, dict[tuple[str, str], PacketLink]] = {}
			for link in source.get('links', []):
				if not isinstance(link, dict) or not str(link.get('href', '')).startswith(('http://', 'https://')):
					continue
				text, href, context = str(link.get('text', '')), link['href'], str(link.get('context', ''))
				contexts.setdefault(context, {})[(text, href)] = PacketLink(text=text, href=href)
			for context, links in contexts.items():
				key = (source['url'], context, tuple(links))
				if key not in groups:
					groups[key] = ObservedLinkGroup(
						source_url=source['url'], states=[], context_ref='', links=list(links.values())
					)
				state = str(source.get('id', hashlib.sha256(repr(source).encode()).hexdigest()[:20]))
				if state not in groups[key].states:
					groups[key].states.append(state)
		# Preserve source order within ranked contexts and never split a context's links.
		ordered = sorted(
			groups, key=lambda key: max((relevance(r.requirement, key[1]) for r in requirements), default=0), reverse=True
		)
		selected_groups = []
		for key in ordered[:80]:
			group = groups[key]
			group.context_ref = text_ref(key[1])
			selected_groups.append(group)
		return RequirementPacket(
			texts=texts,
			citations=citations,
			requirements=fields,
			link_groups=selected_groups,
			retained_count=len(self.citations),
			retained_link_groups=len(groups),
			presentation_truncated=len(citations) < len(self.citations) or len(selected_groups) < len(groups),
		)

	def compile(self, draft: CitedDelivery, required_ids: list[str]) -> CompleteDelivery:
		"""Reject unsupported references; copy exact quotations and render duplicate values."""
		if draft.disposition == 'continue':
			raise ValueError('Continue requires more evidence before compiling a terminal delivery')
		ids = [r.id for r in draft.records]
		if len(set(ids)) != len(ids) or set(ids) != set(required_ids):
			raise ValueError('Return each required field exactly once')
		tables = {table.id: table for table in draft.tables}
		if len(tables) != len(draft.tables):
			raise ValueError('Table IDs must be distinct')
		if {key for r in draft.records for key in r.table_ids} != set(tables):
			raise ValueError('Every table must be referenced by a required record, with no unknown table IDs')
		if {key for r in draft.records if r.citation_ids for key in r.table_ids} != set(tables):
			raise ValueError('Every table must be referenced by a cited record')
		csv_paths = [table.csv_path for table in draft.tables if table.csv_path is not None]
		if len(csv_paths) != len(set(csv_paths)):
			raise ValueError('Table CSV paths must be distinct')
		records = []
		for record in draft.records:
			# A shared table reference does not populate an explicitly missing field.
			if record.value is None and not record.citation_ids:
				records.append(ResultRecord(id=record.id, value=None, quote=None, source_url=None))
				continue
			value = (
				'\n\n'.join(
					[
						*([record.value] if record.value is not None else []),
						*[tables[key].markdown() for key in dict.fromkeys(record.table_ids)],
					]
				)
				or None
			)
			if value is None:
				if record.citation_ids:
					raise ValueError('Null fields cannot claim supporting citations')
				records.append(ResultRecord(id=record.id, value=None, quote=None, source_url=None))
				continue
			if not record.citation_ids or len(set(record.citation_ids)) != len(record.citation_ids):
				raise ValueError('Populated fields require distinct citation IDs')
			if any(cid not in self.offered for cid in record.citation_ids):
				raise ValueError('Citation was not offered from retained browser evidence')
			quotes = [self.citations[cid] for cid in record.citation_ids]
			records.append(
				ResultRecord(
					id=record.id,
					value=value,
					quote=quotes[0].text,
					source_url=quotes[0].url,
					supporting_quotes=[SupportingQuote(quote=c.text, source_url=c.url) for c in quotes[1:]],
				)
			)
		parts = []
		shown_tables = set()
		for record, compiled in zip(draft.records, records):
			if record.value:
				parts.append(f'{record.value}\n\n[source]({compiled.source_url})')
			if compiled.value is None:
				parts.append(f'{record.id}: evidence is missing.')
			for key in record.table_ids:
				if key not in shown_tables:
					urls = dict.fromkeys(
						self.citations[cid].url for r in draft.records if key in r.table_ids for cid in r.citation_ids
					)
					parts.append(tables[key].markdown() + '\n\n' + ' '.join(f'[source]({url})' for url in urls))
					shown_tables.add(key)
		answer = '\n\n'.join(parts)
		if draft.final_answer:
			answer += '\n\n' + draft.final_answer
		delivery = CompleteDelivery(
			disposition=draft.disposition,
			browser_summary=f'{sum(r.value is not None for r in records)}/{len(records)} cited fields',
			records=records,
			issues=draft.issues,
			final_answer=answer,
		)
		validate_delivery(delivery, required_ids, self.evidence)
		return delivery


def save_tables(draft: CitedDelivery, directory: Path) -> list[Path]:
	"""Save validated table data to requested CSV paths; never overwrite a path outside outputs."""
	paths = []
	for table in draft.tables:
		if table.csv_path is not None:
			path = directory / table.csv_path
			if not path.resolve().is_relative_to(directory.resolve()):
				raise ValueError('Table CSV path must stay inside outputs')
			paths.append((path, table))
	for path, table in paths:
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(table.csv_text(), encoding='utf-8', newline='')
	return [path for path, _ in paths]


class LedgerExplorer(DecisionRuntime):
	"""Generic observed operations with shared multi-source evidence and a bounded DONE check."""

	def __init__(self, *args, plan: PurposePlan, ledger: EvidenceLedger, **kwargs):
		super().__init__(*args, **kwargs)
		self.plan = plan
		self.sources: dict[tuple[str, str], dict] = {}
		self.missing_condition: str | None = None
		self.ledger = ledger

	def _build_actions(self, observation: Observation, inputs: list[str]):
		actions = super()._build_actions(observation, inputs)
		if not observation.focused:
			actions = [a for a in actions if a.op != 'enter']
		query = self.missing_condition or ' '.join(c.requirement for c in self.plan.conditions)
		clicks = [a for a in actions if a.op == 'click']
		keep = {a.id for a in sorted(clicks, key=lambda a: relevance(query, a.description), reverse=True)[:35]}
		keep.update(a.id for a in clicks[:45])
		actions = [a for a in actions if a.op != 'click' or a.id in keep]
		headings = sorted(
			[h for h in observation.headings if h.get('id')],
			key=lambda h: relevance(query, h['text']),
			reverse=True,
		)
		for heading in headings[: min(20, max(0, 255 - len(actions)))]:
			actions.append(
				Action(
					id=str(len(actions)),
					observation_id=observation.id,
					op='read_page',
					text=heading['id'],
					description=f'Read the observed document section {heading["text"]!r}; reveal its enclosing details and move to its anchor',
				)
			)
		present = {t.href for t in observation.targets}
		links = [
			link for link in observation.links if link['href'] not in present and link['href'].startswith(('http://', 'https://'))
		]
		links.sort(key=lambda link: relevance(query, link['text'] + ' ' + link['href']), reverse=True)
		for link in links[: min(20, max(0, 255 - len(actions)))]:
			actions.append(
				Action(
					id=str(len(actions)),
					observation_id=observation.id,
					op='navigate',
					text=link['href'],
					description=f'Follow observed document link {link["text"]!r}; href={link["href"]}; context={link["context"][:250]}',
				)
			)
		for index, action in enumerate(actions):
			action.id = str(index)
		return actions

	async def _decision_context(self, observation: Observation) -> dict:
		self.ledger.add(observation.model_dump(exclude={'screenshot'}))
		key = (observation.url, hashlib.sha256(observation.text.encode()).hexdigest())
		self.sources[key] = {
			'id': observation.id,
			**observation.decision_page().model_dump(),
			'links': observation.links[:30],
		}
		while len(self.sources) > 8:
			del self.sources[next(iter(self.sources))]
		context = {
			'purpose_conditions': [condition.model_dump() for condition in self.plan.conditions],
			'completion_missing_condition': self.missing_condition,
			'observed_sources': list(self.sources.values()),
			'observation_policy': 'Only these bounded captured excerpts are known; omitted text is unknown.',
		}
		query = self.result.task.goal if self.result else ' '.join(c.requirement for c in self.plan.conditions)
		context['retained_sources'] = self.ledger.packet(self.missing_condition or query, max_characters=18000)
		context['requirements'] = [c.model_dump() for c in self.plan.conditions]
		context['observed_input_states'] = self.ledger.input_states().model_dump()
		return context

	def _decision_policy(self) -> str:
		return (
			POLICY
			+ '\nUse purpose_conditions and observed_sources to continue the entire task. If completion_missing_condition is set, seek actual evidence for it before DONE. No parent is reading pages during normal exploration.'
			+ "\nClick and Enter use native browser interactions, including page event handlers. Links that request a new tab are followed in this task's tab. Read section choices acquire body text at observed headings. A relevant link or table of contents is not evidence of the linked document. DONE triggers evidence selection; seek body examples and displayed results before DONE."
			+ '\nobserved_input_states pairs controls and displayed text from the SAME observation. Compare successive states: a filled value alone is not applied, and old output beside a new input is not a new result. Preserve already observed input/output pairs; do not re-enter an earlier value merely to collect it again.'
		)

	def _completion_handoff_reason(self) -> str | None:
		"""A repeated denial is a blocked handoff, never a successful completion."""
		return 'completion_evidence_unchanged' if getattr(self, '_completion_stalled', False) else None

	def _completion_evidence_key(self, observation: Observation) -> str:
		"""Ignore fresh observation IDs, retaining source, state, pixels and action changes."""
		assert self.result
		sources = set()
		for source in self.ledger.evidence:
			data = {k: v for k, v in source.items() if k not in ('id', 'screenshot')}
			prefix = str(source.get('id', '')) + '-'
			data['targets'] = [{**t, 'ref': str(t.get('ref', '')).removeprefix(prefix)} for t in source.get('targets', [])]
			data['focused'] = str(source.get('focused', '')).removeprefix(prefix)
			sources.add(hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
		data = {
			'goal': self.result.task.model_dump(),
			'requirements': self.plan.model_dump(),
			'sources': sorted(sources),
			'current_url': observation.url,
			'pixels': hashlib.sha256((observation.screenshot or '').encode()).hexdigest(),
			'receipts': [h.model_dump() for h in self.result.history],
			'recovery_receipts': [r.model_dump() for r in self.result.recovery_receipts],
		}
		return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

	async def _accept_done(self, observation: Observation) -> bool:
		"""Check the whole goal across multiple original chunks, only on a proposed DONE."""
		import time

		assert self.result and self.decisions
		self._completion_stalled = False
		key = self._completion_evidence_key(observation)
		if key == getattr(self, '_denied_completion_key', None):
			self._completion_stalled = True
			await self._event('completion_handoff', missing=self.missing_condition, evidence_key=key)
			return False
		if self.result.budget.decisions_used >= self.result.budget.max_decisions:
			return False
		question = ChoiceQuestion(
			name='coverage',
			instructions='Check every unchanged requirement using ALL supplied original chunks and operation receipts. Multiple chunks may support one requirement. COMPLETE only if every requested browser outcome is actually observed. A heading, search modal, title, matching URL or typed unsubmitted input is insufficient. Check only the original task requirements. Output files and final prose are produced after exploration.',
			choices=[Choice(value='COMPLETE', description='All browser requirements are observed')]
			+ [Choice(value=c.id, description=c.requirement) for c in self.plan.conditions],
		)
		self.result.budget.decisions_used += 1
		answer = await self.decisions.choose(
			{
				'goal': self.result.task.goal,
				'evidence': self.ledger.packet(self.result.task.goal),
				'observed_input_states': self.ledger.input_states().model_dump(),
				'recent_actions': [h.model_dump() for h in self.result.history[-12:]],
			},
			question,
			deadline=time.monotonic() + self.result.budget.deadline_at - time.time(),
		)
		self.missing_condition = (
			None
			if answer.answer.choice == 'COMPLETE'
			else next((c.requirement for c in self.plan.conditions if c.id == answer.answer.choice), 'unresolved source coverage')
		)
		self._denied_completion_key = key if self.missing_condition is not None else None
		self.result.completion_missing_condition = self.missing_condition
		await self._event('completion_check', decision=answer.model_dump(), missing=self.missing_condition)
		return self.missing_condition is None
