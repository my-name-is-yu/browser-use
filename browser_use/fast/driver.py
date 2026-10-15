"""Browser observation and native operations for the goal loop."""

import asyncio
import json
from urllib.parse import quote, urljoin, urlparse
from uuid import uuid4

from cdp_use.cdp.runtime.types import CallArgument

from browser_use.actor.page import Page
from browser_use.dom.serializer.serializer import DOMTreeSerializer
from browser_use.dom.views import NodeType
from browser_use.fast.views import Action, DisplayChange, Observation
from browser_use.fast.workflow_support import native_selector_element

OBSERVE = r"""(cfg, ...nodes) => {
const norm = s => (s || '').replace(/\s+/g, ' ').trim();
const rendered = e => {if(!e)return false;const r=e.getBoundingClientRect(),s=getComputedStyle(e);
return r.width>0&&r.height>0&&s.visibility!=='hidden'&&s.display!=='none'&&!e.closest('[inert],[aria-hidden="true"]')};
const onview = e => {const r=e.getBoundingClientRect();return r.bottom>0&&r.top<innerHeight&&r.right>0&&r.left<innerWidth};
const rootsOf=root=>{const roots=[root];for(let i=0;i<roots.length;i++)for(const e of roots[i].querySelectorAll('*'))if(e.shadowRoot)roots.push(e.shadowRoot);return roots};
const deep=(root,selector)=>rootsOf(root).flatMap(r=>[...r.querySelectorAll(selector)]);
const textOf=root=>rootsOf(root).map(r=>r.innerText??[...r.children].filter(rendered).map(e=>e.innerText||'').join('\n')).join('\n');
document.__buFocused=()=>{let e=document.activeElement;while(e?.shadowRoot?.activeElement)e=e.shadowRoot.activeElement;return e};
document.__buDocument ||= crypto.randomUUID();
document.__buTargets?.forEach(e=>e.removeAttribute('data-bu-ref'));document.__buTargets=new Map();
const modal=deep(document,'dialog[open],[role=dialog],[aria-modal=true]').filter(rendered).at(-1);
const root=modal||document.body||document;
const state=e=>[e.getAttribute('aria-expanded'),e.getAttribute('aria-selected'),e.getAttribute('aria-pressed'),e.getAttribute('aria-checked'),typeof e.checked==='boolean'?String(e.checked):''].join('|');
let all=nodes.filter(e=>{
let scope=e;while(scope&&!root.contains(scope))scope=scope.getRootNode().host;
const r=e.getBoundingClientRect();return !!scope&&e.isConnected&&rendered(e)&&r.right>0&&r.left<innerWidth&&!e.disabled&&e.type!=='hidden'});
all.sort((a,b)=>Number(onview(b))-Number(onview(a)));
const groups=new Map();
const label=e=>norm(e.getAttribute('aria-label')||e.getAttribute('title')||e.getAttribute('placeholder')||e.innerText||e.getAttribute('name')||[...e.querySelectorAll('img[alt]')].map(x=>x.alt).join(' ')).slice(0,200);
const targets=all.slice(0,180).map(e=>{
const ref=cfg.id+'-'+nodes.indexOf(e);e.setAttribute('data-bu-ref',ref);document.__buTargets.set(ref,e);
const p=e.closest('form,tr,[role=row],[role=dialog],[data-record-id],li')||e.parentElement||e.getRootNode().host;
if(!groups.has(p))groups.set(p,'g'+groups.size);
return {ref,tag:e.tagName.toLowerCase(),type:e.type||'',label:label(e),context:norm(p?.innerText).slice(0,600),href:e.getAttribute('href')||'',value:e.type==='password'?'':e.value||'',group:groups.get(p),state:state(e),options:e.tagName==='SELECT'?[...e.options].filter(o=>!o.disabled).slice(0,240).map(o=>o.value):[],option_labels:e.tagName==='SELECT'?Object.fromEntries([...e.options].filter(o=>!o.disabled).slice(0,240).map(o=>[o.value,norm(o.label)])):{},new_tab:e.getAttribute('target')==='_blank'};
});
let texts=[],count=0;
for(const region of rootsOf(root)){const w=document.createTreeWalker(region,NodeFilter.SHOW_TEXT);let n;
while((n=w.nextNode())&&count++<20000){
if(!norm(n.textContent)||!rendered(n.parentElement))continue;
const range=document.createRange();range.selectNodeContents(n);const r=range.getBoundingClientRect();
let top=Math.max(0,r.top),bottom=Math.min(innerHeight,r.bottom),left=Math.max(0,r.left),right=Math.min(innerWidth,r.right);
for(let e=n.parentElement;e;e=e.parentElement||e.getRootNode().host){
const s=getComputedStyle(e),b=e.getBoundingClientRect();
if(['hidden','clip','scroll','auto'].includes(s.overflowY)){top=Math.max(top,b.top+e.clientTop);bottom=Math.min(bottom,b.top+e.clientTop+e.clientHeight)}
if(['hidden','clip','scroll','auto'].includes(s.overflowX)){left=Math.max(left,b.left+e.clientLeft);right=Math.min(right,b.left+e.clientLeft+e.clientWidth)}
}
if(bottom>top&&right>left)texts.push(norm(n.textContent));
}
}
let fragment='';
try{
const e=document.getElementById(decodeURIComponent(location.hash.slice(1)));
if(e&&!modal){
const h=/^H[1-6]$/.test(e.tagName)?e:e.closest('h1,h2,h3,h4,h5,h6');
if(h){let parts=[h.innerText],next=h.nextElementSibling,level=Number(h.tagName[1]);
while(next){if(/^H[1-6]$/.test(next.tagName)&&Number(next.tagName[1])<=level)break;parts.push(next.innerText||'');next=next.nextElementSibling;if(parts.join(' ').length>12000)break}
fragment=norm(parts.join(' ')).slice(0,12000);
}else fragment=norm((e.closest('dl,section')||e.parentElement)?.innerText).slice(0,12000);
}
}catch{}
const fullText=textOf(root);
document.__buDisplaySnapshot={id:cfg.id,text:fullText,read:()=>textOf(root.isConnected?root:(document.body||document)),roots:()=>rootsOf(root)};
const headings=deep(root,'h1,h2,h3,h4,h5,h6,dt').filter(rendered).slice(0,2000).map(e=>({text:norm(e.innerText),id:e.id||''}));
const links=deep(root,'a[href]').filter(rendered).slice(0,2000).map(e=>({text:norm(e.innerText),href:e.href,context:norm((e.closest('li,tr,[role=row]')||e.parentElement)?.innerText).slice(0,1000)}));
return {id:cfg.id,document_id:document.__buDocument,target_id:cfg.target,url:location.href,title:document.title,text:fullText.slice(0,200000),text_truncated:fullText.length>200000,viewport_text:texts.join('\n').slice(0,12000),fragment_text:fragment,targets,total_targets:all.length,focused:document.__buFocused()?.getAttribute('data-bu-ref')||'',scroll_y:scrollY,modal:!!modal,ready:document.readyState,headings,links};
}"""


class StaleTarget(RuntimeError):
	"""The observed node or document changed before dispatch."""


class BrowserDriver:
	"""Observe current targets, apply native interactions and collect direct evidence."""

	def __init__(self, page: Page, target_id: str):
		self.page = page
		self.target_id = target_id
		self._target_nodes: dict[str, int] = {}

	async def observe(self, *, screenshot: bool = False) -> Observation:
		"""Capture current controls and source evidence; screenshots are optional."""
		config = {'id': uuid4().hex, 'target': self.target_id}
		dom_service = self.page.dom_service
		tree, _ = await dom_service.get_dom_tree(target_id=self.target_id)
		state, _ = DOMTreeSerializer(
			tree, None, paint_order_filtering=True, session_id=dom_service.browser_session.id
		).serialize_accessible_elements()
		# Keep the existing document/shadow-root scope; frame recovery remains a separate operation.
		nodes = []
		for node in state.selector_map.values():
			owner = node
			while owner.parent_node and owner.node_type != NodeType.DOCUMENT_NODE:
				owner = owner.parent_node
			if owner is tree:
				nodes.append(node)
		client = dom_service.browser_session.cdp_client
		session_id = await self.page.session_id
		# Actor operations need the document requested in the Page's own CDP session.
		await client.send.DOM.getDocument({'depth': 0, 'pierce': True}, session_id=session_id)
		# Pass OSS candidates to the evidence snapshot as DOM objects, without enumerating them again.
		try:
			resolved = await asyncio.gather(
				*(
					client.send.DOM.resolveNode(
						{'backendNodeId': node.backend_node_id, 'objectGroup': config['id']}, session_id=session_id
					)
					for node in nodes
				),
				return_exceptions=True,
			)
			bindings: list[tuple[int, str]] = []
			for node, resolved_node in zip(nodes, resolved):
				if isinstance(resolved_node, BaseException):
					raise resolved_node
				if object_id := resolved_node['object'].get('objectId'):
					bindings.append((node.backend_node_id, object_id))
			global_object = await client.send.Runtime.evaluate(
				{'expression': 'globalThis', 'objectGroup': config['id']}, session_id=session_id
			)
			global_id = global_object['result'].get('objectId')
			if not global_id:
				raise StaleTarget('Document was replaced')
			arguments: list[CallArgument] = [{'value': config}, *({'objectId': object_id} for _, object_id in bindings)]
			result = await client.send.Runtime.callFunctionOn(
				{
					'functionDeclaration': 'function(cfg, ...nodes) {return (' + OBSERVE + ')(cfg, ...nodes)}',
					'objectId': global_id,
					'arguments': arguments,
					'returnByValue': True,
				},
				session_id=session_id,
			)
			observation = Observation.model_validate(result['result'].get('value'))
			self._target_nodes = {config['id'] + '-' + str(i): node_id for i, (node_id, _) in enumerate(bindings)}
		finally:
			await client.send.Runtime.releaseObjectGroup({'objectGroup': config['id']}, session_id=session_id)
		if screenshot:
			observation.screenshot = await self.page.screenshot()
		return observation

	async def watch_display_change(self, observation: Observation) -> DisplayChange:
		"""Race visible text changes against a decision; never wait for a quiet interval."""
		try:
			result = await self.page.evaluate(
				r"""(c) => {
const source=document.__buDisplaySnapshot;
if(!source||source.id!==c.id||document.__buDocument!==c.document_id||location.href!==c.url)return {reason:'observation_replaced'};
document.__buDisplayWatch?.stop();
return new Promise(resolve=>{
let ended=false;
const finish=reason=>{if(ended)return;ended=true;observer.disconnect();
if(document.__buDisplayWatch?.id===c.id)delete document.__buDisplayWatch;resolve({reason})};
const check=()=>{if(source.read()!==source.text)finish('changed')};
const observer=new MutationObserver(check);
const options={subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['hidden','style','class','aria-hidden','open']};
observer.observe(document,options);
for(const root of source.roots())observer.observe(root,options);
document.__buDisplayWatch={id:c.id,stop:()=>finish('stopped')};
check();
});
}""",
				{'id': observation.id, 'document_id': observation.document_id, 'url': observation.url},
			)
			return DisplayChange.model_validate_json(result)
		except Exception:
			# A destroyed document or failed watcher cannot validate an old decision.
			return DisplayChange(reason='monitor_failed')

	async def stop_display_watch(self, observation: Observation) -> None:
		"""Release the page observer and its promise when the decision wins the race."""
		try:
			async with asyncio.timeout(1):
				await self.page.evaluate(
					'(id) => {if(document.__buDisplayWatch?.id===id)document.__buDisplayWatch.stop()}', observation.id
				)
		except Exception:
			pass  # Navigation may already have destroyed the watched document.

	async def execute(self, action: Action, observation: Observation) -> None:
		"""Rebind the observed node, then click natively so SPA event handlers execute."""
		if action.observation_id != observation.id:
			raise StaleTarget('Observation changed')
		fresh = await self.page.evaluate(
			'(c) => ({passed: document.__buDocument === c.document && location.href === c.url && (!c.enter || (document.__buFocused()?.getAttribute("data-bu-ref") || "") === c.focused)})',
			{
				'document': observation.document_id,
				'url': observation.url,
				'enter': action.op == 'enter',
				'focused': observation.focused,
			},
		)
		if not json.loads(fresh)['passed']:
			raise StaleTarget('Document or focused field changed')
		if action.op == 'navigate':
			link = next((link for link in observation.links if link['href'] == action.text), None)
			if not link or urlparse(action.text).scheme not in ('http', 'https'):
				raise StaleTarget('Navigation URL was not observed')
			valid = await self.page.evaluate(
				r"""(c) => {
const norm=s=>(s||'').replace(/\s+/g,' ').trim(),roots=[document];
for(let i=0;i<roots.length;i++)for(const e of roots[i].querySelectorAll('*'))if(e.shadowRoot)roots.push(e.shadowRoot);
const links=[...new Set(roots.flatMap(r=>[...r.querySelectorAll('a[href]')]))];
const index=links.findIndex(e=>e.href===c.href&&norm(e.innerText)===c.text&&norm((e.closest('li,tr,[role=row]')||e.parentElement)?.innerText).slice(0,1000)===c.context);
return {passed:index>=0,index,new_tab:index>=0&&links[index].getAttribute('target')==='_blank'};
}""",
				link,
			)
			bound = json.loads(valid)
			if not bound['passed']:
				raise StaleTarget('Observed link meaning changed')
			if bound['new_tab']:
				await self.page.goto(action.text)
			else:
				element = await native_selector_element(self.page, 'a[href]', bound['index'])
				await element.click()
			return
		if action.ref:
			target = next((t for t in observation.targets if t.ref == action.ref), None)
			if target is None:
				raise StaleTarget('Target is not in this observation')
			valid = await self.page.evaluate(
				r"""(c) => {
const e=document.__buTargets?.get(c.ref),norm=s=>(s||'').replace(/\s+/g,' ').trim();
if(!e||!e.isConnected||document.__buDocument!==c.document_id||e.disabled)return {passed:false};
const label=norm(e.getAttribute('aria-label')||e.getAttribute('title')||e.getAttribute('placeholder')||e.innerText||e.getAttribute('name')||[...e.querySelectorAll('img[alt]')].map(x=>x.alt).join(' ')).slice(0,200);
const p=e.closest('form,tr,[role=row],[role=dialog],[data-record-id],li')||e.parentElement||e.getRootNode().host;
const state=[e.getAttribute('aria-expanded'),e.getAttribute('aria-selected'),e.getAttribute('aria-pressed'),e.getAttribute('aria-checked'),typeof e.checked==='boolean'?String(e.checked):''].join('|');
return {passed:e.tagName.toLowerCase()===c.tag&&label===c.label&&(e.getAttribute('href')||'')===c.href&&norm(p?.innerText).slice(0,600)===c.context&&state===c.state&&(!['input','textarea','select'].includes(c.tag)||(e.value||'')===c.value)};
}""",
				{**target.model_dump(), 'document_id': observation.document_id},
			)
			if not json.loads(valid)['passed']:
				raise StaleTarget('Node meaning or document changed')
			if action.op == 'click' and target.new_tab and target.href:
				url = urljoin(observation.url, target.href)
				if urlparse(url).scheme in ('http', 'https'):
					await self.page.goto(url)
					return
			node_id = self._target_nodes.get(action.ref)
			if node_id is None:
				raise StaleTarget('Target is not bound to this driver')
			element = await self.page.get_element(node_id)
			if action.op == 'fill':
				await element.fill(action.text)
				await self.page.evaluate('(c) => document.__buTargets.get(c.ref).focus()', {'ref': action.ref})
			elif action.op == 'select':
				await element.select_option(action.text)
			elif action.op == 'click':
				await element.click()
			else:
				raise ValueError('Unsupported target operation')
		elif action.op == 'enter':
			if observation.focused:
				target = next((t for t in observation.targets if t.ref == observation.focused), None)
				valid = await self.page.evaluate(
					'(c) => {const e=document.__buTargets?.get(c.ref);return {passed:!!e&&e.isConnected&&(e.value||"")===c.value}}',
					{'ref': observation.focused, 'value': target.value if target else ''},
				)
				if target is None or not json.loads(valid)['passed']:
					raise StaleTarget('Focused input changed before form submission')
			await self.page.press('Enter')
		elif action.op == 'read_page':
			if not any(h.get('id') == action.text for h in observation.headings):
				raise StaleTarget('Section was not observed')
			await self.page.evaluate(
				'(c) => {const e=document.getElementById(c.id);if(!e)throw Error("Section disappeared");for(let p=e;p;p=p.parentElement)if(p.tagName==="DETAILS")p.open=true;e.scrollIntoView();}',
				{'id': action.text},
			)
			await self.page.goto(observation.url.split('#')[0] + '#' + quote(action.text))
		elif action.op == 'escape':
			await self.page.press('Escape')
		elif action.op == 'back':
			await self.page.go_back()
		elif action.op in ('scroll_down', 'scroll_up'):
			amount = '.8' if action.op == 'scroll_down' else '-.8'
			await self.page.evaluate('() => window.scrollBy(0,innerHeight*' + amount + ')')
		elif action.op == 'wait':
			await asyncio.sleep(0.5)
		else:
			raise ValueError('Unsupported browser operation')


def build_actions(observation: Observation, inputs: list[str]) -> list[Action]:
	"""Bind complete choices to observed nodes and supplied strings."""
	actions = []

	def add(op, ref='', text='', description=''):
		actions.append(
			Action(id=str(len(actions)), observation_id=observation.id, op=op, ref=ref, text=text, description=description)
		)

	for t in observation.targets:
		if t.tag in ('input', 'textarea') and t.type not in (
			'submit',
			'button',
			'checkbox',
			'radio',
			'file',
			'hidden',
			'password',
		):
			for value in inputs:
				if value != t.value:
					add('fill', t.ref, value, f'Fill {t.label!r} with {value!r}; context={t.context}')
			add(
				'needs_input',
				t.ref,
				description=f'Ask Codex for NEW text for {t.label!r}; current={t.value!r}; context={t.context}. Choose only if no supplied value fits.',
			)
		elif t.tag == 'select':
			for value in t.options:
				if value != t.value:
					add('select', t.ref, value, f'Select {t.option_labels.get(value, value)!r} in {t.label!r}')
		else:
			add('click', t.ref, description=f'Click {t.label!r}; href={t.href}; context={t.context}')
	for op, desc in {
		'enter': 'Submit the focused field',
		'escape': 'Close the active menu',
		'back': 'Go back',
		'scroll_down': 'Scroll down',
		'scroll_up': 'Scroll up',
		'wait': 'Wait for rendering',
		'done': 'Every requested browser outcome has been observed; finish',
		'blocked': 'No supported operation can advance this goal',
	}.items():
		add(op, description=desc)
	return actions
