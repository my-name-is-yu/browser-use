"""Browser observation and native operations for the goal loop."""

import asyncio
import json
from urllib.parse import urljoin, urlparse
from uuid import uuid4

from browser_use.actor.page import Page
from browser_use.fast.views import Action, Observation

OBSERVE = r"""(cfg) => {
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
let all=deep(root,'a[href],button,input,textarea,select,summary,[role=button],[role=option],[role=checkbox],[role=tab]').filter(e=>{
const r=e.getBoundingClientRect();return rendered(e)&&r.right>0&&r.left<innerWidth&&!e.disabled&&e.type!=='hidden'});
all.sort((a,b)=>Number(onview(b))-Number(onview(a)));
const groups=new Map();
const label=e=>norm(e.getAttribute('aria-label')||e.getAttribute('title')||e.getAttribute('placeholder')||e.innerText||e.getAttribute('name')||[...e.querySelectorAll('img[alt]')].map(x=>x.alt).join(' ')).slice(0,200);
const targets=all.slice(0,180).map((e,i)=>{
const ref=cfg.id+'-'+i;e.setAttribute('data-bu-ref',ref);document.__buTargets.set(ref,e);
const p=e.closest('form,tr,[role=row],[role=dialog],[data-record-id],li')||e.parentElement||e.getRootNode().host;
if(!groups.has(p))groups.set(p,'g'+groups.size);
return {ref,tag:e.tagName.toLowerCase(),type:e.type||'',label:label(e),context:norm(p?.innerText).slice(0,600),href:e.getAttribute('href')||'',value:e.type==='password'?'':e.value||'',group:groups.get(p),state:state(e),options:e.tagName==='SELECT'?[...e.options].filter(o=>!o.disabled).slice(0,240).map(o=>o.value):[],option_labels:e.tagName==='SELECT'?Object.fromEntries([...e.options].filter(o=>!o.disabled).slice(0,240).map(o=>[o.value,norm(o.label)])):{},new_tab:e.getAttribute('target')==='_blank'};
});
let texts=[],count=0;
for(const region of rootsOf(root)){const w=document.createTreeWalker(region,NodeFilter.SHOW_TEXT);let n;
while((n=w.nextNode())&&count++<20000){
if(!norm(n.textContent)||!rendered(n.parentElement))continue;
const range=document.createRange();range.selectNodeContents(n);const r=range.getBoundingClientRect();
if(r.height>0&&r.bottom>0&&r.top<innerHeight&&r.right>0&&r.left<innerWidth)texts.push(norm(n.textContent));
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

	async def observe(self, *, screenshot: bool = False) -> Observation:
		"""Capture current controls and source evidence; screenshots are optional."""
		config = {'id': uuid4().hex, 'target': self.target_id}
		observation = Observation.model_validate_json(await self.page.evaluate(OBSERVE, config))
		if screenshot:
			observation.screenshot = await self.page.screenshot()
		return observation

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
			client = self.page.dom_service.browser_session.cdp_client
			session_id = await self.page.session_id
			await client.send.DOM.getDocument({'depth': 0, 'pierce': True}, session_id=session_id)
			node = await client.send.Runtime.evaluate(
				{'expression': 'document.__buTargets?.get(' + json.dumps(action.ref) + ')', 'returnByValue': False},
				session_id=session_id,
			)
			object_id = node.get('result', {}).get('objectId')
			if not object_id:
				raise StaleTarget('Target was replaced')
			try:
				description = await client.send.DOM.describeNode({'objectId': object_id}, session_id=session_id)
				element = await self.page.get_element(description['node']['backendNodeId'])
			finally:
				await client.send.Runtime.releaseObject({'objectId': object_id}, session_id=session_id)
			if action.op == 'fill':
				await element.fill(action.text)
			elif action.op == 'select':
				await element.select_option(action.text)
			elif target.new_tab and target.href and urlparse(urljoin(observation.url, target.href)).scheme in ('http', 'https'):
				await self.page.goto(urljoin(observation.url, target.href))
			else:
				await element.click()
		elif action.op == 'enter':
			await self.page.press('Enter')
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
