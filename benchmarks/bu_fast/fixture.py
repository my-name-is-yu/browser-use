"""Deterministic application and independent saved-state oracle for comparisons."""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HTML = """<!doctype html><html><head><title>Browser benchmark</title>
<style>body{font:18px system-ui;padding:32px;max-width:800px} input{display:block;padding:8px;margin:12px 0} button{padding:10px;margin:8px} .plans{display:flex;gap:16px}.plan{border:1px solid #ccc;padding:16px}</style></head><body>
<h1>Team subscription</h1><p>Enter Alice and Tokyo. Choose the cheapest plan supporting at least 3 people, then save.</p>
<label>Name<input id="name" aria-label="Name"></label><label>City<input id="city" aria-label="City"></label>
<div class="plans"><div class="plan" data-record-id="solo"><h2>Solo</h2><p>1 person · $10</p><button id="solo" onclick="select('solo')">Select Solo</button></div>
<div class="plan" data-record-id="team"><h2>Team</h2><p>3 people · $24</p><button id="team" onclick="select('team')">Select Team</button></div>
<div class="plan" data-record-id="business"><h2>Business</h2><p>10 people · $80</p><button id="business" onclick="select('business')">Select Business</button></div></div>
<p id="selected">No plan selected</p><button id="save" onclick="save()">Save subscription</button><p id="status" role="status"></p>
<script>let plan=null; window.saved=null;
function select(p){plan=p;document.querySelector('#selected').textContent='Selected: '+p;}
async function save(){const data={name:document.querySelector('#name').value,city:document.querySelector('#city').value,plan};
const response=await fetch(location.pathname+'/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
window.saved=await response.json();document.querySelector('#status').textContent='Saved: '+window.saved.name+', '+window.saved.city+', '+window.saved.plan;}</script></body></html>"""


class FixtureHandler(BaseHTTPRequestHandler):
	runs: dict = {}

	def log_message(self, *_args):
		pass

	def respond(self, body, content_type='application/json', status=200):
		encoded = body.encode('utf-8')
		self.send_response(status)
		self.send_header('Content-Type', content_type + '; charset=utf-8')
		self.send_header('Content-Length', str(len(encoded)))
		self.end_headers()
		self.wfile.write(encoded)

	def do_GET(self):
		path = urlparse(self.path).path
		if path.startswith('/run/') and not path.endswith('/state'):
			self.runs[path] = {'started_at': time.monotonic(), 'saves': 0, 'state': None}
			self.respond(HTML, 'text/html')
		elif path.endswith('/state'):
			run = self.runs.get(path.removesuffix('/state'), {})
			self.respond(json.dumps(run))
		else:
			self.respond('{}', status=404)

	def do_POST(self):
		path = urlparse(self.path).path.removesuffix('/save')
		run = self.runs.get(path)
		if run is None:
			self.respond('{}', status=404)
			return
		state = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
		run['saves'] += 1
		run['state'] = state
		run['elapsed_ms'] = (time.monotonic() - run['started_at']) * 1000
		self.respond(json.dumps(state))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--port', type=int, default=8765)
	args = parser.parse_args()
	server = ThreadingHTTPServer(('127.0.0.1', args.port), FixtureHandler)
	print(f'Fixture ready on http://127.0.0.1:{server.server_port}', flush=True)
	server.serve_forever()


if __name__ == '__main__':
	main()
