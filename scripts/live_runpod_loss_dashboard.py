#!/usr/bin/env python3
"""Serve a tiny live dashboard for the active Runpod training log."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


LOSS_RE = re.compile(
    r"epoch=(?P<epoch>\d+) step=(?P<step>\d+)(?: scheduled_removal=(?P<removed>\d+)(?:/\d+)?)? "
    r"loss=(?P<loss>[0-9.eE+-]+)(?: steps_per_second=(?P<sps>[0-9.]+))?"
)
VAL_RE = re.compile(
    r"validation epoch=(?P<epoch>\d+)(?: removed=(?P<removed>\d+)/(?:\d+))?: "
    r"(?P<correct>\d+)/(?P<total>\d+) \((?P<pct>[0-9.]+)%\)"
)
TEST_RE = re.compile(r"test: (?P<correct>\d+)/(?P<total>\d+) \((?P<pct>[0-9.]+)%\)")
STAGE_RE = re.compile(r"stage=(?P<stage>\S+) (?P<event>started|finished)_at=(?P<when>\S+)")


class State:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.payload: dict = {"status": "starting", "updated_at": None, "stages": []}

    def resolve_endpoint(self) -> tuple[str, str]:
        result = subprocess.run(
            ["runpodctl", "ssh", "info", self.args.pod],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        match = re.search(
            r"(?P<host>(?:\d{1,3}\.){3}\d{1,3}).*?(?:-p\s+|:)(?P<port>\d+)",
            result.stdout,
            re.S,
        )
        if not match:
            raise RuntimeError(f"Could not parse endpoint: {result.stdout.strip()}")
        return match.group("host"), match.group("port")

    def fetch_log(self) -> str:
        host, port = self.resolve_endpoint()
        result = subprocess.run(
            [
                "ssh", "-i", self.args.key, "-p", port,
                "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                f"root@{host}", f"tail -n {self.args.lines} {self.args.log}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout

    @staticmethod
    def parse(text: str) -> dict:
        stages: list[dict] = []
        current: dict | None = None
        for line in text.splitlines():
            stage_match = STAGE_RE.search(line)
            if stage_match:
                if stage_match.group("event") == "started":
                    current = {
                        "name": stage_match.group("stage"),
                        "started_at": stage_match.group("when"),
                        "finished_at": None,
                        "points": [],
                        "validations": [],
                        "test": None,
                    }
                    stages.append(current)
                elif current is not None:
                    current["finished_at"] = stage_match.group("when")
                continue
            if current is None:
                continue
            loss_match = LOSS_RE.search(line)
            if loss_match:
                current["points"].append(
                    {
                        "epoch": int(loss_match.group("epoch")),
                        "step": int(loss_match.group("step")),
                        "removed": int(loss_match.group("removed")) if loss_match.group("removed") else None,
                        "loss": float(loss_match.group("loss")),
                        "sps": float(loss_match.group("sps")) if loss_match.group("sps") else None,
                    }
                )
                continue
            val_match = VAL_RE.search(line)
            if val_match:
                current["validations"].append(
                    {
                        "epoch": int(val_match.group("epoch")),
                        "removed": int(val_match.group("removed")) if val_match.group("removed") else None,
                        "correct": int(val_match.group("correct")),
                        "total": int(val_match.group("total")),
                        "accuracy": float(val_match.group("pct")),
                    }
                )
                continue
            test_match = TEST_RE.search(line)
            if test_match:
                current["test"] = {
                    "correct": int(test_match.group("correct")),
                    "total": int(test_match.group("total")),
                    "accuracy": float(test_match.group("pct")),
                }
        return {"status": "ok", "updated_at": time.time(), "stages": stages}

    def refresh_forever(self) -> None:
        while True:
            try:
                payload = self.parse(self.fetch_log())
            except Exception as exc:  # Keep serving the last good sample during endpoint changes.
                with self.lock:
                    payload = dict(self.payload)
                payload["status"] = f"refresh error: {exc}"
                payload["updated_at"] = time.time()
            with self.lock:
                self.payload = payload
            time.sleep(self.args.refresh)


HTML = r'''<!doctype html>
<meta charset="utf-8">
<title>Deep 1 · Live training</title>
<style>
  :root { color-scheme: dark; --bg:#0b1020; --panel:#121a2e; --ink:#ecf2ff; --muted:#8fa1c3; --cyan:#54d6ff; --green:#6ce5a5; --grid:#263451; }
  * { box-sizing:border-box } body { margin:0; font:15px system-ui,sans-serif; background:var(--bg); color:var(--ink) }
  main { max-width:1180px; margin:auto; padding:28px } h1 { margin:0 0 6px; font-size:25px } .sub { color:var(--muted); margin-bottom:20px }
  .cards { display:grid; grid-template-columns:repeat(4,minmax(150px,1fr)); gap:12px; margin-bottom:14px }
  .card,.panel { background:var(--panel); border:1px solid #22304c; border-radius:12px; padding:16px }
  .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em } .value { font-size:24px; margin-top:5px }
  .tabs { display:flex; gap:8px; flex-wrap:wrap; margin:15px 0 } button { background:#192540; color:var(--ink); border:1px solid #314264; padding:8px 12px; border-radius:8px; cursor:pointer }
  button.active { border-color:var(--cyan); color:var(--cyan) } canvas { width:100%; height:430px; display:block }
  table { width:100%; border-collapse:collapse; margin-top:10px } th,td { text-align:left; padding:8px; border-bottom:1px solid #22304c } th { color:var(--muted) }
  .good { color:var(--green) } @media(max-width:700px){.cards{grid-template-columns:1fr 1fr} main{padding:16px}}
</style>
<main>
  <h1>Deep 1 · live multiplication training</h1>
  <div class="sub" id="status">Connecting…</div>
  <div class="tabs" id="tabs"></div>
  <div class="cards">
    <div class="card"><div class="label">Stage</div><div class="value" id="stage">—</div></div>
    <div class="card"><div class="label">Latest loss</div><div class="value" id="loss">—</div></div>
    <div class="card"><div class="label">Removal</div><div class="value" id="removed">—</div></div>
    <div class="card"><div class="label">Throughput</div><div class="value" id="sps">—</div></div>
  </div>
  <div class="panel"><canvas id="chart"></canvas></div>
  <div class="panel" style="margin-top:14px"><div class="label">Accuracy checkpoints</div><table><thead><tr><th>Epoch</th><th>CoT removed</th><th>Validation</th></tr></thead><tbody id="vals"></tbody></table><div id="test"></div></div>
</main>
<script>
let selected=null, payload=null;
const $=id=>document.getElementById(id);
function draw(stage){
  const canvas=$('chart'), dpr=devicePixelRatio||1, rect=canvas.getBoundingClientRect(); canvas.width=rect.width*dpr; canvas.height=rect.height*dpr;
  const c=canvas.getContext('2d'); c.scale(dpr,dpr); const W=rect.width,H=rect.height,p={l:62,r:22,t:22,b:42}; c.clearRect(0,0,W,H);
  const pts=stage.points||[]; if(!pts.length){c.fillStyle='#8fa1c3';c.fillText('Waiting for loss samples…',p.l,p.t+20);return}
  const losses=pts.map(x=>Math.max(x.loss,1e-8)), logs=losses.map(Math.log10), lo=Math.min(...logs), hi=Math.max(...logs), span=Math.max(.2,hi-lo);
  c.strokeStyle='#263451'; c.fillStyle='#8fa1c3'; c.font='12px system-ui'; c.lineWidth=1;
  for(let i=0;i<=5;i++){const y=p.t+(H-p.t-p.b)*i/5;c.beginPath();c.moveTo(p.l,y);c.lineTo(W-p.r,y);c.stroke();const v=Math.pow(10,hi-span*i/5);c.fillText(v.toPrecision(2),5,y+4)}
  c.strokeStyle='#54d6ff';c.lineWidth=2;c.beginPath();pts.forEach((pt,i)=>{const x=p.l+(W-p.l-p.r)*i/Math.max(1,pts.length-1),y=p.t+(H-p.t-p.b)*(hi-Math.log10(Math.max(pt.loss,1e-8)))/span;i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();
  c.fillStyle='#8fa1c3';c.fillText('training progress →',W/2-45,H-12);c.save();c.translate(15,H/2+30);c.rotate(-Math.PI/2);c.fillText('loss (log scale)',0,0);c.restore();
}
function render(){
  if(!payload||!payload.stages.length)return; const tabs=$('tabs'); tabs.innerHTML='';
  if(selected===null||selected>=payload.stages.length)selected=payload.stages.length-1;
  payload.stages.forEach((s,i)=>{const b=document.createElement('button');b.textContent=s.name.replaceAll('_',' ');b.className=i===selected?'active':'';b.onclick=()=>{selected=i;render()};tabs.appendChild(b)});
  const s=payload.stages[selected], last=s.points.at(-1)||{}; $('stage').textContent=s.finished_at?'complete':'running'; $('loss').textContent=last.loss?.toPrecision(4)??'—'; $('removed').textContent=last.removed==null?'—':last.removed; $('sps').textContent=last.sps?last.sps.toFixed(2)+'/s':'—';
  $('vals').innerHTML=(s.validations||[]).map(v=>`<tr><td>${v.epoch}</td><td>${v.removed??'—'}</td><td class="${v.accuracy>=99?'good':''}">${v.accuracy.toFixed(1)}%</td></tr>`).join('')||'<tr><td colspan="3">No validation checkpoint yet</td></tr>';
  $('test').innerHTML=s.test?`<p><b>Test:</b> <span class="good">${s.test.accuracy.toFixed(1)}%</span></p>`:''; draw(s);
}
async function poll(){try{payload=await fetch('/data',{cache:'no-store'}).then(r=>r.json());$('status').textContent=`${payload.status} · updated ${new Date(payload.updated_at*1000).toLocaleTimeString()} · refreshes every 10 seconds`;render()}catch(e){$('status').textContent='Waiting for dashboard service…'}setTimeout(poll,10000)}
addEventListener('resize',render);poll();
</script>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh", type=int, default=10)
    parser.add_argument("--lines", type=int, default=10000)
    args = parser.parse_args()
    state = State(args)
    threading.Thread(target=state.refresh_forever, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/data":
                with state.lock:
                    body = json.dumps(state.payload).encode()
                content_type = "application/json"
            else:
                body = HTML.encode()
                content_type = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
