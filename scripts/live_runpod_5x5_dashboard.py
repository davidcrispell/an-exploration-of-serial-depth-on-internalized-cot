#!/usr/bin/env python3
"""Serve separate live monitors for Deep 5x5 CoT training."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


LOSS_RE = re.compile(
    r"epoch=(?P<epoch>\d+) step=(?P<step>\d+)"
    r"(?: scheduled_removal=(?P<removed>\d+)(?:/\d+)?)? "
    r"loss=(?P<loss>[0-9.eE+-]+)(?: steps_per_second=(?P<sps>[0-9.]+))?"
)
VAL_RE = re.compile(
    r"validation epoch=(?P<epoch>\d+)(?: removed=(?P<removed>\d+)/(?:\d+))?: "
    r"(?P<correct>\d+)/(?P<total>\d+) \((?P<pct>[0-9.]+)%\)"
)
TEST_RE = re.compile(
    r"test(?: removed=(?P<removed>\d+)/(?:\d+))?: "
    r"(?P<correct>\d+)/(?P<total>\d+) \((?P<pct>[0-9.]+)%\)"
)
STAGE_RE = re.compile(r"stage=(?P<stage>\S+) (?P<event>started|finished)_at=(?P<when>\S+)")


class State:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.payload: dict = {"status": "starting", "updated_at": None, "stages": []}

    def resolve_endpoint(self) -> tuple[str, str]:
        result = subprocess.run(
            ["runpodctl", "pod", "get", self.args.pod],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        try:
            pod = json.loads(result.stdout)
            ssh = pod["ssh"]
            return str(ssh["ip"]), str(ssh["port"])
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
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
                "ssh",
                "-i",
                self.args.key,
                "-p",
                port,
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=10",
                f"root@{host}",
                f"tail -n {self.args.lines} {self.args.log}",
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
                        "removed": (
                            int(loss_match.group("removed"))
                            if loss_match.group("removed")
                            else None
                        ),
                        "loss": float(loss_match.group("loss")),
                        "sps": (
                            float(loss_match.group("sps"))
                            if loss_match.group("sps")
                            else None
                        ),
                    }
                )
                continue
            val_match = VAL_RE.search(line)
            if val_match:
                current["validations"].append(
                    {
                        "epoch": int(val_match.group("epoch")),
                        "removed": (
                            int(val_match.group("removed"))
                            if val_match.group("removed")
                            else 0
                        ),
                        "correct": int(val_match.group("correct")),
                        "total": int(val_match.group("total")),
                        "accuracy": float(val_match.group("pct")),
                    }
                )
                continue
            test_match = TEST_RE.search(line)
            if test_match:
                current["test"] = {
                    "removed": (
                        int(test_match.group("removed"))
                        if test_match.group("removed")
                        else 0
                    ),
                    "correct": int(test_match.group("correct")),
                    "total": int(test_match.group("total")),
                    "accuracy": float(test_match.group("pct")),
                }
        return {"status": "ok", "updated_at": time.time(), "stages": stages}

    def refresh_forever(self) -> None:
        while True:
            try:
                payload = self.parse(self.fetch_log())
            except Exception as exc:
                with self.lock:
                    payload = dict(self.payload)
                payload["status"] = f"refresh error: {exc}"
                payload["updated_at"] = time.time()
            with self.lock:
                self.payload = payload
            time.sleep(self.args.refresh)


HTML = r'''<!doctype html>
<meta charset="utf-8">
<title>Deep 5x5 internalization</title>
<style>
  :root { color-scheme: dark; --bg:#0b1020; --panel:#121a2e; --ink:#ecf2ff; --muted:#8fa1c3; --cyan:#54d6ff; --green:#6ce5a5; --orange:#ffbd69; --grid:#263451; }
  * { box-sizing:border-box } body { margin:0; font:14px system-ui,sans-serif; background:var(--bg); color:var(--ink) }
  main { max-width:1240px; margin:auto; padding:24px } h1 { margin:0 0 5px; font-size:24px } h2 { margin:0 0 10px; font-size:16px; font-weight:500 }
  .sub { color:var(--muted); margin-bottom:16px } .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px }
  .panel { background:var(--panel); border:1px solid #22304c; border-radius:12px; padding:14px; min-width:0 }
  .wide { grid-column:1/-1 } canvas { width:100%; height:300px; display:block } .wide canvas { height:340px }
  .metric { color:var(--muted); margin-bottom:6px } .metric b { color:var(--ink); font-weight:500 }
  table { width:100%; border-collapse:collapse; margin-top:8px } th,td { text-align:right; padding:7px; border-bottom:1px solid #22304c }
  th:first-child,td:first-child { text-align:left } th { color:var(--muted); font-weight:400 }
  @media(max-width:760px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}main{padding:14px}}
</style>
<main>
  <h1>Deep 1 · 5×5 CoT internalization</h1>
  <div class="sub" id="status">Connecting…</div>
  <div class="grid">
    <section class="panel"><h2>Explicit-CoT fine-tuning loss</h2><div class="metric" id="explicitMetric">Waiting for samples…</div><canvas id="explicitChart"></canvas></section>
    <section class="panel"><h2>Token-internalization rollout loss</h2><div class="metric" id="internalMetric">Waiting for rollout…</div><canvas id="internalChart"></canvas></section>
    <section class="panel wide"><h2>Validation accuracy by CoT tokens removed</h2><div class="metric" id="accuracyMetric">Validation runs at each standard epoch checkpoint; 75/75 is answer-only.</div><canvas id="accuracyChart"></canvas><table><thead><tr><th>Epoch</th><th>Removed</th><th>Validation</th></tr></thead><tbody id="accuracyRows"><tr><td colspan="3">No validation checkpoint yet</td></tr></tbody></table></section>
  </div>
</main>
<script>
let payload=null; const $=id=>document.getElementById(id);
function setup(id){const el=$(id),d=devicePixelRatio||1,r=el.getBoundingClientRect();el.width=r.width*d;el.height=r.height*d;const c=el.getContext('2d');c.scale(d,d);return {c,W:r.width,H:r.height}}
function empty(c,msg){c.fillStyle='#8fa1c3';c.font='12px system-ui';c.fillText(msg,58,34)}
function lossChart(id,pts,color){const {c,W,H}=setup(id),p={l:58,r:18,t:18,b:36};c.clearRect(0,0,W,H);if(!pts.length){empty(c,'Waiting for loss samples…');return}const logs=pts.map(x=>Math.log10(Math.max(x.loss,1e-8))),hi=Math.max(...logs),lo=Math.min(...logs),span=Math.max(.2,hi-lo);c.font='11px system-ui';c.strokeStyle='#263451';c.fillStyle='#8fa1c3';c.lineWidth=1;for(let i=0;i<=4;i++){const y=p.t+(H-p.t-p.b)*i/4;c.beginPath();c.moveTo(p.l,y);c.lineTo(W-p.r,y);c.stroke();c.fillText(Math.pow(10,hi-span*i/4).toPrecision(2),4,y+4)}c.strokeStyle=color;c.lineWidth=2;c.beginPath();pts.forEach((q,i)=>{const x=p.l+(W-p.l-p.r)*i/Math.max(1,pts.length-1),y=p.t+(H-p.t-p.b)*(hi-Math.log10(Math.max(q.loss,1e-8)))/span;i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();c.fillStyle='#8fa1c3';c.fillText('training progress →',W/2-42,H-10)}
function accuracyChart(vals){const {c,W,H}=setup('accuracyChart'),p={l:58,r:24,t:18,b:42};c.clearRect(0,0,W,H);c.font='11px system-ui';c.strokeStyle='#263451';c.fillStyle='#8fa1c3';for(let a=0;a<=100;a+=20){const y=p.t+(H-p.t-p.b)*(1-a/100);c.beginPath();c.moveTo(p.l,y);c.lineTo(W-p.r,y);c.stroke();c.fillText(a+'%',18,y+4)}for(let x=0;x<=75;x+=15){const px=p.l+(W-p.l-p.r)*x/75;c.fillText(x,px-5,H-18)}const y99=p.t+(H-p.t-p.b)*.01;c.strokeStyle='#6ce5a5';c.setLineDash([4,4]);c.beginPath();c.moveTo(p.l,y99);c.lineTo(W-p.r,y99);c.stroke();c.setLineDash([]);if(!vals.length){empty(c,'Waiting for validation checkpoints…');return}c.strokeStyle='#ffbd69';c.fillStyle='#ffbd69';c.lineWidth=2;c.beginPath();vals.forEach((v,i)=>{const x=p.l+(W-p.l-p.r)*v.removed/75,y=p.t+(H-p.t-p.b)*(1-v.accuracy/100);i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();vals.forEach(v=>{const x=p.l+(W-p.l-p.r)*v.removed/75,y=p.t+(H-p.t-p.b)*(1-v.accuracy/100);c.beginPath();c.arc(x,y,4,0,Math.PI*2);c.fill();c.fillStyle='#ecf2ff';c.fillText(v.accuracy.toFixed(1)+'%',Math.min(x+7,W-52),Math.max(14,y-7));c.fillStyle='#ffbd69'});c.fillStyle='#8fa1c3';c.fillText('CoT tokens removed (75 = answer-only)',W/2-88,H-6)}
function render(){if(!payload)return;const exp=payload.stages.find(s=>s.name.includes('explicit'))||{points:[],validations:[]},intr=payload.stages.find(s=>s.name.includes('internalization'))||{points:[],validations:[]};const ep=exp.points||[],ip=intr.points||[],vals=intr.validations||[];lossChart('explicitChart',ep,'#54d6ff');lossChart('internalChart',ip,'#6ce5a5');accuracyChart(vals);const e=ep.at(-1),i=ip.at(-1);$('explicitMetric').innerHTML=e?`step <b>${e.step.toLocaleString()}</b> · loss <b>${e.loss.toFixed(4)}</b> · ${e.sps?.toFixed(2)??'—'} steps/s`:'Waiting for samples…';$('internalMetric').innerHTML=i?`step <b>${i.step.toLocaleString()}</b> · removed <b>${i.removed}/75</b> · loss <b>${i.loss.toFixed(4)}</b>`:'Waiting for rollout…';$('accuracyRows').innerHTML=vals.length?vals.map(v=>`<tr><td>${v.epoch}</td><td>${v.removed}/75</td><td>${v.accuracy.toFixed(1)}%</td></tr>`).join(''):'<tr><td colspan="3">No validation checkpoint yet</td></tr>'}
async function poll(){try{payload=await fetch('/data',{cache:'no-store'}).then(r=>r.json());$('status').textContent=`${payload.status} · refreshed ${new Date(payload.updated_at*1000).toLocaleTimeString()}`;render()}catch(e){$('status').textContent='Waiting for dashboard service…'}setTimeout(poll,10000)}
addEventListener('resize',render);poll();
</script>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh", type=int, default=10)
    parser.add_argument("--lines", type=int, default=50000)
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
