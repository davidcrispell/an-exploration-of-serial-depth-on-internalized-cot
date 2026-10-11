#!/usr/bin/env python3
"""Serve separate live monitors for Deep 5x5 CoT training."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


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

ARCHIVED_TASK_LOGS = {
    "3x3": {
        "cot_tokens": 25,
        "deep_explicit": "artifacts/checkpoints/deep1-3x3-explicit-cot/train.log",
        "deep_internal": "artifacts/checkpoints/deep1-3x3-internalized-cot/train.log",
        "shallow_explicit": "artifacts/checkpoints/shallow1-3x3-explicit-cot/train.log",
        "shallow_internal": "artifacts/checkpoints/shallow1-3x3-internalized-cot/train.log",
    },
    "4x4": {
        "cot_tokens": 47,
        "deep_explicit": "artifacts/checkpoints/deep1-4x4-explicit-cot/train.log",
        "deep_internal": "artifacts/checkpoints/deep1-4x4-internalized-cot/train.log",
        "shallow_explicit": "results/multiplication/4x4-explicit-cot-fp32.log",
        "shallow_internal": "results/multiplication/4x4-internalized-cot-fp32.log",
        "shallow_explicit_validation": "results/multiplication/shallow-4x4-explicit-cot-validation.json",
    },
}


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

    @staticmethod
    def parse_single_stage(text: str, name: str) -> dict:
        stage = {
            "name": name,
            "started_at": None,
            "finished_at": None,
            "points": [],
            "validations": [],
            "test": None,
        }
        for line in text.splitlines():
            loss_match = LOSS_RE.search(line)
            if loss_match:
                stage["points"].append(
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
                stage["validations"].append(
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
                stage["test"] = {
                    "removed": (
                        int(test_match.group("removed"))
                        if test_match.group("removed")
                        else 0
                    ),
                    "correct": int(test_match.group("correct")),
                    "total": int(test_match.group("total")),
                    "accuracy": float(test_match.group("pct")),
                }
        return stage

    def load_shallow_stages(self) -> list[dict]:
        paths = [
            (self.args.shallow_explicit_log, "shallow1_5x5_explicit_cot"),
            (self.args.shallow_internal_log, "shallow1_5x5_internalization"),
        ]
        stages = []
        for raw_path, name in paths:
            path = Path(raw_path)
            if path.exists():
                stages.append(self.parse_single_stage(path.read_text(), name))
        return stages

    def load_archived_tasks(self) -> dict[str, dict]:
        tasks = {}
        for task, config in ARCHIVED_TASK_LOGS.items():
            stages = {}
            for key in (
                "deep_explicit",
                "deep_internal",
                "shallow_explicit",
                "shallow_internal",
            ):
                path = Path(config[key])
                stages[key] = (
                    self.parse_single_stage(path.read_text(), f"{key}_{task}")
                    if path.exists()
                    else {"name": f"{key}_{task}", "points": [], "validations": []}
                )
            validation_path = config.get("shallow_explicit_validation")
            if validation_path and Path(validation_path).exists():
                metric = json.loads(Path(validation_path).read_text())
                if "results" in metric:
                    metric = metric["results"][0]
                stages["shallow_explicit"]["validations"] = [
                    {
                        "epoch": 0,
                        "removed": 0,
                        "correct": metric["correct"],
                        "total": metric["examples"],
                        "accuracy": metric["accuracy"] * 100,
                    }
                ]
            tasks[task] = {"cot_tokens": config["cot_tokens"], **stages}
        return tasks

    @staticmethod
    def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)

    def save_artifacts(self, payload: dict) -> None:
        archive_dir = Path(self.args.archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        data_path = archive_dir / "live-dashboard-data.json"
        temp_data = data_path.with_suffix(".json.tmp")
        temp_data.write_text(json.dumps(payload, indent=2) + "\n")
        temp_data.replace(data_path)

        for task, task_data in payload.get("task_curves", {}).items():
            for key in (
                "deep_explicit",
                "deep_internal",
                "shallow_explicit",
                "shallow_internal",
            ):
                stage = task_data[key]
                model, kind = key.split("_", 1)
                kind = "internalization" if kind == "internal" else kind
                self._write_csv(
                    archive_dir / f"{model}-{task}-{kind}-loss.csv",
                    stage.get("points", []),
                    ["epoch", "step", "removed", "loss", "sps"],
                )
                self._write_csv(
                    archive_dir / f"{model}-{task}-{kind}-validation.csv",
                    stage.get("validations", []),
                    ["epoch", "removed", "correct", "total", "accuracy"],
                )

        embedded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
        snapshot = HTML_TEMPLATE.replace("__ARCHIVED_PAYLOAD__", embedded)
        snapshot_path = archive_dir / "dashboard-snapshot.html"
        temp_snapshot = snapshot_path.with_suffix(".html.tmp")
        temp_snapshot.write_text(snapshot)
        temp_snapshot.replace(snapshot_path)

    def refresh_forever(self) -> None:
        while True:
            try:
                payload = self.parse(self.fetch_log())
                payload["shallow_stages"] = self.load_shallow_stages()
                archived_tasks = self.load_archived_tasks()
                deep_explicit = next(
                    stage for stage in payload["stages"] if "explicit" in stage["name"]
                )
                deep_internal = next(
                    stage
                    for stage in payload["stages"]
                    if "internalization" in stage["name"]
                )
                shallow_explicit = next(
                    stage
                    for stage in payload["shallow_stages"]
                    if "explicit" in stage["name"]
                )
                shallow_internal = next(
                    stage
                    for stage in payload["shallow_stages"]
                    if "internalization" in stage["name"]
                )
                payload["task_curves"] = {
                    "5x5": {
                        "cot_tokens": 75,
                        "deep_explicit": deep_explicit,
                        "deep_internal": deep_internal,
                        "shallow_explicit": shallow_explicit,
                        "shallow_internal": shallow_internal,
                    },
                    "4x4": archived_tasks["4x4"],
                    "3x3": archived_tasks["3x3"],
                }
                payload["archive_dir"] = str(Path(self.args.archive_dir).resolve())
                self.save_artifacts(payload)
            except Exception as exc:
                with self.lock:
                    payload = dict(self.payload)
                payload["status"] = f"refresh error: {exc}"
                payload["updated_at"] = time.time()
            with self.lock:
                self.payload = payload
            time.sleep(self.args.refresh)


HTML_TEMPLATE = r'''<!doctype html>
<meta charset="utf-8">
<title>Deep vs Shallow 5x5 internalization</title>
<style>
  :root { color-scheme: dark; --bg:#0b1020; --panel:#121a2e; --ink:#ecf2ff; --muted:#8fa1c3; --deep:#54d6ff; --shallow:#ffbd69; --target:#6ce5a5; --grid:#263451; }
  * { box-sizing:border-box } body { margin:0; font:14px system-ui,sans-serif; background:var(--bg); color:var(--ink) }
  main { max-width:1240px; margin:auto; padding:24px } h1 { margin:0 0 5px; font-size:24px } h2 { margin:28px 0 10px; font-size:20px; font-weight:500 } h3 { margin:0 0 10px; font-size:16px; font-weight:500 }
  .sub { color:var(--muted); margin-bottom:16px } .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px }
  .panel { background:var(--panel); border:1px solid #22304c; border-radius:12px; padding:14px; min-width:0 }
  .wide { grid-column:1/-1 } canvas { width:100%; height:300px; display:block } .wide canvas { height:340px }
  .metric { color:var(--muted); margin-bottom:6px } .metric b { color:var(--ink); font-weight:500 }
  .legend { display:flex; gap:16px; flex-wrap:wrap; color:var(--muted); margin:4px 0 8px }
  .legend span::before { content:''; display:inline-block; width:18px; height:2px; margin:0 6px 3px 0; background:var(--deep) }
  .legend .shallow::before { background:var(--shallow) }
  table { width:100%; border-collapse:collapse; margin-top:8px } th,td { text-align:right; padding:7px; border-bottom:1px solid #22304c }
  th:first-child,td:first-child { text-align:left } th { color:var(--muted); font-weight:400 }
  @media(max-width:760px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}main{padding:14px}}
</style>
<main>
  <h1>Deep 1 vs Shallow 1 · multiplication CoT internalization</h1>
  <div class="sub" id="status">Connecting…</div>
  <div id="taskSections"></div>
</main>
<script>
const archivedPayload=__ARCHIVED_PAYLOAD__; let payload=archivedPayload; const $=id=>document.getElementById(id);
const COLORS={deep:'#54d6ff',shallow:'#ffbd69',target:'#6ce5a5',grid:'#263451',muted:'#8fa1c3',ink:'#ecf2ff'};
function setup(id){const el=$(id),d=devicePixelRatio||1,r=el.getBoundingClientRect();el.width=r.width*d;el.height=r.height*d;const c=el.getContext('2d');c.scale(d,d);return {c,W:r.width,H:r.height}}
function empty(c,msg){c.fillStyle=COLORS.muted;c.font='12px system-ui';c.fillText(msg,58,34)}
function findStage(stages,needle){return (stages||[]).find(s=>s.name.includes(needle))||{points:[],validations:[]}}
function lossChart(id,series){const {c,W,H}=setup(id),p={l:58,r:18,t:18,b:36},all=series.flatMap(s=>s.points||[]);c.clearRect(0,0,W,H);if(!all.length){empty(c,'Waiting for loss samples…');return}const logs=all.map(x=>Math.log10(Math.max(x.loss,1e-8))),hi=Math.max(...logs),lo=Math.min(...logs),span=Math.max(.2,hi-lo),xmax=Math.max(...all.map(x=>x.step),1);c.font='11px system-ui';c.strokeStyle=COLORS.grid;c.fillStyle=COLORS.muted;c.lineWidth=1;for(let i=0;i<=4;i++){const y=p.t+(H-p.t-p.b)*i/4;c.beginPath();c.moveTo(p.l,y);c.lineTo(W-p.r,y);c.stroke();c.fillText(Math.pow(10,hi-span*i/4).toPrecision(2),4,y+4)}series.forEach(s=>{const pts=s.points||[];if(!pts.length)return;c.strokeStyle=s.color;c.lineWidth=2;c.beginPath();pts.forEach((q,i)=>{const x=p.l+(W-p.l-p.r)*q.step/xmax,y=p.t+(H-p.t-p.b)*(hi-Math.log10(Math.max(q.loss,1e-8)))/span;i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke()});c.fillStyle=COLORS.muted;c.fillText('optimizer step →',W/2-38,H-10)}
function checkpointLabel(index,total){const firstFull=Math.ceil(total/8);if(index===0)return `0/${total}`;if(index<firstFull)return `${index*8}/${total}`;if(index===firstFull)return `${total}/${total}`;return `${total}+A${index-firstFull}`}
function validationSeries(exp,intr){const baseline=(exp.validations||[]).at(-1);const vals=[];if(baseline)vals.push({index:0,removed:0,epoch:-1,accuracy:baseline.accuracy});(intr.validations||[]).forEach(v=>vals.push({...v,index:v.epoch+1}));return vals}
function accuracyChart(id,series,total){const {c,W,H}=setup(id),p={l:58,r:44,t:18,b:48},all=series.flatMap(s=>s.points||[]),xmax=Math.max(Math.ceil(total/8),...all.map(v=>v.index));c.clearRect(0,0,W,H);c.font='11px system-ui';c.strokeStyle=COLORS.grid;c.fillStyle=COLORS.muted;for(let a=0;a<=100;a+=20){const y=p.t+(H-p.t-p.b)*(1-a/100);c.beginPath();c.moveTo(p.l,y);c.lineTo(W-p.r,y);c.stroke();c.fillText(a+'%',18,y+4)}for(let index=0;index<=xmax;index++){if(xmax>9&&index%2&&index!==Math.ceil(total/8)&&index!==xmax)continue;const x=p.l+(W-p.l-p.r)*index/xmax,label=checkpointLabel(index,total);c.fillText(label,x-c.measureText(label).width/2,H-20)}const y99=p.t+(H-p.t-p.b)*.01;c.strokeStyle=COLORS.target;c.setLineDash([4,4]);c.beginPath();c.moveTo(p.l,y99);c.lineTo(W-p.r,y99);c.stroke();c.setLineDash([]);if(!all.length){empty(c,'Waiting for validation checkpoints…');return}series.forEach(s=>{const vals=s.points||[];if(!vals.length)return;c.strokeStyle=s.color;c.fillStyle=s.color;c.lineWidth=2;c.beginPath();vals.forEach((v,i)=>{const x=p.l+(W-p.l-p.r)*v.index/xmax,y=p.t+(H-p.t-p.b)*(1-v.accuracy/100);i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();vals.forEach(v=>{const x=p.l+(W-p.l-p.r)*v.index/xmax,y=p.t+(H-p.t-p.b)*(1-v.accuracy/100);c.beginPath();c.arc(x,y,s.name==='Deep 1'?4:3,0,Math.PI*2);c.fill()});const last=vals.at(-1),x=p.l+(W-p.l-p.r)*last.index/xmax,y=p.t+(H-p.t-p.b)*(1-last.accuracy/100);c.fillStyle=COLORS.ink;c.fillText(`${s.name} ${last.accuracy.toFixed(1)}%`,Math.min(x+7,W-96),Math.max(14,y-7))});c.fillStyle=COLORS.muted;c.fillText('curriculum checkpoint →',W/2-58,H-6)}
function metricText(deep,shallow,internal=false){const d=(deep.points||[]).at(-1),s=(shallow.points||[]).at(-1);if(!d&&!s)return 'Waiting for samples…';const one=(label,p)=>p?`${label} <b>${p.loss.toFixed(4)}</b>${internal&&p.removed!=null?` at ${p.removed}/75`:''}`:`${label} —`;return `${one('Deep',d)} · ${one('Shallow',s)}`}
function tableRows(deepVals,shallowVals,total){const byDeep=new Map(deepVals.map(v=>[v.index,v])),byShallow=new Map(shallowVals.map(v=>[v.index,v])),indices=[...new Set([...byDeep.keys(),...byShallow.keys()])].sort((a,b)=>a-b);return indices.map(index=>{const d=byDeep.get(index),s=byShallow.get(index),removed=d?.removed??s?.removed??0;return `<tr><td>${checkpointLabel(index,total)}</td><td>${removed}/${total}</td><td>${d?d.accuracy.toFixed(1)+'%':'—'}</td><td>${s?s.accuracy.toFixed(1)+'%':'—'}</td></tr>`}).join('')}
function taskMarkup(key,total){return `<section class="task"><h2>${key} multiplication · ${total} CoT tokens</h2><div class="grid"><section class="panel"><h3>Explicit-CoT fine-tuning loss</h3><div class="legend"><span>Deep 1</span><span class="shallow">Shallow 1</span></div><div class="metric" id="${key}-explicitMetric">Waiting for samples…</div><canvas id="${key}-explicitChart" aria-label="${key} explicit CoT loss for Deep 1 and Shallow 1"></canvas></section><section class="panel"><h3>Token-internalization rollout loss</h3><div class="legend"><span>Deep 1</span><span class="shallow">Shallow 1</span></div><div class="metric" id="${key}-internalMetric">Waiting for rollout…</div><canvas id="${key}-internalChart" aria-label="${key} internalization loss for Deep 1 and Shallow 1"></canvas></section><section class="panel wide"><h3>Validation accuracy across the curriculum</h3><div class="legend"><span>Deep 1</span><span class="shallow">Shallow 1</span></div><div class="metric">${total}/${total} is the first answer-only checkpoint; A1+ are full-removal adaptation epochs.</div><canvas id="${key}-accuracyChart" aria-label="${key} validation accuracy for Deep 1 and Shallow 1"></canvas><table><thead><tr><th>Checkpoint</th><th>Removed</th><th>Deep 1</th><th>Shallow 1</th></tr></thead><tbody id="${key}-accuracyRows"><tr><td colspan="4">No validation checkpoint yet</td></tr></tbody></table></section></div></section>`}
function renderTask(key,data){const total=data.cot_tokens,de=data.deep_explicit,di=data.deep_internal,se=data.shallow_explicit,si=data.shallow_internal,deepVals=validationSeries(de,di),shallowVals=validationSeries(se,si);lossChart(`${key}-explicitChart`,[{name:'Deep 1',points:de.points,color:COLORS.deep},{name:'Shallow 1',points:se.points,color:COLORS.shallow}]);lossChart(`${key}-internalChart`,[{name:'Deep 1',points:di.points,color:COLORS.deep},{name:'Shallow 1',points:si.points,color:COLORS.shallow}]);accuracyChart(`${key}-accuracyChart`,[{name:'Deep 1',points:deepVals,color:COLORS.deep},{name:'Shallow 1',points:shallowVals,color:COLORS.shallow}],total);$(`${key}-explicitMetric`).innerHTML=metricText(de,se);$(`${key}-internalMetric`).innerHTML=metricText(di,si,true).replaceAll('/75',`/${total}`);$(`${key}-accuracyRows`).innerHTML=tableRows(deepVals,shallowVals,total)||'<tr><td colspan="4">No validation checkpoint yet</td></tr>'}
function render(){if(!payload?.task_curves)return;const entries=Object.entries(payload.task_curves);$('taskSections').innerHTML=entries.map(([key,data])=>taskMarkup(key,data.cot_tokens)).join('');entries.forEach(([key,data])=>renderTask(key,data))}
async function poll(){try{payload=await fetch('/data',{cache:'no-store'}).then(r=>r.json());$('status').textContent=`${payload.status} · refreshed ${new Date(payload.updated_at*1000).toLocaleTimeString()}`;render()}catch(e){$('status').textContent='Waiting for dashboard service…'}setTimeout(poll,10000)}
addEventListener('resize',render);if(archivedPayload){$('status').textContent=`Saved ${new Date(payload.updated_at*1000).toLocaleString()} · raw JSON and CSV files are beside this snapshot`;render()}else{poll()}
</script>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh", type=int, default=10)
    parser.add_argument("--lines", type=int, default=50000)
    parser.add_argument(
        "--shallow-explicit-log",
        default="artifacts/checkpoints/shallow-5x5-explicit-cot/train.log",
    )
    parser.add_argument(
        "--shallow-internal-log",
        default="artifacts/checkpoints/shallow-5x5-internalized-cot/train.log",
    )
    parser.add_argument(
        "--archive-dir",
        default="artifacts/dashboards/deep1-5x5",
    )
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
                body = HTML_TEMPLATE.replace("__ARCHIVED_PAYLOAD__", "null").encode()
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
