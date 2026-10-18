#!/usr/bin/env python3
"""Live GSM8K paired-pilot dashboard with durable data and figure exports."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


START_RE = re.compile(r"^\[(?P<when>[^]]+)] START (?P<name>\S+)")
DONE_RE = re.compile(r"^\[(?P<when>[^]]+)] DONE (?P<name>\S+)")
VAL_RE = re.compile(
    r"validation epoch=(?P<epoch>\d+)(?: removal=(?P<removed>all|\d+))?: "
    r"(?P<correct>\d+)/(?P<total>\d+) \((?P<pct>[0-9.]+)%\)"
)
STAGE_NAMES = (
    "deep1-explicit",
    "shallow1-explicit",
    "deep1-internalized",
    "shallow1-internalized",
)


def empty_stage(name: str) -> dict:
    return {
        "name": name,
        "started_at": None,
        "finished_at": None,
        "loss": [],
        "validation": [],
    }


def parse_log(text: str) -> dict:
    stages = {name: empty_stage(name) for name in STAGE_NAMES}
    current: dict | None = None
    for raw in text.splitlines():
        match = START_RE.match(raw)
        if match:
            current = stages.setdefault(match.group("name"), empty_stage(match.group("name")))
            current["started_at"] = match.group("when")
            continue
        match = DONE_RE.match(raw)
        if match:
            stage = stages.setdefault(match.group("name"), empty_stage(match.group("name")))
            stage["finished_at"] = match.group("when")
            current = None
            continue
        if current is None:
            continue
        if raw.startswith("{"):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                record = None
            if isinstance(record, dict) and record.get("event") == "train":
                current["loss"].append(
                    {
                        "epoch": int(record["epoch"]),
                        "global_step": int(record["global_step"]),
                        "optimizer_step": int(record["optimizer_step"]),
                        "scheduled_removal": record.get("scheduled_removal"),
                        "loss": float(record["loss"]),
                        "elapsed_seconds": float(record["elapsed_seconds"]),
                    }
                )
                continue
        match = VAL_RE.search(raw)
        if match:
            removed = match.group("removed")
            current["validation"].append(
                {
                    "epoch": int(match.group("epoch")),
                    "scheduled_removal": removed if removed == "all" else int(removed or 0),
                    "correct": int(match.group("correct")),
                    "total": int(match.group("total")),
                    "accuracy": float(match.group("pct")),
                }
            )
    return {
        "status": "ok",
        "updated_at": time.time(),
        "test_split_touched": False,
        "stages": stages,
    }


class Dashboard:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.payload = parse_log("")

    def endpoint(self) -> tuple[str, str]:
        result = subprocess.run(
            ["runpodctl", "pod", "get", self.args.pod, "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        pod = json.loads(result.stdout)
        return str(pod["ssh"]["ip"]), str(pod["ssh"]["port"])

    def remote_log(self) -> str:
        host, port = self.endpoint()
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
                f"test -f {self.args.log} && tail -n {self.args.lines} {self.args.log} || true",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout

    @staticmethod
    def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)

    def export_figures(self, payload: dict, output: Path) -> None:
        colors = {"deep1": "#1772b4", "shallow1": "#e2851b"}
        for kind, suffix, title in (
            ("explicit", "explicit-loss", "GSM8K explicit-CoT fine-tuning loss"),
            ("internalized", "internalization-loss", "GSM8K token-internalization loss"),
        ):
            fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
            plotted = False
            for model in ("deep1", "shallow1"):
                stage = payload["stages"][f"{model}-{kind}"]
                rows = stage["loss"]
                if rows:
                    ax.plot(
                        [row["optimizer_step"] for row in rows],
                        [row["loss"] for row in rows],
                        label="Deep 1" if model == "deep1" else "Shallow 1",
                        color=colors[model],
                        linewidth=1.5,
                    )
                    plotted = True
            ax.set(title=title, xlabel="Optimizer step", ylabel="Training loss")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            if plotted:
                ax.legend(frameon=False)
            else:
                ax.text(0.5, 0.5, "Waiting for training samples", ha="center", va="center")
            for extension in ("png", "svg", "pdf"):
                fig.savefig(output / f"gsm8k-{suffix}.{extension}", dpi=220)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
        plotted = False
        for model in ("deep1", "shallow1"):
            explicit = payload["stages"][f"{model}-explicit"]["validation"]
            internal = payload["stages"][f"{model}-internalized"]["validation"]
            points = []
            if explicit:
                points.append({"x": -1, "accuracy": explicit[-1]["accuracy"]})
            points.extend({"x": row["epoch"], "accuracy": row["accuracy"]} for row in internal)
            if points:
                ax.plot(
                    [row["x"] for row in points],
                    [row["accuracy"] for row in points],
                    marker="o",
                    label="Deep 1" if model == "deep1" else "Shallow 1",
                    color=colors[model],
                )
                plotted = True
        ax.axhline(99, color="#2a9d68", linestyle="--", linewidth=1, label="99% target")
        ax.set(
            title="GSM8K validation exact-answer accuracy",
            xlabel="Curriculum epoch (−1 = explicit CoT)",
            ylabel="Accuracy (%)",
            ylim=(0, 101),
        )
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "Waiting for validation checkpoints", ha="center", va="center")
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output / f"gsm8k-validation-accuracy.{extension}", dpi=220)
        plt.close(fig)

    def save(self, payload: dict) -> None:
        output = Path(self.args.archive_dir)
        output.mkdir(parents=True, exist_ok=True)
        temp = output / "live-dashboard-data.json.tmp"
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        temp.replace(output / "live-dashboard-data.json")
        for name, stage in payload["stages"].items():
            self.write_csv(
                output / f"{name}-loss.csv",
                stage["loss"],
                [
                    "epoch",
                    "global_step",
                    "optimizer_step",
                    "scheduled_removal",
                    "loss",
                    "elapsed_seconds",
                ],
            )
            self.write_csv(
                output / f"{name}-validation.csv",
                stage["validation"],
                ["epoch", "scheduled_removal", "correct", "total", "accuracy"],
            )
        self.export_figures(payload, output)
        embedded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
        (output / "dashboard-snapshot.html").write_text(
            HTML.replace("__PAYLOAD__", embedded)
        )

    def refresh(self) -> None:
        while True:
            try:
                payload = parse_log(self.remote_log())
                self.save(payload)
            except Exception as exc:
                with self.lock:
                    payload = dict(self.payload)
                payload["status"] = f"refresh error: {exc}"
                payload["updated_at"] = time.time()
            with self.lock:
                self.payload = payload
            time.sleep(self.args.refresh)


HTML = r'''<!doctype html><meta charset="utf-8"><title>GSM8K paired pilot</title>
<style>:root{color-scheme:dark;--bg:#0b1020;--panel:#121a2e;--ink:#edf3ff;--muted:#91a2c3;--deep:#54d6ff;--shallow:#ffbd69;--grid:#263451}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:24px}h1{margin:0 0 6px}.sub{color:var(--muted);margin-bottom:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{background:var(--panel);border:1px solid #22304c;border-radius:12px;padding:14px}.wide{grid-column:1/-1}canvas{width:100%;height:320px;display:block}.legend{display:flex;gap:18px;color:var(--muted);margin:5px 0}.legend i{display:inline-block;width:18px;height:3px;margin-right:6px;background:var(--deep)}.legend .s i{background:var(--shallow)}table{width:100%;border-collapse:collapse;margin-top:10px}th,td{text-align:right;padding:7px;border-bottom:1px solid #22304c}th:first-child,td:first-child{text-align:left}@media(max-width:760px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}}</style>
<main><h1>Deep 1 vs Shallow 1 · GSM8K internalization pilot</h1><div class="sub" id="status">Connecting…</div><div class="grid"><section class="panel"><h2>Explicit-CoT loss</h2><div class="legend"><span><i></i>Deep 1</span><span class="s"><i></i>Shallow 1</span></div><canvas id="explicit"></canvas></section><section class="panel"><h2>Internalization loss</h2><div class="legend"><span><i></i>Deep 1</span><span class="s"><i></i>Shallow 1</span></div><canvas id="internal"></canvas></section><section class="panel wide"><h2>Validation exact-answer accuracy</h2><div class="legend"><span><i></i>Deep 1</span><span class="s"><i></i>Shallow 1</span></div><canvas id="accuracy"></canvas><table><thead><tr><th>Model / stage</th><th>Epoch</th><th>Removed</th><th>Correct</th><th>Accuracy</th></tr></thead><tbody id="rows"></tbody></table></section></div></main>
<script>let P=__PAYLOAD__;const C={d:'#54d6ff',s:'#ffbd69',g:'#263451',m:'#91a2c3',i:'#edf3ff'};const $=x=>document.getElementById(x);function ctx(id){const e=$(id),r=e.getBoundingClientRect(),z=devicePixelRatio||1;e.width=r.width*z;e.height=r.height*z;const c=e.getContext('2d');c.scale(z,z);return{c,W:r.width,H:r.height}}function chart(id,series,ylog=false,accuracy=false){const{c,W,H}=ctx(id),p={l:58,r:18,t:18,b:40},all=series.flatMap(x=>x.p);c.clearRect(0,0,W,H);c.font='11px system-ui';c.strokeStyle=C.g;c.fillStyle=C.m;if(!all.length){c.fillText('Waiting for samples…',p.l,p.t+18);return}const xmax=Math.max(...all.map(x=>x.x),1),vals=all.map(x=>ylog?Math.log10(Math.max(x.y,1e-8)):x.y),hi=accuracy?100:Math.max(...vals),lo=accuracy?0:Math.min(...vals),span=Math.max(accuracy?100:.2,hi-lo);for(let i=0;i<=4;i++){const y=p.t+(H-p.t-p.b)*i/4;c.beginPath();c.moveTo(p.l,y);c.lineTo(W-p.r,y);c.stroke();const v=hi-span*i/4;c.fillText(accuracy?v.toFixed(0)+'%':(ylog?Math.pow(10,v).toPrecision(2):v.toPrecision(3)),4,y+4)}series.forEach(s=>{if(!s.p.length)return;c.strokeStyle=s.color;c.fillStyle=s.color;c.lineWidth=2;c.beginPath();s.p.forEach((q,i)=>{const x=p.l+(W-p.l-p.r)*q.x/xmax,y=p.t+(H-p.t-p.b)*(hi-(ylog?Math.log10(Math.max(q.y,1e-8)):q.y))/span;i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();if(accuracy)s.p.forEach(q=>{const x=p.l+(W-p.l-p.r)*q.x/xmax,y=p.t+(H-p.t-p.b)*(hi-q.y)/span;c.beginPath();c.arc(x,y,4,0,7);c.fill()})});c.fillStyle=C.m;c.fillText(accuracy?'curriculum epoch →':'optimizer step →',W/2-42,H-9)}function loss(name){return['deep1','shallow1'].map((m,i)=>({color:i?C.s:C.d,p:P.stages[m+'-'+name].loss.map(x=>({x:x.optimizer_step,y:x.loss}))}))}function vals(m){const e=P.stages[m+'-explicit'].validation,n=P.stages[m+'-internalized'].validation,a=[];if(e.length)a.push({x:0,y:e.at(-1).accuracy});n.forEach(x=>a.push({x:x.epoch+1,y:x.accuracy}));return a}function render(){chart('explicit',loss('explicit'),true);chart('internal',loss('internalized'),true);chart('accuracy',[{color:C.d,p:vals('deep1')},{color:C.s,p:vals('shallow1')}],false,true);const rows=[];for(const [m,label] of [['deep1','Deep 1'],['shallow1','Shallow 1']])for(const k of ['explicit','internalized'])for(const v of P.stages[m+'-'+k].validation)rows.push(`<tr><td>${label} · ${k}</td><td>${v.epoch}</td><td>${v.scheduled_removal}</td><td>${v.correct}/${v.total}</td><td>${v.accuracy.toFixed(2)}%</td></tr>`);$('rows').innerHTML=rows.join('')||'<tr><td colspan="5">Waiting for validation…</td></tr>'}async function poll(){try{P=await fetch('/data',{cache:'no-store'}).then(r=>r.json());$('status').textContent=`${P.status} · refreshed ${new Date(P.updated_at*1000).toLocaleTimeString()} · test split untouched · figures and CSV archived locally`;render()}catch(e){$('status').textContent='Waiting for dashboard service…'}setTimeout(poll,10000)}addEventListener('resize',render);if(P){render()}poll()</script>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh", type=int, default=15)
    parser.add_argument("--lines", type=int, default=200000)
    parser.add_argument(
        "--archive-dir", default="artifacts/dashboards/gsm8k-paired-pilot"
    )
    args = parser.parse_args()
    dashboard = Dashboard(args)
    threading.Thread(target=dashboard.refresh, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/data":
                with dashboard.lock:
                    body = json.dumps(dashboard.payload).encode()
                content_type = "application/json"
            else:
                body = HTML.replace("__PAYLOAD__", "null").encode()
                content_type = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
