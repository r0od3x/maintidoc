"""MaintiDoc — demonstration front-end.

Runs the pipeline step by step and shows what happens in a browser page
instead of a terminal. Standard library only: no dependency is added to the
project, and nothing here belongs to phase 8 (API / interface), which is
still to be built.

    .venv\\Scripts\\python.exe demo\\app.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PORT = 8750
QUESTION = "Que signifie le voyant orange fixe sur le SAN-LAB 2000 ?"

ENV = {
    **os.environ,
    "HF_HOME": r"D:\hf_cache",
    "UV_CACHE_DIR": r"D:\uv_cache",
    "OLLAMA_MODELS": r"D:\ollama\models",
    # Le budget de 60 s du cahier des charges n'est PAS releve ici : un test
    # verifie qu'il vaut 60, et qwen2.5vl:3b repond en une cinquantaine de
    # secondes. Seul le 7b, trop gros pour cette carte, exigeait davantage.
    "MAINTIDOC_VLM_MODEL": "qwen2.5vl:3b",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}

STEPS = [
    {
        "id": "tests",
        "title": "Tests automatises",
        "subtitle": "192 tests, sans reseau ni service externe",
        "cmd": ["-m", "pytest", "-q", "--no-header"],
        "note": "Contrats, ingestion, recuperation, MCP et agent sont couverts hors ligne.",
    },
    {
        "id": "index",
        "title": "L'index est reel",
        "subtitle": "41 documents, deux familles d'equipement",
        "cmd": ["-m", "maintidoc.retrieval", "stats"],
        "note": "316 points de texte (BGE-M3) et 46 points visuels (CLIP), construits en 302 s sur CPU.",
    },
    {
        "id": "norerank",
        "title": "SANS reranker",
        "subtitle": "fusion RRF seule — la ligne de base",
        "cmd": ["-m", "maintidoc.retrieval", "ask", QUESTION, "--no-rerank"],
        "note": "Dix modeles d'equipement emploient la meme formule pour le voyant orange fixe.",
        "expect": "SAN-R240",
        "verdict_bad": True,
    },
    {
        "id": "rerank",
        "title": "AVEC reranker",
        "subtitle": "meme question, seul le cross-encoder change",
        "cmd": ["-m", "maintidoc.retrieval", "ask", QUESTION],
        "note": "Le cross-encoder lit question et passage ensemble : c'est lui qui tranche.",
        "expect": "SAN-2000",
        "verdict_good": True,
    },
    {
        "id": "ollama",
        "title": "Modele vision-langage",
        "subtitle": "Ollama + qwen2.5vl:3b",
        "cmd": ["-m", "maintidoc.generation", "check"],
        "note": "29 couches sur 37 tiennent sur la RTX 3050 : environ 26 tokens/s.",
    },
    {
        "id": "answer",
        "title": "Generation sourcee",
        "subtitle": "reponse contrainte par le schema + verification des citations",
        "cmd": ["-m", "maintidoc.generation", "answer", QUESTION],
        "note": "Chaque citation est verifiee deux fois : structurellement, puis semantiquement.",
    },
]

STATE = {
    "question": QUESTION,
    "started": False,
    "finished": False,
    "steps": [
        {
            "id": s["id"],
            "title": s["title"],
            "subtitle": s["subtitle"],
            "note": s["note"],
            "status": "pending",
            "output": "",
            "seconds": 0.0,
            "top": None,
            "verdict": None,
        }
        for s in STEPS
    ],
}
LOCK = threading.Lock()

TOP_RE = re.compile(r"^\s*1\.\s+(\S+)", re.M)


def run_steps() -> None:
    with LOCK:
        STATE["started"] = True
    for spec, slot in zip(STEPS, STATE["steps"]):
        with LOCK:
            slot["status"] = "running"
        t0 = time.time()
        try:
            proc = subprocess.run(
                [str(PY), *spec["cmd"]],
                cwd=str(ROOT),
                env=ENV,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            ok = proc.returncode == 0
        except Exception as exc:  # noqa: BLE001 - surfaced in the page
            out, ok = f"{type(exc).__name__}: {exc}", False

        noise = ("FutureWarning", "get_sentence_embedding", "Loading weights",
                 "HF_TOKEN", "warnings.warn", "  warn(", "self._texts", "self._images")
        lines = [ln.rstrip() for ln in out.splitlines()
                 if ln.strip() and not any(n in ln for n in noise)]
        text = "\n".join(lines[-60:])

        top = None
        m = TOP_RE.search(text)
        if m:
            top = m.group(1)

        verdict = None
        if spec.get("expect"):
            hit = bool(top and spec["expect"] in top)
            if spec.get("verdict_bad"):
                verdict = "bad" if hit else "neutral"
            elif spec.get("verdict_good"):
                verdict = "good" if hit else "neutral"

        with LOCK:
            slot["status"] = "done" if ok else "failed"
            slot["output"] = text
            slot["seconds"] = round(time.time() - t0, 1)
            slot["top"] = top
            slot["verdict"] = verdict
    with LOCK:
        STATE["finished"] = True


PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>MaintiDoc - Demonstration</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font:15px/1.55 "Segoe UI",system-ui,sans-serif;background:#0d1117;color:#e6edf3;padding:28px 20px 60px}
.wrap{max-width:1080px;margin:0 auto}
header{border-bottom:2px solid #1f6feb;padding-bottom:16px;margin-bottom:8px}
h1{font-size:27px;letter-spacing:-.02em}
h1 span{color:#1f6feb}
.sub{color:#8b949e;margin-top:5px;font-size:14px}
.q{background:#161b22;border-left:3px solid #1f6feb;padding:12px 16px;border-radius:5px;margin:20px 0 26px;font-size:15px}
.q b{color:#8b949e;font-size:11px;letter-spacing:.14em;text-transform:uppercase;display:block;margin-bottom:5px}
.card{background:#161b22;border:1px solid #30363d;border-radius:9px;margin-bottom:16px;overflow:hidden}
.card.running{border-color:#1f6feb;box-shadow:0 0 0 1px #1f6feb33}
.card.good{border-color:#238636}
.card.bad{border-color:#9e6a03}
.card.failed{border-color:#da3633}
.hd{display:flex;align-items:center;gap:13px;padding:14px 18px;background:#1c2128}
.n{width:27px;height:27px;border-radius:50%;background:#30363d;color:#e6edf3;display:grid;place-items:center;font-size:13px;font-weight:700;flex:none}
.card.done .n,.card.good .n{background:#238636}.card.bad .n{background:#9e6a03}
.card.running .n{background:#1f6feb}.card.failed .n{background:#da3633}
.t{font-weight:650;font-size:16px}.s{color:#8b949e;font-size:13px;margin-top:1px}
.meta{margin-left:auto;text-align:right;flex:none}
.badge{display:inline-block;padding:3px 10px;border-radius:11px;font-size:11.5px;font-weight:700;letter-spacing:.05em;text-transform:uppercase}
.b-pending{background:#30363d;color:#8b949e}.b-running{background:#1f6feb;color:#fff}
.b-done{background:#238636;color:#fff}.b-failed{background:#da3633;color:#fff}
.secs{color:#8b949e;font-size:12px;margin-top:4px}
.body{padding:0 18px 16px}
.note{color:#8b949e;font-size:13px;padding:11px 0 0;font-style:italic}
pre{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:13px;margin-top:12px;
    font:12.5px/1.5 "Cascadia Mono",Consolas,monospace;white-space:pre-wrap;color:#c9d1d9;max-height:420px;overflow:auto}
.top{margin-top:12px;padding:11px 14px;border-radius:6px;font-size:14px}
.top.good{background:#23863622;border:1px solid #238636}
.top.bad{background:#9e6a0322;border:1px solid #9e6a03}
.top b{font-family:"Cascadia Mono",Consolas,monospace}
.lbl{font-size:11px;letter-spacing:.12em;text-transform:uppercase;display:block;margin-bottom:4px;opacity:.85}
.spin{display:inline-block;width:13px;height:13px;border:2px solid #1f6feb44;border-top-color:#1f6feb;
      border-radius:50%;animation:r .8s linear infinite;vertical-align:-2px;margin-right:5px}
@keyframes r{to{transform:rotate(360deg)}}
.done-banner{background:#238636;color:#fff;padding:15px 20px;border-radius:9px;margin-top:22px;font-weight:600}
.key{background:#161b22;border:1px solid #30363d;border-left:3px solid #1f6feb;border-radius:7px;padding:16px 20px;margin-top:22px}
.key h3{font-size:14px;color:#1f6feb;letter-spacing:.1em;text-transform:uppercase;margin-bottom:9px}
.key p{color:#c9d1d9;font-size:14px;margin-bottom:7px}
</style></head><body><div class="wrap">
<header><h1>Mainti<span>Doc</span></h1>
<div class="sub">Assistant documentaire multimodal et agentique pour la maintenance SAN &middot; phases 1 a 7</div></header>
<div class="q"><b>Question posee</b>__QUESTION__</div>
<div id="steps"></div>
<div id="end"></div>
<div class="key"><h3>Ce que la demonstration montre</h3>
<p>Dix modeles d'equipement emploient la meme formule pour le <i>voyant orange fixe</i>, avec dix significations
differentes. Repondre correctement impose donc d'identifier le modele d'abord.</p>
<p>La fusion seule place <b>SAN-RACK 240</b> en tete. Seul le cross-encoder, qui lit la question et le passage
ensemble, remet <b>SAN-LAB 2000</b> en premier. C'est pourquoi le reranker n'est pas optionnel &mdash; et c'est la
ligne de base sur laquelle la phase 10 mesurerait son fine-tuning.</p></div>
</div>
<script>
const BN={pending:"b-pending",running:"b-running",done:"b-done",failed:"b-failed"};
const BT={pending:"en attente",running:"en cours",done:"termine",failed:"echec"};
async function tick(){
 const r=await fetch('/status'); const d=await r.json();
 document.getElementById('steps').innerHTML=d.steps.map((s,i)=>{
  const cls=s.verdict==='good'?'good':s.verdict==='bad'?'bad':s.status;
  let top='';
  if(s.top&&s.verdict==='bad') top=`<div class="top bad"><span class="lbl">Premier resultat &mdash; mauvais equipement</span><b>${s.top}</b></div>`;
  if(s.top&&s.verdict==='good') top=`<div class="top good"><span class="lbl">Premier resultat &mdash; equipement correct</span><b>${s.top}</b></div>`;
  return `<div class="card ${cls}">
   <div class="hd"><div class="n">${i+1}</div>
    <div><div class="t">${s.title}</div><div class="s">${s.subtitle}</div></div>
    <div class="meta"><span class="badge ${BN[s.status]}">${s.status==='running'?'<span class=spin></span>':''}${BT[s.status]}</span>
    ${s.seconds?`<div class="secs">${s.seconds} s</div>`:''}</div></div>
   <div class="body"><div class="note">${s.note}</div>${top}
   ${s.output?`<pre>${s.output.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}</pre>`:''}</div></div>`;
 }).join('');
 document.getElementById('end').innerHTML = d.finished
   ? '<div class="done-banner">Demonstration terminee &mdash; les six etapes ont ete executees.</div>' : '';
 if(!d.finished) setTimeout(tick,700);
}
tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence the console
        pass

    def do_GET(self):
        if self.path.startswith("/status"):
            with LOCK:
                payload = json.dumps(STATE).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        body = PAGE.replace("__QUESTION__", QUESTION).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    if not PY.exists():
        print(f"Interpreteur introuvable : {PY}")
        return 1
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}/"
    print(f"MaintiDoc - demonstration ouverte sur {url}")
    print("Fermer cette fenetre pour arreter.")
    webbrowser.open(url)
    worker = threading.Thread(target=run_steps, daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.5)
        print("Demonstration terminee. La page reste consultable.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
