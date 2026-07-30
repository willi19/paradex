"""Web dashboard for :mod:`paradex.process` distributed runs.

The console dashboard in :func:`paradex.process.run_distributed` prints the same
numbers, but a rig-wide undistort/upload run is long and multi-PC — you want it
on a second monitor, from your laptop, without an SSH session held open. This
serves exactly the aggregated status the console prints, over HTTP.

    main PC:  run_distributed("python src/util/upload_video/worker.py", web_port=8080)
              # -> http://<main-pc>:8080

Stdlib only (``http.server`` + ``json``): no Flask/SocketIO, and no CDN assets —
the capture rig is frequently offline, and the old Flask monitor's socket.io CDN
tag rendered a blank page there. The page polls ``/api/progress`` once a second.

Serving is read-only and side-effect free: the collector is owned by the caller,
:class:`ProgressWebServer` only reads its latest snapshot.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from paradex.process.processor import DONE_STATUSES, _live_elapsed, fmt_dur

SENTINEL_PREFIX = "_pc::"


# --------------------------------------------------------------------------- #
# snapshot -> JSON
# --------------------------------------------------------------------------- #
def build_snapshot(data: dict, pc_list: List[str]) -> dict:
    """Shape a collector snapshot into the JSON the page renders.

    Mirrors ``distributed._print_dashboard``: per-PC sentinels, per-job rows, and
    a rig ETA that is the slowest still-working PC (PCs run in parallel).
    """
    sentinels = {k[len(SENTINEL_PREFIX):]: v
                 for k, v in data.items() if k.startswith(SENTINEL_PREFIX)}
    jobs = {k: v for k, v in data.items() if not k.startswith(SENTINEL_PREFIX)}

    counts = {"completed": 0, "skipped": 0, "failed": 0,
              "processing": 0, "pending": 0}
    rows = []
    for jid, v in jobs.items():
        st = v.get("status", "pending")
        counts[st] = counts.get(st, 0) + 1
        rows.append({
            "name": jid,
            "pc": v.get("pc"),
            "status": st,
            "progress": v.get("progress", 0.0),
            "message": v.get("message", ""),
            "frame": v.get("frame"),
            "total": v.get("total"),
            "fps": v.get("fps"),
            "elapsed": _live_elapsed(v),
            "eta": v.get("eta"),
        })
    # Running first, then failed, then the rest — the page shows the top slice.
    order = {"processing": 0, "failed": 1, "pending": 2,
             "completed": 3, "skipped": 4}
    rows.sort(key=lambda r: (order.get(r["status"], 5), r["name"]))

    pcs = []
    for pc in pc_list:
        s = sentinels.get(pc)
        if s is None:
            pcs.append({"name": pc, "reported": False})
            continue
        pcs.append({
            "name": pc,
            "reported": True,
            "finished": bool(s.get("finished")),
            "counts": s.get("counts", {}),
            "total": s.get("total", 0),
            "num_workers": s.get("num_workers"),
            "eta": s.get("eta"),
        })

    pc_etas = [p["eta"] for p in pcs
               if p.get("reported") and not p.get("finished") and p.get("eta") is not None]
    all_reported = all(p.get("reported") for p in pcs) and len(pcs) > 0
    done = all_reported and all(p.get("finished") for p in pcs)

    return {
        "pcs": pcs,
        "jobs": rows,
        "summary": {
            "total": len(jobs),
            "counts": counts,
            "rig_eta": max(pc_etas) if pc_etas else None,
            "done": done,
        },
    }


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #
class ProgressWebServer:
    """Background HTTP server exposing a collector's live status.

    Args:
        collector: a started :class:`DataCollector` (or anything with ``get_data()``).
        pc_list:   PCs expected to report, in display order.
        port:      HTTP port to bind on all interfaces.
        title:     page heading, e.g. the worker command being run.
    """

    def __init__(self, collector, pc_list: List[str], port: int = 8080,
                 title: str = "paradex.process"):
        self.collector = collector
        self.pc_list = list(pc_list)
        self.port = port
        self.title = title
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (http.server API)
                if self.path.startswith("/api/progress"):
                    payload = build_snapshot(outer.collector.get_data(), outer.pc_list)
                    body = json.dumps(payload).encode()
                    self._respond(200, "application/json", body)
                elif self.path in ("/", "/index.html"):
                    self._respond(200, "text/html; charset=utf-8",
                                  _PAGE.replace("__TITLE__", outer.title).encode())
                else:
                    self._respond(404, "text/plain", b"not found")

            def _respond(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # keep the console dashboard readable

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print(f"[paradex.process] web dashboard: http://localhost:{self.port}")

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


def serve_progress(collector, pc_list: List[str], port: int = 8080,
                   title: str = "paradex.process") -> ProgressWebServer:
    """Start a :class:`ProgressWebServer` and return it (call ``.stop()`` when done)."""
    server = ProgressWebServer(collector, pc_list, port=port, title=title)
    server.start()
    return server


# --------------------------------------------------------------------------- #
# page (self-contained: no CDN, works on an offline rig)
# --------------------------------------------------------------------------- #
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>paradex.process monitor</title>
<style>
  :root{
    --bg:#f6f7f9; --card:#fff; --fg:#1b1d21; --muted:#6b7280; --line:#e3e6ea;
    --ok:#12a150; --run:#2f6fed; --fail:#e5484d; --skip:#9aa0a6; --bar:#eceef1;
  }
  @media (prefers-color-scheme:dark){
    :root{--bg:#111316;--card:#191c20;--fg:#e8eaed;--muted:#9aa0a6;--line:#2a2e34;--bar:#23272c}
  }
  *{box-sizing:border-box}
  body{margin:0;padding:24px;background:var(--bg);color:var(--fg);
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:1200px;margin:0 auto}
  h1{font-size:20px;margin:0 0 2px}
  .sub{color:var(--muted);font-size:13px;margin-bottom:20px}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
  .card .n{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
  .card .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:22px 0 10px}
  table{width:100%;border-collapse:collapse;background:var(--card);
        border:1px solid var(--line);border-radius:10px;overflow:hidden}
  th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:middle}
  th{font-size:12px;color:var(--muted);font-weight:500}
  tr:last-child td{border-bottom:none}
  td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
  .name{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
        max-width:420px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;
        font-weight:600;color:#fff}
  .s-completed{background:var(--ok)} .s-processing{background:var(--run)}
  .s-failed{background:var(--fail)} .s-skipped{background:var(--skip)}
  .s-pending{background:var(--bar);color:var(--muted)}
  .bar{height:6px;background:var(--bar);border-radius:3px;overflow:hidden;min-width:90px}
  .bar>i{display:block;height:100%;background:var(--run);border-radius:3px;transition:width .3s}
  .bar.done>i{background:var(--ok)}
  .msg{color:var(--muted);font-size:12px;max-width:280px;overflow:hidden;
       text-overflow:ellipsis;white-space:nowrap}
  .foot{color:var(--muted);font-size:12px;margin-top:14px}
  .off{color:var(--fail)}
</style></head><body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub" id="sub">connecting…</div>
  <div class="cards" id="cards"></div>
  <h2>Capture PCs</h2>
  <table id="pcs"><tbody></tbody></table>
  <h2>Jobs <span id="jobcount" style="color:var(--muted);font-weight:400"></span></h2>
  <table id="jobs">
    <thead><tr><th>job</th><th>pc</th><th>status</th><th>progress</th>
      <th class="num">frames</th><th class="num">elapsed</th><th class="num">eta</th><th>message</th></tr></thead>
    <tbody></tbody></table>
  <div class="foot" id="foot"></div>
</div>
<script>
const MAX_ROWS = 200;
function dur(s){
  if(s===null||s===undefined) return "-";
  s=Math.max(0,Math.round(s));
  if(s<60) return s+"s";
  if(s<3600) return Math.floor(s/60)+"m"+String(s%60).padStart(2,"0")+"s";
  return Math.floor(s/3600)+"h"+String(Math.floor(s%3600/60)).padStart(2,"0")+"m";
}
function esc(t){const d=document.createElement("div");d.textContent=t==null?"":t;return d.innerHTML}
function card(n,l){return `<div class="card"><div class="n">${n}</div><div class="l">${l}</div></div>`}

function render(d){
  const c=d.summary.counts, s=d.summary;
  document.getElementById("cards").innerHTML =
    card(s.total,"jobs")+card(c.completed||0,"done")+card(c.processing||0,"running")+
    card(c.skipped||0,"skipped")+card(c.failed||0,"failed")+
    card(s.done?"—":dur(s.rig_eta),"rig eta");
  document.getElementById("sub").textContent = s.done
    ? "all PCs finished"
    : `${d.pcs.filter(p=>p.reported&&!p.finished).length} PC(s) running`;

  document.getElementById("pcs").querySelector("tbody").innerHTML = d.pcs.map(p=>{
    if(!p.reported) return `<tr><td>${esc(p.name)}</td><td colspan="3" class="msg">no report yet</td></tr>`;
    const k=p.counts||{}, done=(k.completed||0)+(k.skipped||0)+(k.failed||0);
    const pct = p.total ? 100*done/p.total : 0;
    return `<tr><td>${esc(p.name)}</td>
      <td><span class="pill ${p.finished?"s-completed":"s-processing"}">${p.finished?"done":"running"}</span></td>
      <td style="width:40%"><div class="bar ${p.finished?"done":""}"><i style="width:${pct}%"></i></div></td>
      <td class="num">${done}/${p.total}${p.finished?"":" • ETA "+dur(p.eta)}</td></tr>`;
  }).join("");

  const rows=d.jobs.slice(0,MAX_ROWS);
  document.getElementById("jobcount").textContent =
    d.jobs.length>rows.length ? `(showing ${rows.length} of ${d.jobs.length})` : "";
  document.getElementById("jobs").querySelector("tbody").innerHTML = rows.map(j=>{
    const fr = (j.frame!=null&&j.total) ? `${j.frame}/${j.total}`+(j.fps?` @${j.fps.toFixed(0)}fps`:"") : "-";
    const done = j.status==="completed"||j.status==="skipped";
    return `<tr><td class="name" title="${esc(j.name)}">${esc(j.name)}</td>
      <td>${esc(j.pc||"")}</td>
      <td><span class="pill s-${j.status}">${j.status}</span></td>
      <td style="width:160px"><div class="bar ${done?"done":""}"><i style="width:${done?100:(j.progress||0)}%"></i></div></td>
      <td class="num">${fr}</td><td class="num">${dur(j.elapsed)}</td>
      <td class="num">${done?"-":dur(j.eta)}</td>
      <td class="msg" title="${esc(j.message)}">${esc(j.message)}</td></tr>`;
  }).join("");
}

async function tick(){
  const foot=document.getElementById("foot");
  try{
    const r=await fetch("/api/progress",{cache:"no-store"});
    render(await r.json());
    foot.className="foot"; foot.textContent="updated "+new Date().toLocaleTimeString();
  }catch(e){
    foot.className="foot off"; foot.textContent="monitor unreachable — run finished or main PC stopped";
  }
}
tick(); setInterval(tick,1000);
</script></body></html>
"""
