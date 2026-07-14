"""Main-PC web capture UI with distributed capture-PC previews.

Run with::

    python -m paradex.dataset_acqusition.capture_site --host 0.0.0.0 --port 8000

Capture agents remain independent processes.  This app only controls the
existing :class:`CaptureSession` and proxies their low-bandwidth JPEG APIs so
the browser connects to the main PC alone.
"""

from __future__ import annotations

import argparse
import os
import re
import threading
from contextlib import asynccontextmanager
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
import uvicorn

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.file_io import find_latest_index
from paradex.utils.path import shared_dir
from paradex.utils.system import get_pc_ip, pc_info


DATASET_RE = re.compile(r"^[A-Za-z0-9._-]+$")
CAPTURE_ROOT = os.path.join("capture", "site_test")
PREVIEW_PORT = 5484


def camera_routes() -> Dict[str, dict]:
    routes = {}
    for pc_name, config in pc_info.items():
        for serial in config.get("cam_list", []):
            routes[str(serial)] = {
                "serial": str(serial),
                "pc": pc_name,
                "ip": get_pc_ip(pc_name),
            }
    return routes


class SiteCaptureController:
    def __init__(self) -> None:
        self.routes = camera_routes()
        self.session = CaptureSession(camera=True)
        self._lock = threading.RLock()
        self.recording = False
        self.dataset: Optional[str] = None
        self.episode: Optional[int] = None
        self.save_path: Optional[str] = None
        self.last_error: Optional[str] = None

    @staticmethod
    def _validate_dataset(name: str) -> str:
        name = str(name).strip()
        if not name or not DATASET_RE.fullmatch(name) or name in (".", ".."):
            raise ValueError(
                "dataset must contain only letters, numbers, '.', '_' or '-'"
            )
        return name

    def start(self, dataset: str) -> dict:
        dataset = self._validate_dataset(dataset)
        with self._lock:
            if self.recording:
                raise RuntimeError("a capture session is already recording")
            dataset_dir = os.path.join(shared_dir, CAPTURE_ROOT, dataset)
            episode = int(find_latest_index(dataset_dir)) + 1
            save_path = os.path.join(CAPTURE_ROOT, dataset, str(episode))
            self.last_error = None
            try:
                self.session.start(save_path)
            except Exception as exc:
                self.last_error = str(exc)
                raise
            self.recording = True
            self.dataset = dataset
            self.episode = episode
            self.save_path = save_path
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            if not self.recording:
                raise RuntimeError("no capture session is recording")
            try:
                self.session.stop()
            except Exception as exc:
                self.last_error = str(exc)
                raise
            finally:
                self.recording = False
            return self.status()

    def status(self) -> dict:
        return {
            "recording": self.recording,
            "dataset": self.dataset,
            "episode": self.episode,
            "save_path": self.save_path,
            "last_error": self.last_error,
            "cameras": list(self.routes.values()),
        }

    def fetch_preview(self, serial: str) -> bytes:
        route = self.routes.get(str(serial))
        if route is None:
            raise KeyError(serial)
        url = "http://{}:{}/preview/{}".format(
            route["ip"], PREVIEW_PORT, route["serial"]
        )
        request = Request(url, headers={"Cache-Control": "no-cache"})
        with urlopen(request, timeout=1.0) as response:
            return response.read()

    def close(self) -> None:
        with self._lock:
            try:
                if self.recording:
                    self.session.stop()
                    self.recording = False
            finally:
                self.session.end()


INDEX_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paradex Capture</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#121212;color:#eee}*{box-sizing:border-box}
body{margin:0}header{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:12px;padding:12px 18px;background:#181818;border-bottom:1px solid #333}
h1{font-size:17px;margin:0 12px 0 0}.grow{flex:1}input,button{font:inherit;border-radius:5px;border:1px solid #444;padding:8px 11px}
input{background:#222;color:#fff;width:220px}button{background:#1976d2;color:#fff;cursor:pointer}button.stop{background:#b3261e}button:disabled{opacity:.45;cursor:not-allowed}
#state{font:12px ui-monospace,monospace;color:#aaa}.error{color:#ff7373!important}main{padding:14px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}
.tile{background:#181818;border:1px solid #303030;border-radius:6px;overflow:hidden}.tile img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#080808}
.meta{display:flex;justify-content:space-between;padding:7px 9px;font:12px ui-monospace,monospace;color:#aaa}.offline{opacity:.35}
</style></head><body>
<header><h1>Paradex Capture</h1><input id="dataset" value="test" placeholder="dataset name"><button id="start">Record</button><button id="stop" class="stop" disabled>Stop</button><div class="grow"></div><div id="state">connecting...</div></header>
<main><div id="grid" class="grid"></div></main>
<script>
let cameras=[], recording=false;
const stateEl=document.getElementById('state'), startBtn=document.getElementById('start'), stopBtn=document.getElementById('stop');
function renderCameras(next){if(JSON.stringify(cameras)===JSON.stringify(next))return;cameras=next;const g=document.getElementById('grid');g.innerHTML='';
 for(const c of cameras){const t=document.createElement('div');t.className='tile';t.innerHTML=`<img id="cam-${c.serial}" alt="${c.serial}"><div class="meta"><span>${c.serial}</span><span>${c.pc}</span></div>`;g.appendChild(t);poll(c.serial);}}
function poll(serial){const img=document.getElementById('cam-'+serial);if(!img)return;const next=()=>setTimeout(()=>poll(serial),200);
 img.onload=()=>{img.classList.remove('offline');next()};img.onerror=()=>{img.classList.add('offline');setTimeout(()=>poll(serial),700)};
 img.src='/api/preview/'+encodeURIComponent(serial)+'?t='+Date.now();}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'}),s=await r.json();recording=s.recording;renderCameras(s.cameras);
 stateEl.textContent=recording?`RECORDING  ${s.save_path}`:(s.last_error?`ERROR  ${s.last_error}`:'IDLE');stateEl.className=s.last_error?'error':'';startBtn.disabled=recording;stopBtn.disabled=!recording;
 }catch(e){stateEl.textContent='main server disconnected';stateEl.className='error'}setTimeout(refresh,1000)}
async function command(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});const x=await r.json();if(!r.ok)throw new Error(x.detail||'request failed');return x;}
startBtn.onclick=async()=>{try{startBtn.disabled=true;await command('/api/start',{dataset:document.getElementById('dataset').value});await refresh()}catch(e){alert(e.message);startBtn.disabled=false}};
stopBtn.onclick=async()=>{try{stopBtn.disabled=true;await command('/api/stop');await refresh()}catch(e){alert(e.message);stopBtn.disabled=false}};
refresh();
</script></body></html>"""


def create_app(controller: SiteCaptureController) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            controller.close()

    app = FastAPI(title="Paradex Capture Site", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/status")
    def status():
        return controller.status()

    @app.post("/api/start")
    def start(payload: dict):
        try:
            return controller.start(payload.get("dataset", ""))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/stop")
    def stop():
        try:
            return controller.stop()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/preview/{serial}")
    def preview(serial: str):
        try:
            jpeg = controller.fetch_preview(serial)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown camera") from exc
        except (HTTPError, URLError, TimeoutError) as exc:
            raise HTTPException(status_code=503, detail="preview unavailable") from exc
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Paradex main-PC capture website")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    controller = SiteCaptureController()
    uvicorn.run(create_app(controller), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
