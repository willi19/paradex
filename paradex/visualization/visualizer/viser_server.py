"""Long-running viser server.

Bootstraps a ViserViewer with a single robot URDF, then listens on a ZMQ REP
socket for incoming commands (object swap, trajectory replace, qpos set, etc).
Lets capture scripts attach to an already-running viewer instead of paying the
viser startup cost each run.

Usage:
    python -m paradex.visualization.visualizer.viser_server \
        --robot_urdf rsc/robot/xarm_allegro_v5.urdf \
        --addr tcp://*:5572
"""
from __future__ import annotations

import argparse
import pickle
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import trimesh
import zmq

from paradex.visualization.visualizer.viser import ViserViewer


def _make_mesh(vertices, faces, vertex_colors=None) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    if vertex_colors is not None:
        mesh.visual.vertex_colors = np.asarray(vertex_colors, dtype=np.uint8)
    return mesh


class ViserServer:
    """Dispatches ZMQ messages to a ViserViewer running in this process."""

    def __init__(self, vis: ViserViewer):
        self.vis = vis
        self._lock = threading.Lock()

    # ---- command handlers ------------------------------------------------
    def cmd_ping(self, _msg: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "msg": "pong"}

    def cmd_set_object(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        name = str(msg["name"])
        pose = np.asarray(msg["pose"], dtype=np.float32)
        mesh = _make_mesh(msg["vertices"], msg["faces"], msg.get("vertex_colors"))
        opacity = float(msg.get("opacity", 1.0))
        with self._lock:
            if name in self.vis.obj_dict:
                # Replace by removing then re-adding so geometry/pose both update.
                try:
                    self.vis.obj_dict[name]["frame"].remove()
                except Exception:
                    pass
                try:
                    self.vis.obj_dict[name]["handle"].remove()
                except Exception:
                    pass
                del self.vis.obj_dict[name]
                self.vis.frame_nodes.pop(name, None)
            self.vis.add_object(name, mesh, pose, opacity=opacity)
        return {"ok": True}

    def cmd_set_qpos(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        name = str(msg.get("name", "robot"))
        qpos = np.asarray(msg["qpos"], dtype=np.float32)
        with self._lock:
            if name not in self.vis.robot_dict:
                return {"ok": False, "msg": f"no robot named '{name}'"}
            self.vis.robot_dict[name].update_cfg(qpos)
        return {"ok": True}

    def cmd_set_traj(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Replace any existing trajectory with this one."""
        name = str(msg.get("name", "traj"))
        robot_name = str(msg.get("robot_name", "robot"))
        traj = np.asarray(msg["traj"], dtype=np.float32)
        with self._lock:
            # Reset trajectory state then add fresh.
            self.vis.traj_list = []
            self.vis.num_frames = 0
            try:
                self.vis.gui_timestep.max = 0
                self.vis.gui_timestep.value = 0
            except Exception:
                pass
            self.vis.add_traj(name, robot_traj={robot_name: traj})
        return {"ok": True, "n_frames": int(traj.shape[0])}

    def cmd_clear_objects(self, _msg: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            for name in list(self.vis.obj_dict.keys()):
                try:
                    self.vis.obj_dict[name]["frame"].remove()
                except Exception:
                    pass
                try:
                    self.vis.obj_dict[name]["handle"].remove()
                except Exception:
                    pass
                del self.vis.obj_dict[name]
                self.vis.frame_nodes.pop(name, None)
        return {"ok": True}

    # ---- ZMQ loop --------------------------------------------------------
    def serve(self, addr: str):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        sock.bind(addr)
        print(f"[viser_server] listening on {addr}")
        handlers = {
            "ping": self.cmd_ping,
            "set_object": self.cmd_set_object,
            "set_qpos": self.cmd_set_qpos,
            "set_traj": self.cmd_set_traj,
            "clear_objects": self.cmd_clear_objects,
        }
        while True:
            try:
                raw = sock.recv()
                msg = pickle.loads(raw)
                cmd = str(msg.get("cmd", ""))
                handler = handlers.get(cmd)
                if handler is None:
                    resp = {"ok": False, "msg": f"unknown cmd '{cmd}'"}
                else:
                    resp = handler(msg)
            except Exception as e:
                resp = {"ok": False, "msg": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()}
                print(f"[viser_server] error: {e}\n{traceback.format_exc()}")
            try:
                sock.send(pickle.dumps(resp))
            except Exception as e:
                print(f"[viser_server] send error: {e}")


def _update_loop(vis: ViserViewer, hz: float):
    period = 1.0 / max(1.0, float(hz))
    while True:
        t0 = time.perf_counter()
        try:
            vis.update()
        except Exception as e:
            print(f"[viser_server] update error: {e}")
        sleep = period - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_urdf", required=True)
    ap.add_argument("--robot_name", default="robot")
    ap.add_argument("--addr", default="tcp://*:5572")
    ap.add_argument("--update_hz", type=float, default=30.0)
    args = ap.parse_args()

    urdf = str(Path(args.robot_urdf).expanduser())
    if not Path(urdf).exists():
        raise FileNotFoundError(urdf)

    vis = ViserViewer()
    vis.add_floor(height=0.0)
    vis.add_robot(args.robot_name, urdf)
    print(f"[viser_server] loaded robot '{args.robot_name}' from {urdf}")

    threading.Thread(target=_update_loop, args=(vis, args.update_hz), daemon=True).start()
    ViserServer(vis).serve(args.addr)


if __name__ == "__main__":
    main()
