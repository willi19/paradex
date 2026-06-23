"""Thin ZMQ REQ wrapper around a running viser_server."""
from __future__ import annotations

import pickle
from typing import Any, Dict, Optional

import numpy as np
import trimesh
import zmq


class ViserClient:
    def __init__(self, addr: str = "tcp://localhost:5572", timeout_ms: int = 60000):
        self.addr = addr
        self.timeout_ms = int(timeout_ms)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(addr)

    def _call(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        self._sock.send(pickle.dumps(msg))
        return pickle.loads(self._sock.recv())

    def ping(self) -> Dict[str, Any]:
        return self._call({"cmd": "ping"})

    def set_object(self, name: str, mesh: trimesh.Trimesh, pose: np.ndarray,
                   opacity: float = 1.0) -> Dict[str, Any]:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        vc = getattr(mesh.visual, "vertex_colors", None)
        vc_arr = None
        if vc is not None:
            vc_arr = np.asarray(vc, dtype=np.uint8)
            if vc_arr.ndim != 2 or vc_arr.shape[0] != vertices.shape[0]:
                vc_arr = None
        return self._call({
            "cmd": "set_object",
            "name": str(name),
            "vertices": vertices,
            "faces": faces,
            "vertex_colors": vc_arr,
            "pose": np.asarray(pose, dtype=np.float32),
            "opacity": float(opacity),
        })

    def set_qpos(self, qpos: np.ndarray, name: str = "robot") -> Dict[str, Any]:
        return self._call({
            "cmd": "set_qpos",
            "name": str(name),
            "qpos": np.asarray(qpos, dtype=np.float32),
        })

    def set_traj(self, traj: np.ndarray, name: str = "traj",
                 robot_name: str = "robot") -> Dict[str, Any]:
        return self._call({
            "cmd": "set_traj",
            "name": str(name),
            "robot_name": str(robot_name),
            "traj": np.asarray(traj, dtype=np.float32),
        })

    def clear_objects(self) -> Dict[str, Any]:
        return self._call({"cmd": "clear_objects"})

    def close(self):
        try:
            self._sock.close(0)
        except Exception:
            pass
