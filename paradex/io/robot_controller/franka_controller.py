import time
import numpy as np
from scipy.spatial.transform import Rotation
from threading import Thread, Event, Lock
import os

import zmq
import msgpack

action_dof = 7


def homo2rotmat_pos(h):
    """4x4 homogeneous → (position[3] in meters, rotation_matrix[3x3])."""
    return h[:3, 3].copy(), h[:3, :3].copy()


def rotmat_pos2homo(pos, R):
    """position[3] + rotation_matrix[3x3] → 4x4 homogeneous."""
    h = np.eye(4)
    h[:3, 3] = pos
    h[:3, :3] = R
    return h


class FrankaController:
    """Python controller for Franka FR3 via franka_daemon (ZMQ + msgpack).

    Assumes franka_daemon is already running on the given ip.
    """

    def __init__(self, ip, command_port=5555, state_port=5556):
        self.ip = ip
        self.command_port = command_port
        self.state_port = state_port

        self.lock = Lock()
        self.exit_event = Event()
        self.save_event = Event()
        self.error_event = Event()

        self._latest_state = None
        self.save_path = None
        self.data = {}

        self._connect()

        self._state_thread = Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()

        # Wait for first state
        deadline = time.time() + 5.0
        while self._latest_state is None and time.time() < deadline:
            time.sleep(0.01)
        if self._latest_state is None:
            self.error_event.set()

    # ------------------------------------------------------------------
    # ZMQ connection
    # ------------------------------------------------------------------

    def _connect(self):
        self._ctx = zmq.Context()

        self._cmd_sock = self._ctx.socket(zmq.REQ)
        self._cmd_sock.setsockopt(zmq.RCVTIMEO, 30000)  # 30s timeout
        self._cmd_sock.setsockopt(zmq.SNDTIMEO, 5000)
        self._cmd_sock.connect(f"tcp://{self.ip}:{self.command_port}")

        self._sub_sock = self._ctx.socket(zmq.SUB)
        self._sub_sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub_sock.setsockopt(zmq.RCVTIMEO, 1000)
        self._sub_sock.connect(f"tcp://{self.ip}:{self.state_port}")

    def _reconnect_cmd_sock(self):
        """Recreate REQ socket after timeout corrupts send/recv state."""
        try:
            self._cmd_sock.close()
        except Exception:
            pass
        self._cmd_sock = self._ctx.socket(zmq.REQ)
        self._cmd_sock.setsockopt(zmq.RCVTIMEO, 30000)
        self._cmd_sock.setsockopt(zmq.SNDTIMEO, 5000)
        self._cmd_sock.connect(f"tcp://{self.ip}:{self.command_port}")

    def _send_command(self, msg: dict) -> dict:
        packed = msgpack.packb(msg, use_bin_type=True)
        try:
            self._cmd_sock.send(packed)
            reply = self._cmd_sock.recv()
        except zmq.Again:
            # Timeout → REQ socket stuck in recv state, must recreate
            print(f"[FrankaController] ZMQ timeout for {msg.get('type')}, reconnecting socket")
            self._reconnect_cmd_sock()
            raise
        except zmq.ZMQError as e:
            # EFSM or other state error → socket corrupted, reconnect
            print(f"[FrankaController] ZMQ error for {msg.get('type')}: {e}, reconnecting socket")
            self._reconnect_cmd_sock()
            raise
        return msgpack.unpackb(reply, raw=False)

    # ------------------------------------------------------------------
    # State streaming thread
    # ------------------------------------------------------------------

    def _state_loop(self):
        while not self.exit_event.is_set():
            try:
                raw = self._sub_sock.recv()
                state = msgpack.unpackb(raw, raw=False)
                if state.get("type") != "state_update":
                    continue

                with self.lock:
                    self._latest_state = state

                    if self.save_event.is_set():
                        self.data["time"].append(state["timestamp"])
                        self.data["position"].append(np.array(state["qpos"]))
                        self.data["velocity"].append(np.array(state["qvel"]))
                        self.data["torque"].append(np.array(state["tau_ext"]))
                        self.data["wrench"].append(np.array(state["wrench"]))
                        self.data["O_T_EE"].append(np.array(state["O_T_EE"]))
                        self.data["gripper_width"].append(state["gripper_width"])

            except zmq.Again:
                continue
            except Exception as e:
                print(f"[FrankaController] state loop error: {e}")
                self.error_event.set()

    # ------------------------------------------------------------------
    # Public API (matches XArmController)
    # ------------------------------------------------------------------

    def move(self, action, is_servo=True, speed_scale=0.15, move_speed=0.05):
        """Move robot to target.

        Args:
            action: np.array(7,) for joint space or np.array(4,4) for cartesian.
            is_servo: if False, blocks until motion completes.
                      (franka_daemon always blocks, so is_servo is ignored.)
            speed_scale: joint speed scale (0.0-1.0) for joint moves.
            move_speed: cartesian speed (m/s) for cartesian moves.
        """
        assert action.shape == (7,) or action.shape == (4, 4)

        if action.shape == (7,):
            resp = self._send_command({
                "type": "move_to_qpos",
                "qpos": action.tolist(),
                "speed_scale": speed_scale,
            })
        else:
            pos, R = homo2rotmat_pos(action)
            resp = self._send_command({
                "type": "move_to_cartesian",
                "position": pos.tolist(),
                "orientation": R.flatten().tolist(),  # row-major 9 elements
                "move_speed": move_speed,
            })

        if resp.get("type") == "error":
            print(f"[FrankaController] move error: {resp.get('message')}")
            self.error_event.set()

        return resp

    def get_data(self):
        """Get current robot state.

        Returns:
            dict with keys: qpos(7,), qvel(7,), position(4,4), wrench(6,),
                           tau_ext(7,), gripper_width, gripper_grasping, time
        """
        with self.lock:
            if self._latest_state is None:
                return None
            s = self._latest_state

        qpos = np.array(s["qpos"])

        # O_T_EE is column-major 16 elements → 4x4 matrix
        O_T_EE = np.array(s["O_T_EE"]).reshape(4, 4, order="F")

        return {
            "qpos": qpos,
            "qvel": np.array(s["qvel"]),
            "position": O_T_EE,
            "tau_ext": np.array(s["tau_ext"]),
            "wrench": np.array(s["wrench"]),
            "gripper_width": s["gripper_width"],
            "gripper_grasping": s["gripper_grasping"],
            "time": s["timestamp"],
        }

    def start(self, save_path):
        """Start recording data to npy files."""
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.data = {
            "time": [],
            "position": [],
            "velocity": [],
            "torque": [],
            "wrench": [],
            "O_T_EE": [],
            "gripper_width": [],
        }

        with self.lock:
            self.save_event.set()

    def stop(self):
        """Stop recording and save npy files."""
        with self.lock:
            self.save_event.clear()

        if self.save_path is not None:
            for name, value in self.data.items():
                np.save(os.path.join(self.save_path, f"{name}.npy"), np.array(value))
                self.data[name] = []
            self.save_path = None

    def end(self):
        """Shutdown controller: stop recording, close sockets."""
        self.exit_event.set()
        self._state_thread.join(timeout=3.0)

        if self.save_event.is_set():
            self.stop()

        self._cmd_sock.close()
        self._sub_sock.close()
        self._ctx.term()

    def is_error(self):
        return self.error_event.is_set()

    def reset(self):
        """Reconnect to daemon."""
        self.error_event.clear()
        try:
            self._cmd_sock.close()
            self._sub_sock.close()
            self._ctx.term()
        except Exception:
            pass

        self._connect()

    # ------------------------------------------------------------------
    # Franka-specific methods
    # ------------------------------------------------------------------

    def open_gripper(self, width=0.08, speed=0.05):
        resp = self._send_command({
            "type": "open_gripper",
            "width": width,
            "speed": speed,
        })
        if resp.get("type") == "error":
            print(f"[FrankaController] open_gripper error: {resp.get('message')}")
        return resp

    def grasp(self, force=60.0, speed=0.05, eps_inner=0.005, eps_outer=0.005):
        resp = self._send_command({
            "type": "grasp",
            "force": force,
            "speed": speed,
            "eps_inner": eps_inner,
            "eps_outer": eps_outer,
        })
        if resp.get("type") == "error":
            print(f"[FrankaController] grasp error: {resp.get('message')}")
        return resp

    def set_cartesian_velocity(self, twist, duration_ms=100):
        """Send velocity command.

        Args:
            twist: array-like [vx, vy, vz, wx, wy, wz] in base frame.
            duration_ms: how long to apply (ms).
        """
        twist = np.asarray(twist, dtype=float)
        assert twist.shape == (6,)
        resp = self._send_command({
            "type": "set_cartesian_velocity",
            "twist": twist.tolist(),
            "duration_ms": duration_ms,
        })
        if resp.get("type") == "error":
            print(f"[FrankaController] velocity error: {resp.get('message')}")
        return resp

    def set_joint_velocity(self, dq, duration_ms=100):
        """Send joint velocity command.

        Args:
            dq: array-like [dq1..dq7] joint velocities (rad/s).
            duration_ms: how long to apply (ms).
        """
        dq = np.asarray(dq, dtype=float)
        assert dq.shape == (7,)
        resp = self._send_command({
            "type": "set_joint_velocity",
            "dq": dq.tolist(),
            "duration_ms": duration_ms,
        })
        if resp.get("type") == "error":
            print(f"[FrankaController] joint_velocity error: {resp.get('message')}")
        return resp

    def set_torques(self, torques, duration_ms=100):
        """Send direct torque command.

        Args:
            torques: array-like [tau1..tau7] joint torques (Nm).
            duration_ms: how long to apply (ms).
        """
        torques = np.asarray(torques, dtype=float)
        assert torques.shape == (7,)
        resp = self._send_command({
            "type": "set_torques",
            "torques": torques.tolist(),
            "duration_ms": duration_ms,
        })
        if resp.get("type") == "error":
            print(f"[FrankaController] torque error: {resp.get('message')}")
        return resp

    def set_joint_impedance(self, K_theta):
        """Set joint impedance (stiffness).

        Args:
            K_theta: array-like [K1..K7] joint stiffness (Nm/rad).
        """
        K_theta = np.asarray(K_theta, dtype=float)
        assert K_theta.shape == (7,)
        resp = self._send_command({
            "type": "set_joint_impedance",
            "K_theta": K_theta.tolist(),
        })
        if resp.get("type") == "error":
            print(f"[FrankaController] set_joint_impedance error: {resp.get('message')}")
        return resp

    def set_cartesian_impedance(self, K_x):
        """Set Cartesian impedance (stiffness).

        Args:
            K_x: array-like [Kx, Ky, Kz, Kroll, Kpitch, Kyaw].
        """
        K_x = np.asarray(K_x, dtype=float)
        assert K_x.shape == (6,)
        resp = self._send_command({
            "type": "set_cartesian_impedance",
            "K_x": K_x.tolist(),
        })
        if resp.get("type") == "error":
            print(f"[FrankaController] set_cartesian_impedance error: {resp.get('message')}")
        return resp

    def set_collision_behavior(self, torque_lower, torque_upper, force_lower, force_upper):
        """Set collision detection thresholds.

        Args:
            torque_lower: array(7,) lower torque thresholds (Nm).
            torque_upper: array(7,) upper torque thresholds (Nm).
            force_lower: array(6,) lower force thresholds (N/Nm).
            force_upper: array(6,) upper force thresholds (N/Nm).
        """
        resp = self._send_command({
            "type": "set_collision_behavior",
            "torque_lower": np.asarray(torque_lower, dtype=float).tolist(),
            "torque_upper": np.asarray(torque_upper, dtype=float).tolist(),
            "force_lower": np.asarray(force_lower, dtype=float).tolist(),
            "force_upper": np.asarray(force_upper, dtype=float).tolist(),
        })
        if resp.get("type") == "error":
            print(f"[FrankaController] set_collision_behavior error: {resp.get('message')}")
        return resp

    def set_load(self, mass, F_x_Cload, load_inertia):
        """Register end-effector payload.

        Args:
            mass: payload mass (kg).
            F_x_Cload: array(3,) center of mass in flange frame (m).
            load_inertia: array(9,) or (3,3) inertia matrix (row-major, kg*m^2).
        """
        load_inertia = np.asarray(load_inertia, dtype=float).flatten()
        assert load_inertia.shape == (9,)
        resp = self._send_command({
            "type": "set_load",
            "mass": float(mass),
            "F_x_Cload": np.asarray(F_x_Cload, dtype=float).tolist(),
            "load_inertia": load_inertia.tolist(),
        })
        if resp.get("type") == "error":
            print(f"[FrankaController] set_load error: {resp.get('message')}")
        return resp

    def set_ee(self, NE_T_EE):
        """Set end-effector transformation (flange → EE).

        Args:
            NE_T_EE: 4x4 homogeneous matrix (column-major when flattened to 16).
                     Pass np.eye(4) for no end-effector offset.
        """
        NE_T_EE = np.asarray(NE_T_EE, dtype=float)
        assert NE_T_EE.shape == (4, 4)
        # libfranka expects column-major 16 elements
        resp = self._send_command({
            "type": "set_ee",
            "NE_T_EE": NE_T_EE.flatten(order="F").tolist(),
        })
        if resp.get("type") == "error":
            print(f"[FrankaController] set_ee error: {resp.get('message')}")
        return resp

    def set_guiding_mode(self, guiding_axes, nullspace=False):
        """Enable hand-guiding mode.

        Args:
            guiding_axes: list of 6 bools [x, y, z, roll, pitch, yaw].
                          True = axis is free for guiding.
            nullspace: allow nullspace motion.
        """
        assert len(guiding_axes) == 6
        resp = self._send_command({
            "type": "set_guiding_mode",
            "guiding_axes": [bool(a) for a in guiding_axes],
            "nullspace": bool(nullspace),
        })
        if resp.get("type") == "error":
            print(f"[FrankaController] set_guiding_mode error: {resp.get('message')}")
        return resp

    def error_recovery(self):
        """Attempt automatic error recovery."""
        resp = self._send_command({"type": "error_recovery"})
        if resp.get("type") == "success":
            self.error_event.clear()
        return resp

    def stop_streaming(self):
        """Stop any active velocity/torque streaming control loop."""
        try:
            resp = self._send_command({"type": "stop_streaming"})
            if resp.get("type") == "error":
                print(f"[FrankaController] stop_streaming error: {resp.get('message')}")
            return resp
        except Exception as e:
            print(f"[FrankaController] stop_streaming exception: {e}, trying emergency_stop")
            try:
                return self._send_command({"type": "stop"})
            except Exception:
                return {"type": "error", "message": str(e)}

    def emergency_stop(self):
        resp = self._send_command({"type": "stop"})
        return resp

    def ping(self):
        resp = self._send_command({"type": "ping"})
        return resp.get("type") == "success"
