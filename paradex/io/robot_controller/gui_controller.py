import os
import tkinter as tk
import time
import numpy as np
from threading import Thread
from scipy.spatial.transform import Rotation
from collections import deque
from enum import Enum

_CART_BUTTONS = frozenset([
    'X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-',
    'Roll+', 'Roll-', 'Pitch+', 'Pitch-', 'Yaw+', 'Yaw-',
])


class WaypointType(Enum):
    JOINT = 'joint'
    CARTESIAN = 'cartesian'

class Waypoint:
    def __init__(self, name, wp_type, target=None, hand_qpos=None, threshold=None,
                 repeat_ticks=0):
        """
        wp_type: WaypointType.JOINT or CARTESIAN (target이 None이면 무시됨)
        target: None이면 arm 안 움직임
        hand_qpos: None이면 hand 안 움직임
        """
        self.type = wp_type
        self.target = target
        self.hand_qpos = hand_qpos

        self.name = name
        if threshold is None:
            self.threshold = 0.01 if wp_type == WaypointType.JOINT else (0.001, 0.01)
        else:
            self.threshold = threshold

        self.trajectory = None
        self.hand_trajectory = None
        self.robot_step = 0
        self.hand_step = 0

        self.repeat_ticks = int(repeat_ticks)
        self.repeat_count = 0
        self.reached = False

class RobotGUIController:
    def __init__(self, robot_controller, hand_controller=None,
                 jog_only=False, save_path=None,
                 urdf_path=None, eef_link=None):
        """
        Args:
            jog_only: keep the window open with an empty waypoint queue. The normal
                mode exits as soon as the queue drains, which would close the window
                immediately when it is used purely as a jog panel.
            save_path: enables a "Save Pose" button writing `<idx>_qpos.npy` (DOF,)
                and `<idx>_pose.npy` (4,4), the same layout `*_teaching.py` produces
                and `src/calibration/handeye/capture.py` replays. Resumes after any
                existing indices.
            urdf_path, eef_link: enable Jacobian-based cartesian jogging. The daemon's
                cartesian velocity stream bounds the *twist's* acceleration but not the
                resulting joint accelerations, so near a limit or singularity a mild
                cartesian rate demands a joint jerk and the motion aborts with
                "cartesian_motion_generator_joint_acceleration_discontinuity". With a
                URDF we instead map the twist to joint velocities ourselves (damped
                least squares) and use the well-behaved joint velocity stream.
        """
        print(">>> Initializing Robot GUI Controller...")

        self.robot = robot_controller
        self.hand = hand_controller
        self.jog_only = jog_only
        self.save_path = save_path

        # Auto-detect DOF from robot controller
        data = self.robot.get_data()
        self.arm_dof = data['qpos'].shape[0] if data is not None else 7

        # Auto-detect hand DOF
        if self.hand is not None:
            hand_data = self.hand.get_data()
            self.hand_dof = hand_data['qpos'].shape[0] if hand_data is not None else 16
        else:
            self.hand_dof = 0

        # Check if robot has gripper methods (Franka)
        self.has_gripper = hasattr(self.robot, 'open_gripper') and hasattr(self.robot, 'grasp')
        # Check if robot has guiding mode (Franka)
        self.has_guiding = hasattr(self.robot, 'set_guiding_mode')

        # Franka's move() is blocking (is_servo is ignored), so per-tick position
        # commands stutter. Jog it with the streaming velocity interface instead.
        self.jog_by_velocity = (hasattr(self.robot, 'set_joint_velocity')
                                and hasattr(self.robot, 'stop_streaming'))
        self._velocity_streaming = False
        # In streaming mode the daemon keeps applying the last setpoint, so the
        # command only needs to go out when it changes. Re-sending every 10 ms tick
        # makes jogging stutter — each send is a blocking ZMQ round trip.
        self._last_jog_cmd = None

        # Control parameters
        self.joint_vel_limit = 0.06  # rad per tick
        self.cart_vel_limit = 1.0    # mm per tick
        self.rot_vel_limit = 0.01    # rad per tick
        self.hand_vel_limit = 0.05  # per tick
        self.tick_s = 0.01           # _control_loop period

        # Jog speeds for the streaming-velocity path. Deliberately NOT derived from
        # the per-tick limits above (those are xArm servo increments; /tick_s would
        # give 6 rad/s, past the FR3 joint limit). Hand-jog pace, tune if sluggish.
        self.jog_joint_vel = 0.25    # rad/s
        self.jog_cart_vel = 0.02     # m/s (2 cm/s)
        self.jog_rot_vel = 0.15      # rad/s (~9 deg/s)
        # Roll/Pitch/Yaw about the tool's own axes rather than the robot base's.
        self.rotate_in_tool_frame = True

        # Every send is a blocking ZMQ round trip, so what makes motion smooth is
        # sending FEW commands and letting the daemon's 1 kHz loop interpolate (it
        # rate-limits each setpoint change at 10 rad/s^2 per joint / 5 m/s^2 linear).
        # A held jog button is therefore one command, and a slider target is a constant
        # velocity plus a stop — not a per-tick recomputed setpoint.
        self.jog_send_deadband = 0.02   # skip a send if the setpoint barely changed
        self._jog_kind = None           # 'j' | 'c' — which stream is open
        self._jog_vec = None            # last commanded setpoint

        self._joint_limits = self._resolve_joint_limits()

        # Jacobian-based cartesian jogging (see docstring).
        self._robot_wrapper = None
        self._eef_link_id = None
        self.dls_damping = 0.05         # damps joint speed blow-up near singularities
        self.jog_joint_vel_max = 0.8    # rad/s ceiling for jacobian-resolved jogging
        if urdf_path is not None and eef_link is not None:
            try:
                from paradex.robot.robot_wrapper import RobotWrapper
                self._robot_wrapper = RobotWrapper(urdf_path)
                self._eef_link_id = self._robot_wrapper.get_link_index(eef_link)
                print(f">>> Cartesian jog via Jacobian on '{eef_link}'")
            except Exception as e:
                print(f">>> Jacobian cartesian jog unavailable ({e}); "
                      "falling back to the daemon's cartesian velocity stream")
                self._robot_wrapper = None

        # Slider / arrow-key joint targeting. The control loop drives the arm toward
        # `_joint_target` with the same ramped velocity stream the jog buttons use,
        # so a slider drag is smooth instead of a burst of blocking position moves.
        self._joint_sliders = []
        self._joint_entries = []
        self._joint_target = None       # None = not tracking, follow the robot
        self._selected_joint = 0
        self.slider_step_deg = 1.0      # arrow-key increment
        self.slider_approach_band = 0.25  # rad; start easing off inside this
        self.slider_vel = 0.6           # rad/s cap while tracking a target
        self.slider_tol = 0.004         # rad; stop tracking inside this

        # Waypoint system
        self.waypoint_queue = deque()
        self.current_waypoint = None
        self.manual_override = False
        self.auto_execute = False
        # Button states
        self.button_states = {}
        self.running = True

        self._ui_dirty = True

        # GUI
        self.root = tk.Tk()
        self._build_gui()

        self.root.after(100, self._ui_pump)

    def _build_gui(self):
        self.root.title("Robot Waypoint Controller")
        self.root.geometry("700x1000")

        tk.Label(self.root, text="Robot Waypoint Controller",
                 font=("Arial", 16, "bold")).pack(pady=10)

        # Status
        self._build_status_frame()

        # Manual control
        self._build_manual_control_frame()

        # Exit
        self._build_slider_panel()

        if self.save_path is not None:
            self._build_save_panel()

        tk.Button(self.root, text="Exit", width=20, bg="red", fg="white",
                  command=self._on_exit).pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)

    def _build_status_frame(self):
        frame = tk.LabelFrame(self.root, text="Waypoint Queue Status", font=("Arial", 12))
        frame.pack(pady=10, padx=10, fill="both")

        self.queue_label = tk.Label(frame, text="Queue: 0 waypoints",
                                    font=("Arial", 10))
        self.queue_label.pack(pady=5)

        self.current_label = tk.Label(frame, text="Current: None",
                                    font=("Arial", 10, "bold"), fg="blue")
        self.current_label.pack(pady=5)

        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=5)

        start_btn = tk.Button(btn_frame, text="Start", width=15, bg="green", fg="white")
        start_btn.bind('<ButtonPress-1>', lambda e: self._on_start_press())
        start_btn.bind('<ButtonRelease-1>', lambda e: self._on_start_release())
        start_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="Clear Queue", width=15, bg="orange",
                command=self.clear_queue).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="Stop & Clear", width=15, bg="red",
                command=self.stop_and_clear).pack(side=tk.LEFT, padx=5)

    def _build_manual_control_frame(self):
        frame = tk.LabelFrame(self.root, text="Manual Control (Hold to Override)",
                              font=("Arial", 12))
        frame.pack(pady=10, padx=10, fill="both")

        # No joint jog buttons — the Joint Sliders panel covers joint-space motion,
        # and having both fighting over the same velocity stream caused conflicts.

        # Cartesian control
        cart_frame = tk.Frame(frame)
        cart_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(cart_frame, text="Cartesian Control", font=("Arial", 10, "bold")).pack()

        controls = [
            ("Translation", [("X+", 1), ("X-", -1), ("Y+", 1), ("Y-", -1), ("Z+", 1), ("Z-", -1)]),
            ("Rotation", [("Roll+", 1), ("Roll-", -1), ("Pitch+", 1), ("Pitch-", -1), ("Yaw+", 1), ("Yaw-", -1)])
        ]

        for title, buttons in controls:
            sub_frame = tk.Frame(cart_frame)
            sub_frame.pack()
            tk.Label(sub_frame, text=title, font=("Arial", 9)).pack()

            for label, _ in buttons:
                btn = tk.Button(sub_frame, text=label, width=8)
                self.button_states[label] = False
                btn.bind('<ButtonPress-1>', lambda e, n=label: self._on_button_press(n))
                btn.bind('<ButtonRelease-1>', lambda e, n=label: self._on_button_release(n))
                btn.pack()

        # Gripper control (Franka) or Hand control (XArm+Allegro)
        if self.has_gripper:
            gripper_frame = tk.Frame(frame)
            gripper_frame.pack(side=tk.LEFT, padx=10)
            tk.Label(gripper_frame, text="Gripper", font=("Arial", 10, "bold")).pack()

            open_btn = tk.Button(gripper_frame, text="Open", width=8, bg="lightblue")
            open_btn.bind('<ButtonPress-1>', lambda e: self._on_button_press("gripper_open"))
            open_btn.bind('<ButtonRelease-1>', lambda e: self._on_button_release("gripper_open"))
            self.button_states["gripper_open"] = False
            open_btn.pack(pady=2)

            grasp_btn = tk.Button(gripper_frame, text="Grasp", width=8, bg="lightyellow")
            grasp_btn.bind('<ButtonPress-1>', lambda e: self._on_button_press("gripper_grasp"))
            grasp_btn.bind('<ButtonRelease-1>', lambda e: self._on_button_release("gripper_grasp"))
            self.button_states["gripper_grasp"] = False
            grasp_btn.pack(pady=2)

            if self.has_guiding:
                guide_btn = tk.Button(gripper_frame, text="Guide Mode", width=8, bg="lightgreen",
                                      command=self._toggle_guiding_mode)
                guide_btn.pack(pady=2)
                self._guiding_active = False
                self._guide_btn = guide_btn

        elif self.hand is not None:
            hand_frame = tk.Frame(frame)
            hand_frame.pack(side=tk.LEFT, padx=10)
            tk.Label(hand_frame, text="Hand Control", font=("Arial", 10, "bold")).pack()

            for i in range(self.hand_dof):
                btn_frame = tk.Frame(hand_frame)
                btn_frame.pack()

                for sign, text in [(-1, f"H{i}-"), (1, f"H{i}+")]:
                    btn = tk.Button(btn_frame, text=text, width=6)
                    btn_name = f"hand_{i}_{sign}"
                    self.button_states[btn_name] = False
                    btn.bind('<ButtonPress-1>', lambda e, n=btn_name: self._on_button_press(n))
                    btn.bind('<ButtonRelease-1>', lambda e, n=btn_name: self._on_button_release(n))
                    btn.pack(side=tk.LEFT)

    # ==================== UI PUMP (thread-safe) ====================

    def _ui_pump(self):
        if not self.running:
            return

        if self._ui_dirty:
            self._update_status()
            self._ui_dirty = False

        # While the user is not driving them, sliders mirror the arm — so taking over
        # never starts from a stale value and jerks the robot.
        if self._joint_target is None and self._joint_sliders:
            data = self.robot.get_data()
            if data is not None:
                self._sync_sliders(data['qpos'])

        self._ui_pump_id = self.root.after(100, self._ui_pump)

    def _on_button_press(self, button_name):
        self.button_states[button_name] = True
        if self._is_manual_button(button_name):
            self.manual_override = True
            # Cartesian jogging and slider tracking both drive the same velocity
            # stream. Leaving a stale joint target set would yank the arm back to it
            # the moment the button is released.
            self._joint_target = None

    def _on_button_release(self, button_name):
        self.button_states[button_name] = False
        if not any(self.button_states[n] for n in self.button_states if self._is_manual_button(n)):
            self.manual_override = False

    def _is_manual_button(self, button_name):
        """Check if button is a manual control button"""
        return (button_name.startswith('hand_') or
                button_name.startswith('gripper_') or
                button_name in ['X+','X-','Y+','Y-','Z+','Z-',
                            'Roll+','Roll-','Pitch+','Pitch-','Yaw+','Yaw-'])

    def _toggle_guiding_mode(self):
        """Toggle hand-guiding mode (Franka only)."""
        if self._guiding_active:
            # Disable: move to current position to re-engage position control
            data = self.robot.get_data()
            if data is not None:
                self.robot.move(data["qpos"], speed_scale=0.1)
            self._guiding_active = False
            self._guide_btn.config(bg="lightgreen", text="Guide Mode")
            print("[GUI] Guiding mode OFF")
        else:
            self.robot.set_guiding_mode([True, True, True, True, True, True], nullspace=True)
            self._guiding_active = True
            self._guide_btn.config(bg="yellow", text="Guide ON")
            print("[GUI] Guiding mode ON - move robot by hand")

    def _on_start_press(self):
        """Start executing waypoints"""
        self.auto_execute = True

    def _on_start_release(self):
        self.auto_execute = False

    def stop_and_clear(self):
        self.auto_execute = False
        self.clear_queue()

    # ==================== PUBLIC API ====================

    def add_waypoint(self, name, wp_type, target=None, hand_qpos=None, threshold=None, repeat_ticks=10):
        if isinstance(wp_type, str):
            wp_type = (WaypointType.JOINT if wp_type.lower() == 'joint'
                       else WaypointType.CARTESIAN if wp_type.lower() == 'cartesian'
                       else None)

        if target is None and hand_qpos is None:
            print("Warning: Both target and hand_qpos are None; waypoint will have no effect.")
            raise ValueError("At least one of target or hand_qpos must be specified.")

        wp = Waypoint(name, wp_type, target, hand_qpos, threshold, repeat_ticks=repeat_ticks)
        self.waypoint_queue.append(wp)
        self._ui_dirty = True

    def clear_queue(self):
        self.waypoint_queue.clear()
        self.current_waypoint = None
        self._ui_dirty = True
        print("Queue cleared")

    def _update_status(self):
        queue_size = len(self.waypoint_queue)
        self.queue_label.config(text=f"Queue: {queue_size} waypoints")

        if self.current_waypoint:
            self.current_label.config(
                text=f"Current: {self.current_waypoint.name}",
                fg="blue"
            )
        else:
            self.current_label.config(text="Current: None", fg="gray")

    # ==================== WAYPOINT EXECUTION ====================

    def _check_in_waypoint(self, waypoint):
        if waypoint.target is None:
            return True

        if waypoint.type == WaypointType.JOINT:
            current_qpos = self.robot.get_data()['qpos']
            distance = np.linalg.norm(current_qpos - waypoint.target)
            return distance < waypoint.threshold

        # CARTESIAN
        if waypoint.type == WaypointType.CARTESIAN:
            current_pose = self.robot.get_data()['position']
            t_dist = np.linalg.norm(current_pose[:3, 3] - waypoint.target[:3, 3])

            current_rot = Rotation.from_matrix(current_pose[:3, :3])
            target_rot = Rotation.from_matrix(waypoint.target[:3, :3])
            r_dist = np.linalg.norm((target_rot * current_rot.inv()).as_rotvec())

            t_thresh, r_thresh = waypoint.threshold
            return t_dist < t_thresh and r_dist < r_thresh

        return False

    def _send_exact_target(self, waypoint):
        if waypoint.type == WaypointType.JOINT:
            self.robot.move(waypoint.target, is_servo=True)
        elif waypoint.type == WaypointType.CARTESIAN:
            self.robot.move(waypoint.target, is_servo=True)

        if waypoint.hand_qpos is not None and self.hand is not None:
            self.hand.move(waypoint.hand_qpos)

    def _execute_waypoint(self, waypoint):
        """Execute motion toward waypoint"""
        # Update hand if specified
        if waypoint.hand_qpos is not None and self.hand is not None:
            current_hand_qpos = self.hand.get_data()['qpos']
            hand_delta = waypoint.hand_qpos - current_hand_qpos

            hand_delta_norm = np.linalg.norm(hand_delta)
            if hand_delta_norm > 0:
                hand_delta = hand_delta / hand_delta_norm * min(self.hand_vel_limit, hand_delta_norm)

            self.hand.move(waypoint.hand_qpos)

        if waypoint.type == WaypointType.JOINT:
            current_qpos = self.robot.get_data()['qpos']
            delta = waypoint.target - current_qpos

            # Limit velocity
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 0:
                delta = delta / delta_norm * min(self.joint_vel_limit, delta_norm)

            self.robot.move(current_qpos + delta, is_servo=True)

        elif waypoint.type == WaypointType.CARTESIAN:  # CARTESIAN
            current_pose = self.robot.get_data()['position'].copy()

            # Translation
            t_delta = waypoint.target[:3, 3] - current_pose[:3, 3]
            t_dist = np.linalg.norm(t_delta)

            if t_dist > 0.001:
                t_step = min(self.cart_vel_limit / 1000.0, t_dist)
                current_pose[:3, 3] += (t_delta / t_dist) * t_step

            # Rotation
            current_rot = Rotation.from_matrix(current_pose[:3, :3])
            target_rot = Rotation.from_matrix(waypoint.target[:3, :3])
            r_delta = (target_rot * current_rot.inv()).as_rotvec()
            r_dist = np.linalg.norm(r_delta)

            if r_dist > 0.01:
                r_step = min(self.rot_vel_limit, r_dist)
                partial_rot = Rotation.from_rotvec((r_delta / r_dist) * r_step)
                current_pose[:3, :3] = (partial_rot * current_rot).as_matrix()

            self.robot.move(current_pose, is_servo=True)


    def _trajectory_finished(self, waypoint):
        """Whether precomputed trajectories are fully played (if they exist)."""
        robot_fin = (waypoint.trajectory is None) or (waypoint.robot_step >= len(waypoint.trajectory))
        hand_fin = (waypoint.hand_trajectory is None) or (waypoint.hand_step >= len(waypoint.hand_trajectory))
        return robot_fin and hand_fin

    def _is_waypoint_done(self, waypoint):
        if self._check_in_waypoint(waypoint):
            waypoint.repeat_count += 1
            self._send_exact_target(waypoint)

            return waypoint.repeat_count >= waypoint.repeat_ticks
        else:
            waypoint.repeat_count = 0
            return False

    # ==================== MANUAL CONTROL ====================

    def _wait_for_standstill(self, tol=0.02, timeout=2.0):
        """Block until the arm has actually stopped (or we give up)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.robot.get_data()
            if data is None:
                return
            qvel = data.get('qvel')
            if qvel is None or np.abs(qvel).max() < tol:
                return
            time.sleep(0.02)

    def _twist_to_dq(self, twist):
        """Resolve a base-frame twist to joint velocities (damped least squares).

        Returns None when no URDF was supplied, leaving the caller to fall back to the
        daemon's cartesian stream.

        Why this exists: that stream rate-limits the *twist*, not the joint motion it
        implies. Near a limit or singularity a mild cartesian rate needs a joint jerk,
        libfranka aborts with "..._joint_acceleration_discontinuity", the stream is
        torn down and reopened — and each of those cycles is a visible stutter. The
        joint velocity stream has no such failure mode. Damping trades a little
        accuracy near singularities for bounded joint speed.
        """
        if self._robot_wrapper is None or self._eef_link_id is None:
            return None

        data = self.robot.get_data()
        if data is None:
            return None

        try:
            J = self._robot_wrapper.compute_single_link_local_jacobian(
                data['qpos'], self._eef_link_id)

            # That Jacobian is expressed in the LOCAL (end-effector) frame, while
            # `twist` is given in base axes — feeding it in directly sends the arm off
            # in the wrong direction. Rotate the request into the EE frame first.
            R = data['position'][:3, :3]
            twist_local = np.concatenate([R.T @ twist[:3], R.T @ twist[3:]])

            lam = self.dls_damping
            dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), twist_local)
        except Exception as e:
            print(f"[jog] jacobian failed ({e}); using the cartesian stream")
            return None

        # Keep the slowest-but-safe interpretation: scale the whole vector rather than
        # clipping per joint, so the motion direction is preserved.
        peak = np.abs(dq).max()
        if peak > self.jog_joint_vel_max:
            dq = dq * (self.jog_joint_vel_max / peak)
        return dq

    def _stream_jog(self, kind, target):
        """Send a streaming velocity setpoint.

        No client-side ramping: the daemon's 1 kHz control callback already rate-limits
        every setpoint change (10 rad/s^2 per joint, 5 m/s^2 linear) and starts each
        stream from zero. Ramping here as well produced the stutter — the ramp emitted
        a command per control tick, each a blocking ZMQ round trip, so the arm got the
        acceleration in coarse jumps instead of the daemon's smooth 1 kHz interpolation.

        Sending only on change matters though: `duration_ms=0` is streaming mode, where
        the daemon holds the last setpoint until it is replaced or the stream stops.
        """
        target = np.asarray(target, dtype=float)

        if self._jog_kind != kind or self._jog_vec is None:
            # The daemon runs one control loop at a time, and stop_streaming only
            # *starts* a ramp-down. Opening the next stream while the arm is still
            # coasting gives "Motion finished commanded, but the robot is still
            # moving!" and an immediate reflex abort — so stop, then wait for standstill.
            if self._velocity_streaming and self._jog_kind is not None:
                self.robot.stop_streaming()
                self._velocity_streaming = False
                self._wait_for_standstill()
            self._jog_kind = kind
            self._jog_vec = np.zeros_like(target)

        # Skip near-duplicate setpoints. Without this a proportional controller (the
        # slider tracker) emits a slightly different velocity every tick, turning the
        # motion into a series of blocking round trips — the stutter.
        if (self._last_jog_cmd is not None and self._last_jog_cmd[0] == kind
                and np.all(np.abs(target - self._last_jog_cmd[1]) <
                           self.jog_send_deadband * max(1.0, np.abs(target).max()))
                and np.any(target) == np.any(self._last_jog_cmd[1])):
            self._jog_vec = target
            return

        self._jog_vec = target
        cmd = (kind, target.copy())

        if kind == 'j':
            resp = self.robot.set_joint_velocity(target, duration_ms=0)
        else:
            resp = self.robot.set_cartesian_velocity(target, duration_ms=0)

        # A reflex kills the daemon's stream thread AND latches the robot into an
        # error state, where every later command is refused — that is what makes
        # jogging appear to die permanently after one abort. Clear it, and forget the
        # last setpoint so the next command is actually sent instead of being skipped
        # as a duplicate.
        if isinstance(resp, dict) and resp.get("type") == "error":
            print(f"[jog] stream error: {resp.get('message')}")
            self._jog_vec = np.zeros_like(target)
            self._last_jog_cmd = None
            self._velocity_streaming = False
            self._jog_kind = None
            recover = getattr(self.robot, 'error_recovery', None)
            if callable(recover):
                recover()
            return

        self._last_jog_cmd = cmd
        self._velocity_streaming = True

    def _track_joint_target(self):
        """Drive the arm toward the slider target.

        Constant velocity toward the target, decelerating only over the last stretch,
        rather than a pure proportional law. A proportional velocity changes every
        tick, and every change costs a blocking round trip — that is what made slider
        motion stutter. Here the setpoint is flat for most of the travel, so the
        deadband in `_stream_jog` collapses it to a couple of sends.
        """
        if not self.jog_by_velocity:
            # xArm servo mode takes position targets directly.
            self.robot.move(self._joint_target, is_servo=True)
            return

        data = self.robot.get_data()
        if data is None:
            return

        error = self._joint_target - data['qpos']
        if np.abs(error).max() < self.slider_tol:
            if self._velocity_streaming:
                self._brake_jog()
            return

        # Full speed until close, then scale down so it settles instead of overshooting.
        approach = min(1.0, np.abs(error).max() / self.slider_approach_band)
        vel = self.slider_vel * approach * error / max(np.abs(error).max(), 1e-9)
        self._stream_jog('j', vel)

    def _brake_jog(self):
        """Command zero velocity, and only stop the stream once the arm has stopped.

        `stop_streaming` makes the daemon return MotionFinished as soon as its own
        commanded velocity reaches zero, but the arm lags that by a few ms. Ending the
        motion while it is still moving gives "Motion finished commanded, but the robot
        is still moving!" and a reflex abort — so wait for real standstill first.
        """
        if self._jog_kind is None or self._jog_vec is None:
            self._velocity_streaming = False
            return

        self._stream_jog(self._jog_kind, np.zeros_like(self._jog_vec))

        data = self.robot.get_data()
        qvel = None if data is None else data.get('qvel')
        if qvel is not None and np.abs(qvel).max() > 0.02:
            return          # still coasting; hold the zero setpoint and re-check

        self.robot.stop_streaming()
        self._velocity_streaming = False
        self._last_jog_cmd = None
        self._jog_kind = None
        self._jog_vec = None

    def _execute_manual_control(self, pressed_buttons):
        # A velocity stream keeps running at the last setpoint until told otherwise,
        # so release of the last jog button has to ramp it down. Checked before the
        # per-axis handling below, which is what (re)starts it.
        if self._velocity_streaming and not any(
                b in _CART_BUTTONS for b in pressed_buttons):
            self._brake_jog()

        cart_map = {
            'X+': ('t', 0, 1), 'X-': ('t', 0, -1),
            'Y+': ('t', 1, 1), 'Y-': ('t', 1, -1),
            'Z+': ('t', 2, 1), 'Z-': ('t', 2, -1),
            'Roll+': ('r', 0, 1), 'Roll-': ('r', 0, -1),
            'Pitch+': ('r', 1, 1), 'Pitch-': ('r', 1, -1),
            'Yaw+': ('r', 2, 1), 'Yaw-': ('r', 2, -1),
        }

        cart_pressed = [b for b in pressed_buttons if b in cart_map]
        if cart_pressed:
            current_pose = self.robot.get_data()['position'].copy()
            t_delta = np.zeros(3)
            r_delta = np.zeros(3)

            for btn in cart_pressed:
                mode, axis, sign = cart_map[btn]
                if mode == 't':
                    t_delta[axis] = sign * self.cart_vel_limit
                else:
                    r_delta[axis] = sign * self.rot_vel_limit

            if np.any(t_delta != 0) or np.any(r_delta != 0):
                if self.jog_by_velocity:
                    omega = np.sign(r_delta) * self.jog_rot_vel
                    if self.rotate_in_tool_frame and np.any(omega):
                        # libfranka's CartesianVelocities twist is expressed in the
                        # BASE frame, so Roll/Pitch/Yaw would spin the tool about the
                        # robot's axes — unintuitive when posing an end effector.
                        # Rotate the requested tool-frame rate into base coordinates.
                        omega = current_pose[:3, :3] @ omega
                    twist = np.concatenate([
                        np.sign(t_delta) * self.jog_cart_vel,
                        omega,
                    ])

                    # Prefer resolving the twist to joint velocities ourselves — the
                    # daemon's cartesian stream keeps aborting on reflexes here, and
                    # every abort/restart is a visible hitch.
                    dq = self._twist_to_dq(twist)
                    if dq is not None:
                        self._stream_jog('j', dq)
                    else:
                        self._stream_jog('c', twist)
                else:
                    current_pose[:3, 3] += t_delta / 1000.0

                    if np.any(r_delta != 0):
                        current_rot = Rotation.from_matrix(current_pose[:3, :3])
                        delta_rot = Rotation.from_euler('xyz', r_delta)
                        current_pose[:3, :3] = (current_rot * delta_rot).as_matrix()

                    self.robot.move(current_pose, is_servo=True)

        # Gripper (Franka)
        if self.has_gripper:
            if self.button_states.get("gripper_open"):
                self.robot.open_gripper()
                self.button_states["gripper_open"] = False  # one-shot
            if self.button_states.get("gripper_grasp"):
                self.robot.grasp()
                self.button_states["gripper_grasp"] = False  # one-shot

        # Hand (Allegro etc.)
        if self.hand is not None:
            hand_pressed = [b for b in pressed_buttons if b.startswith('hand_')]
            if hand_pressed:
                current_hand_qpos = self.hand.get_data()['qpos'].copy()
                delta = np.zeros(self.hand_dof)

                for btn in hand_pressed:
                    _, h, s = btn.split('_')
                    hand_idx = int(h)
                    sign = int(s)
                    if hand_idx < self.hand_dof:
                        delta[hand_idx] = sign * self.hand_vel_limit * 10
                if np.any(delta != 0):
                    self.hand.move(current_hand_qpos + delta)

    # ==================== CONTROL LOOP ====================

    def _control_loop(self):
        while self.running:
            pressed_buttons = [n for n, s in self.button_states.items() if s]

            if self.manual_override:
                self._execute_manual_control(pressed_buttons)
            elif self._joint_target is not None:
                self._track_joint_target()
            elif self._velocity_streaming:
                # manual_override drops the instant the last button is released, so
                # the brake ramp has to live out here — inside _execute_manual_control
                # it would never run and the arm would coast at the last velocity.
                self._brake_jog()
            elif self.auto_execute:
                if self.current_waypoint is not None and self._is_waypoint_done(self.current_waypoint):
                    print(f"Waypoint '{self.current_waypoint.name}' completed.")
                    self.current_waypoint = None
                    self._ui_dirty = True

                if self.current_waypoint is None and len(self.waypoint_queue) > 0:
                    self.current_waypoint = self.waypoint_queue.popleft()
                    self._ui_dirty = True

                if self.current_waypoint is not None:
                    self._execute_waypoint(self.current_waypoint)

            if (not self.jog_only
                    and self.current_waypoint is None and len(self.waypoint_queue) == 0):
                self.root.after(0, self._on_exit)
                return

            time.sleep(0.01)

    def _build_slider_panel(self):
        """One slider per joint, plus arrow-key nudging of the selected joint.

        Sliders follow the robot until you touch one; from then on the control loop
        drives the arm to the slider values. Release tracking with "Follow robot".
        """
        frame = tk.LabelFrame(self.root, text="Joint Sliders  (click a joint, then Up/Down)",
                              font=("Arial", 12))
        frame.pack(pady=6, padx=10, fill="both")

        for i in range(self.arm_dof):
            row = tk.Frame(frame)
            row.pack(fill="x")

            label = tk.Label(row, text=f"J{i+1}", width=4, font=("Courier", 10, "bold"))
            label.pack(side=tk.LEFT)

            lo, hi = self._joint_limits[i]
            # No `command=` callback: tk fires it for programmatic .set() too, and it
            # arrives through the event queue, so a sync could not be told apart from
            # a drag by value or by flag — the arm kept snapping back to a stale angle.
            # Mouse bindings only ever fire for real user interaction.
            slider = tk.Scale(row, from_=np.degrees(lo), to=np.degrees(hi),
                              resolution=0.5, orient=tk.HORIZONTAL, length=380)
            slider.pack(side=tk.LEFT, fill="x", expand=True)
            slider.bind('<Button-1>', lambda e, j=i: self._select_joint(j))
            slider.bind('<B1-Motion>', lambda e, j=i: self._on_slider_drag(j))
            slider.bind('<ButtonRelease-1>', lambda e, j=i: self._on_slider_drag(j))
            label.bind('<Button-1>', lambda e, j=i: self._select_joint(j))
            self._joint_sliders.append(slider)

            # Type an exact angle and press Enter.
            entry = tk.Entry(row, width=8, justify=tk.RIGHT)
            entry.pack(side=tk.LEFT, padx=4)
            entry.bind('<Return>', lambda e, j=i: self._on_entry(j))
            self._joint_entries.append(entry)

            tk.Label(row, text="deg", font=("Arial", 8)).pack(side=tk.LEFT)

        btn_row = tk.Frame(frame)
        btn_row.pack(pady=4)
        tk.Button(btn_row, text="Follow robot", width=14,
                  command=self._release_tracking).pack(side=tk.LEFT, padx=4)
        self._sel_label = tk.Label(btn_row, text="selected: J1", font=("Arial", 10))
        self._sel_label.pack(side=tk.LEFT, padx=8)

        self.root.bind('<Up>', lambda e: self._nudge_joint(+1))
        self.root.bind('<Down>', lambda e: self._nudge_joint(-1))
        self.root.bind('<Left>', lambda e: self._select_joint(
            (self._selected_joint - 1) % self.arm_dof))
        self.root.bind('<Right>', lambda e: self._select_joint(
            (self._selected_joint + 1) % self.arm_dof))
        self.root.focus_set()

    def _select_joint(self, j):
        self._selected_joint = j
        self._sel_label.config(text=f"selected: J{j+1}")
        for i, s in enumerate(self._joint_sliders):
            s.config(troughcolor='lightyellow' if i == j else 'lightgray')

    def _on_slider_drag(self, j):
        """User dragged slider `j` — bound to mouse events, so never fires for a sync."""
        self._select_joint(j)
        self._begin_tracking()
        if self._joint_target is None:
            return

        deg = float(self._joint_sliders[j].get())
        self._joint_target[j] = np.radians(deg)
        self._set_entry(j, deg)

    def _on_entry(self, j):
        """Enter in a joint's entry box: go to that exact angle (degrees)."""
        try:
            deg = float(self._joint_entries[j].get())
        except ValueError:
            self._joint_entries[j].config(bg='mistyrose')
            return
        self._joint_entries[j].config(bg='white')

        lo, hi = self._joint_limits[j]
        self._select_joint(j)
        self._begin_tracking()
        if self._joint_target is None:
            return

        self._joint_target[j] = float(np.clip(np.radians(deg), lo, hi))
        self._sync_sliders(self._joint_target)

    def _nudge_joint(self, direction):
        j = self._selected_joint
        self._begin_tracking()
        if self._joint_target is None:
            return

        lo, hi = self._joint_limits[j]
        target = self._joint_target[j] + direction * np.radians(self.slider_step_deg)
        self._joint_target[j] = float(np.clip(target, lo, hi))
        self._sync_sliders(self._joint_target)

    def _begin_tracking(self):
        """Seed the target from the live pose the first time the user takes over."""
        if self._joint_target is not None:
            return
        data = self.robot.get_data()
        if data is not None:
            self._joint_target = data['qpos'].copy()

    def _release_tracking(self):
        self._joint_target = None

    def _set_entry(self, j, deg):
        """Write a value into joint j's entry box, unless it is being typed into."""
        entry = self._joint_entries[j]
        if self.root.focus_get() is entry:
            return
        entry.delete(0, tk.END)
        entry.insert(0, f"{deg:.1f}")

    def _sync_sliders(self, qpos):
        for i, s in enumerate(self._joint_sliders):
            deg = round(float(np.degrees(qpos[i])), 1)
            s.set(deg)
            self._set_entry(i, deg)

    def _resolve_joint_limits(self):
        """Per-joint (lower, upper) in rad.

        Read from the robot if it exposes them. The shipped `franka.urdf` is not a
        safe source: it caps fr3_joint5 at ±2.48 rad while the real arm reaches
        -2.87 (measured), so URDF-derived margins read as negative on valid poses.
        Falls back to FR3 datasheet values. Treat the margins as advisory: real poses
        have been observed ~0.07 rad outside the nominal fr3_joint5 range, so a small
        negative margin means "at the edge", not "impossible".
        """
        getter = getattr(self.robot, 'get_joint_limits', None)
        if callable(getter):
            try:
                limits = getter()
                if limits is not None and len(limits) == self.arm_dof:
                    return [tuple(v) for v in limits]
            except Exception:
                pass

        if self.arm_dof == 7:      # FR3 datasheet
            return [(-2.7437, 2.7437), (-1.7837, 1.7837), (-2.9007, 2.9007),
                    (-3.0421, -0.1518), (-2.8065, 2.8065), (0.5445, 4.5169),
                    (-3.0159, 3.0159)]
        return [(-np.pi, np.pi)] * self.arm_dof

    def _build_save_panel(self):
        os.makedirs(self.save_path, exist_ok=True)

        # Continue after existing waypoints; overwriting only the first N would mix
        # old and new poses into one trajectory.
        existing = [int(f.split('_')[0]) for f in os.listdir(self.save_path)
                    if '_qpos' in f]
        self._save_idx = max(existing) + 1 if existing else 0

        frame = tk.LabelFrame(self.root, text="Teach: Save Pose", font=("Arial", 12))
        frame.pack(pady=10, padx=10, fill="both")

        tk.Button(frame, text="Save Pose", width=20, bg="lightgreen",
                  font=("Arial", 11, "bold"), command=self._save_pose).pack(pady=4)

        # A single retracted pose the replay can route through when two waypoints are
        # far apart, so the arm does not sweep the board along the floor.
        tk.Button(frame, text="Save VIA (safe transit)", width=20, bg="lightyellow",
                  command=self._save_via_pose).pack(pady=2)

        self._save_label = tk.Label(frame, text=f"Saved: {self._save_idx} pose(s)",
                                    font=("Arial", 10))
        self._save_label.pack()
        tk.Label(frame, text=self.save_path, font=("Arial", 8), fg="gray").pack()

    def _save_via_pose(self):
        """Save the current pose as `via_qpos.npy` (not part of the waypoint list)."""
        data = self.robot.get_data()
        if data is None:
            self._save_label.config(text="No robot state — is the daemon alive?", fg="red")
            return

        np.save(os.path.join(self.save_path, 'via_qpos.npy'), data['qpos'])
        print("Saved via pose (safe transit)")
        self._save_label.config(text=f"Saved: {self._save_idx} pose(s)  + via", fg="black")

    def _save_pose(self):
        data = self.robot.get_data()
        if data is None:
            self._save_label.config(text="No robot state — is the daemon alive?", fg="red")
            return

        qpos = data['qpos']
        pose = data['position']
        np.save(os.path.join(self.save_path, f'{self._save_idx}_qpos.npy'), qpos)
        np.save(os.path.join(self.save_path, f'{self._save_idx}_pose.npy'), pose)
        print(f"Saved pose {self._save_idx}: "
              f"EE=[{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")

        self._save_idx += 1
        self._save_label.config(text=f"Saved: {self._save_idx} pose(s)", fg="black")

    def _on_exit(self):
        print("Exiting...")
        self.running = False
        if self._velocity_streaming:
            self.robot.stop_streaming()
            self._velocity_streaming = False
            self._last_jog_cmd = None
        self.root.destroy()

    def run(self):

        self.control_thread = Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        self.root.mainloop()