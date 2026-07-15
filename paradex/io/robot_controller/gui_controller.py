import tkinter as tk
import time
import numpy as np
from threading import Thread
from scipy.spatial.transform import Rotation
from collections import deque
from enum import Enum

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
    def __init__(self, robot_controller, hand_controller=None):
        print(">>> Initializing Robot GUI Controller...")

        self.robot = robot_controller
        self.hand = hand_controller

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

        # Control parameters
        self.joint_vel_limit = 0.06  # rad per tick
        self.cart_vel_limit = 1.0    # mm per tick
        self.rot_vel_limit = 0.01    # rad per tick
        self.hand_vel_limit = 0.05  # per tick

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

        # Joint control
        joint_frame = tk.Frame(frame)
        joint_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(joint_frame, text=f"Joint Control ({self.arm_dof}DOF)",
                 font=("Arial", 10, "bold")).pack()

        for i in range(self.arm_dof):
            btn_frame = tk.Frame(joint_frame)
            btn_frame.pack()

            for sign, text in [(-1, f"J{i}-"), (1, f"J{i}+")]:

                btn = tk.Button(btn_frame, text=text, width=6)
                btn_name = f"joint_{i}_{sign}"
                self.button_states[btn_name] = False
                btn.bind('<ButtonPress-1>', lambda e, n=btn_name: self._on_button_press(n))
                btn.bind('<ButtonRelease-1>', lambda e, n=btn_name: self._on_button_release(n))
                btn.pack(side=tk.LEFT)

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

        self._ui_pump_id = self.root.after(100, self._ui_pump)

    # ==================== BUTTON HANDLERS ====================

    def _on_button_press(self, button_name):
        self.button_states[button_name] = True
        if self._is_manual_button(button_name):
            self.manual_override = True

    def _on_button_release(self, button_name):
        self.button_states[button_name] = False
        if not any(self.button_states[n] for n in self.button_states if self._is_manual_button(n)):
            self.manual_override = False

    def _is_manual_button(self, button_name):
        """Check if button is a manual control button"""
        return (button_name.startswith('joint_') or
                button_name.startswith('hand_') or
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

    def _execute_manual_control(self, pressed_buttons):
        joint_pressed = [b for b in pressed_buttons if b.startswith('joint_')]
        if joint_pressed:
            current_qpos = self.robot.get_data()['qpos'].copy()
            delta = np.zeros(self.arm_dof)

            for btn in joint_pressed:
                _, j, s = btn.split('_')
                joint_idx = int(j)
                sign = int(s)
                if joint_idx < self.arm_dof:
                    delta[joint_idx] = sign * self.joint_vel_limit

            if np.any(delta != 0):
                self.robot.move(current_qpos + delta, is_servo=True)

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
                print("Manual override active")
                self._execute_manual_control(pressed_buttons)
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

            if self.current_waypoint is None and len(self.waypoint_queue) == 0:
                self.root.after(0, self._on_exit)
                return

            time.sleep(0.01)

    def _on_exit(self):
        print("Exiting...")
        self.running = False
        self.root.destroy()

    def run(self):

        self.control_thread = Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        self.root.mainloop()