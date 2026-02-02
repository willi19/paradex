import tkinter as tk
import time
import numpy as np
from threading import Thread
from scipy.spatial.transform import Rotation
from .xarm_controller import cart2homo


class QposTrajectoryGUIController:
    """
    Arm(6) + Hand(16) 로 이루어진 qpos trajectory 만으로 제어하는 단순 GUI.
    - Home Pose 보내기
    - Trajectory 실행하기

    Args:
        robot_controller: XArmController 와 같은 인터페이스 (move, get_data, end)
        hand_controller: AllegroController 와 같은 인터페이스 (move, end)
        home_qpos: (22,) = [arm6, hand16]
        traj_qpos: (T, 22) = [arm6, hand16] trajectory
        dt: trajectory 한 스텝당 시간 간격 (초 단위, 기본 0.01s)
    """

    def __init__(self, robot_controller, hand_controller,
                 home_qpos: np.ndarray,
                 traj_qpos: np.ndarray,
                 dt: float = 0.01):
        print(">>> Initializing Qpos Trajectory GUI Controller...")
        self.robot = robot_controller
        self.hand = hand_controller

        self.home_qpos = np.asarray(home_qpos).astype(float).copy()
        self.traj_qpos = np.asarray(traj_qpos).astype(float).copy()
        assert self.home_qpos.shape == (22,), f"home_qpos shape must be (22,), got {self.home_qpos.shape}"
        assert self.traj_qpos.ndim == 2 and self.traj_qpos.shape[1] == 22, \
            f"traj_qpos shape must be (T, 22), got {self.traj_qpos.shape}"

        self.dt = float(dt)
        self._running = True

        # step size for pose interpolation (meter / rad)
        self.cart_delta = 1.0   # mm 단위 (원래 GUI와 동일)
        self.angle_delta = 0.01 # rad

        # 버튼 상태 (눌려있는 동안만 동작하도록)
        self.button_states = {}

        # Trajectory 진행 인덱스
        self.traj_index = 0.0
        
        # GUI 생성
        self.root = tk.Tk()
        self._build_gui()

        # 창 닫기 핸들러
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)

    def _build_gui(self):
        self.root.title("Qpos Trajectory Controller")
        self.root.geometry("400x250")

        title = tk.Label(self.root, text="Qpos Trajectory Controller",
                         font=("Arial", 14, "bold"))
        title.pack(pady=15)

        info = tk.Label(
            self.root,
            text=f"Home / Traj dims: 22 (6 arm + 16 hand)\nTraj length: {self.traj_qpos.shape[0]}",
            font=("Arial", 10)
        )
        info.pack(pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)

        # Home 버튼 (눌려있는 동안만 home 으로 이동)
        home_btn = tk.Button(
            btn_frame,
            text="Go Home Pose (hold)",
            width=18,
            bg="lightblue",
            font=("Arial", 11, "bold"),
        )
        home_name = "home"
        self.button_states[home_name] = False
        home_btn.bind('<ButtonPress-1>', lambda e, name=home_name: self._on_button_press(name))
        home_btn.bind('<ButtonRelease-1>', lambda e, name=home_name: self._on_button_release(name))
        home_btn.pack(pady=5)

        # Trajectory 버튼 (눌려있는 동안만 trajectory 진행)
        traj_btn = tk.Button(
            btn_frame,
            text="Run Trajectory (hold)",
            width=18,
            bg="lightgreen",
            font=("Arial", 11, "bold"),
        )
        traj_name = "traj"
        self.button_states[traj_name] = False
        traj_btn.bind('<ButtonPress-1>', lambda e, name=traj_name: self._on_button_press(name))
        traj_btn.bind('<ButtonRelease-1>', lambda e, name=traj_name: self._on_button_release(name))
        traj_btn.pack(pady=5)

        exit_btn = tk.Button(
            self.root,
            text="Exit",
            width=18,
            bg="red",
            fg="white",
            command=self._on_exit,
        )
        exit_btn.pack(pady=15)

        self.status_label = tk.Label(self.root, text="Idle", font=("Arial", 10))
        self.status_label.pack(pady=5)
        # 컨트롤 루프 시작 (버튼이 눌려있는 동안만 모션 실행)
        self.control_thread = Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

    # ---------------- Button / 상태 관련 ----------------
    def _set_status(self, text: str):
        print(text)
        self.status_label.config(text=text)

    def _on_button_press(self, name: str):
        self.button_states[name] = True
        print(f"{name} pressed")

    def _on_button_release(self, name: str):
        self.button_states[name] = False

    # ---------------- Control loop ----------------
    def _control_loop(self):
        """눌린 버튼 상태를 계속 확인하는 루프 (안전하게 hold-to-run)"""
        while self._running:
            pressed = [n for n, s in self.button_states.items() if s]

            if 'home' in pressed:
                self._step_to_home()
            elif 'traj' in pressed:
                self._step_traj()

            time.sleep(self.dt)

        print(">>> QposTrajectory control loop exited")

    # ---------------- 동작 구현 ----------------
    def _step_to_home(self):
        """home_qpos 를 향해 pose 보간으로 이동 (버튼 눌려있는 동안만)"""
        try:
            # Arm: 현재 pose, target pose 계산
            target_arm_q = self.home_qpos[:6]
            target_pose = self._fk_from_qpos(target_arm_q)
            reached = self._move_to_pose(target_pose)

            # Hand: joint space interpolation (qpos 보간)
            data_hand = self.hand.get_data()
            current_hand = np.asarray(data_hand['qpos']).astype(float)
            target_hand = self.home_qpos[6:]
            hand_delta = target_hand - current_hand
            hand_dist = np.linalg.norm(hand_delta)
            if hand_dist > 1e-4:
                step_h = min(0.05, hand_dist)
                new_hand = current_hand + hand_delta / max(hand_dist, 1e-8) * step_h
                self.hand.move(new_hand)

            if reached:
                self._set_status("Home pose reached.")
            else:
                self._set_status("Moving to home (hold)...")
        except Exception as e:
            self._set_status(f"Home step error: {e}")

    def _step_traj(self):
        """traj_qpos 를 순서대로 따라감 (버튼 눌려있는 동안만 진행)"""
        try:
            if self.traj_index >= len(self.traj_qpos):
                self._set_status("Trajectory complete.")
                return

            idx = int(self.traj_index)
            q = self.traj_qpos[idx]
            arm_q = q[:6]
            hand_q = q[6:]

            # 손은 바로 qpos 명령
            self.hand.move(hand_q)
            # 팔은 servo joint 명령
            self.robot.move(arm_q, is_servo=True)

            self.traj_index += 1.0  # 한 번에 한 스텝씩 진행
            self._set_status(f"Running traj: {idx+1}/{len(self.traj_qpos)} (hold)")
        except Exception as e:
            self._set_status(f"Traj step error: {e}")

    # ----------- 기존 GUI의 _move_to_pose 보간 그대로 사용 ----------- 
    def _move_to_pose(self, target_pose):
        """Helper function to move to a target pose (원래 GUI와 동일한 보간 로직)"""
        current_pose = self.robot.get_data()['position'].copy()
        
        # Translation delta
        t_delta = target_pose[:3, 3] - current_pose[:3, 3]
        t_distance = np.linalg.norm(t_delta)
        
        # Rotation delta
        current_rot = Rotation.from_matrix(current_pose[:3, :3])
        target_rot = Rotation.from_matrix(target_pose[:3, :3])
        delta_rot = target_rot * current_rot.inv()
        r_delta = delta_rot.as_rotvec()
        r_distance = np.linalg.norm(r_delta)
        
        if t_distance < 0.001 and r_distance < 0.01:
            return True  # Reached
        
        # 새로운 pose 계산
        new_pose = current_pose.copy()
        
        # Translation 보간
        if t_distance > 0.001:
            t_step_size = min(self.cart_delta / 1000, t_distance)
            new_pose[:3, 3] = current_pose[:3, 3] + (t_delta / t_distance) * t_step_size
        
        # Rotation 보간
        if r_distance > 0.01:
            r_step_size = min(self.angle_delta, r_distance)
            partial_rot = Rotation.from_rotvec((r_delta / r_distance) * r_step_size)
            new_rot = partial_rot * current_rot
            new_pose[:3, :3] = new_rot.as_matrix()
        
        self.robot.move(new_pose, is_servo=True)
        return False

    def _fk_from_qpos(self, qpos: np.ndarray):
        """
        Joint qpos(6,) -> 4x4 homogeneous pose.
        XArmController 내부의 XArmAPI.forward_kinematics 를 직접 사용 (컨트롤러 코드는 그대로 두고 GUI 에서만 사용).
        """
        qpos = np.asarray(qpos).astype(float)
        assert qpos.shape[0] == 6, f"qpos must have shape (6,), got {qpos.shape}"
        # XArmAPI.get_forward_kinematics 는 cartesian 6D (x,y,z,r,p,y) 를 반환하므로, cart2homo 로 4x4 로 변환
        cart = np.array(
            self.robot.arm.get_forward_kinematics(
                qpos.tolist(), input_is_radian=True, return_is_radian=True
            )[1]
        )
        return cart2homo(cart)

    # ---------------- Lifecycle ----------------
    def _on_exit(self):
        self._running = False
        # control loop thread 종료 대기
        try:
            self.control_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.robot.end()
        except Exception:
            pass
        try:
            self.hand.end()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        """GUI 메인 루프 실행"""
        self.root.mainloop()