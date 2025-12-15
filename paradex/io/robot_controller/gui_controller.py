import tkinter as tk
import time
import numpy as np
from threading import Thread
from scipy.spatial.transform import Rotation


class RobotGUIController:
    def __init__(self, robot_controller, hand_controller, predefined_poses=None, grasp_pose=None):
        """
        Args:
            robot_controller: XArmController 또는 유사한 인터페이스를 가진 로봇 컨트롤러
            hand_controller: Hand controller
            predefined_poses: dict, 미리 정의된 포즈들 (선택사항)
            grasp_pose: dict with 'start' and 'grasp' keys containing hand poses
        """
        
        self.predefined_poses = predefined_poses
        self.predefined_hand_traj = grasp_pose  # {'start': pose, 'grasp': pose}
        self.robot = robot_controller
        self.hand = hand_controller
        
        # Control parameters
        self.joint_delta = 0.01  # 관절 각도 변화량 (radian)
        self.cart_delta = 1.0    # cartesian 위치 변화량 (mm)
        self.angle_delta = 0.01  # orientation 변화량 (radian)
        
        # Hand grasp parameters
        self.hand_grasp_speed = 0.005  # trajectory interpolation step (0~1)
        self.hand_grasp_progress = 0.0  # current progress along trajectory (0~1)
        
        # Button states
        self.button_states = {}
        self.control_mode = 'joint'  # 'joint' or 'cartesian'
        self.running = True

        # Control loop start
        self.control_thread = Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        # GUI 
        self.root = tk.Tk()
        self._build_gui()
    
    def _build_gui(self):
        self.root.title("Robot GUI Controller")
        self.root.geometry("900x700")
        
        # 제목
        title = tk.Label(self.root, text="Robot Controller", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # 모드 표시 및 전환
        self._build_mode_frame()
        
        # Joint Control
        self._build_joint_frame()
        
        # Cartesian Control
        self._build_cartesian_frame()
        
        # Predefined Poses
        self._build_pose_frame()
        
        # Hand Grasp Control
        self._build_hand_grasp_frame()
        
        # Exit Button
        tk.Button(self.root, text="Exit", width=20, bg="red", fg="white",
                  command=self._on_exit).pack(pady=10)
        
        # Window close handling
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
    
    def _build_mode_frame(self):
        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=5)
        
        self.mode_label = tk.Label(mode_frame, text=f"Mode: {self.control_mode.upper()}", 
                                   font=("Arial", 12, "bold"))
        self.mode_label.pack(side=tk.LEFT, padx=10)
        
        tk.Button(mode_frame, text="Toggle Mode", command=self._toggle_mode, 
                  bg="orange").pack(side=tk.LEFT)
    
    def _build_joint_frame(self):
        joint_frame = tk.LabelFrame(self.root, text="Joint Control", font=("Arial", 12))
        joint_frame.pack(pady=10, padx=10, fill="both")
        
        joints = [
            ("Joint 0", "-", "+"),
            ("Joint 1", "-", "+"),
            ("Joint 2", "-", "+"),
            ("Joint 3", "-", "+"),
            ("Joint 4", "-", "+"),
            ("Joint 5", "-", "+"),
        ]
        
        for i, (label, btn_minus, btn_plus) in enumerate(joints):
            frame = tk.Frame(joint_frame)
            frame.pack(side=tk.LEFT, padx=5, pady=5)
            
            tk.Label(frame, text=label).pack()
            
            # - 버튼
            btn = tk.Button(frame, text=btn_minus, width=8)
            btn_name = f"{label} -"
            self.button_states[btn_name] = False
            btn.bind('<ButtonPress-1>', lambda e, name=btn_name: self._on_button_press(name))
            btn.bind('<ButtonRelease-1>', lambda e, name=btn_name: self._on_button_release(name))
            btn.pack(side=tk.LEFT)
            
            # + 버튼
            btn = tk.Button(frame, text=btn_plus, width=8)
            btn_name = f"{label} +"
            self.button_states[btn_name] = False
            btn.bind('<ButtonPress-1>', lambda e, name=btn_name: self._on_button_press(name))
            btn.bind('<ButtonRelease-1>', lambda e, name=btn_name: self._on_button_release(name))
            btn.pack(side=tk.LEFT)
    
    def _build_cartesian_frame(self):
        cart_frame = tk.LabelFrame(self.root, text="Cartesian Control", font=("Arial", 12))
        cart_frame.pack(pady=10, padx=10, fill="both")
        
        # Translation 버튼들
        trans_frame = tk.Frame(cart_frame)
        trans_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(trans_frame, text="Translation", font=("Arial", 10, "bold")).pack()
        
        trans_buttons = [
            ("X+"), ("X-"),
            ("Y-"), ("Y+"),
            ("Z+"), ("Z-"),
        ]
        
        for label in trans_buttons:
            btn = tk.Button(trans_frame, text=f"{label}", width=10)
            self.button_states[label] = False
            btn.bind('<ButtonPress-1>', lambda e, name=label: self._on_button_press(name))
            btn.bind('<ButtonRelease-1>', lambda e, name=label: self._on_button_release(name))
            btn.pack()
        
        # Rotation 버튼들
        rot_frame = tk.Frame(cart_frame)
        rot_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(rot_frame, text="Rotation", font=("Arial", 10, "bold")).pack()
        
        rot_buttons = [
            ("Roll+"), ("Roll-"),
            ("Pitch-"), ("Pitch+"),
            ("Yaw+"), ("Yaw-"),
        ]
        
        for label in rot_buttons:
            btn = tk.Button(rot_frame, text=f"{label}", width=10)
            self.button_states[label] = False
            btn.bind('<ButtonPress-1>', lambda e, name=label: self._on_button_press(name))
            btn.bind('<ButtonRelease-1>', lambda e, name=label: self._on_button_release(name))
            btn.pack()
    
    def _build_pose_frame(self):
        pose_frame = tk.LabelFrame(self.root, text="Predefined Poses", font=("Arial", 12))
        pose_frame.pack(pady=10, padx=10, fill="both")
        
        color_list = ["lightblue", "lightgreen", "lightyellow", "lightpink", "lightgray"]
        
        for idx, label in enumerate(self.predefined_poses.keys()):
            color = color_list[idx % len(color_list)]  # 색상 리스트보다 포즈가 많으면 반복
            btn = tk.Button(pose_frame, text=label, width=15, bg=color)
            btn_name = f"{label}"
            self.button_states[btn_name] = False
            btn.bind('<ButtonPress-1>', lambda e, name=btn_name: self._on_button_press(name))
            btn.bind('<ButtonRelease-1>', lambda e, name=btn_name: self._on_button_release(name))
            btn.pack(side=tk.LEFT, padx=5)
    
    def _build_hand_grasp_frame(self):
        """Hand grasp trajectory control frame"""
        hand_frame = tk.LabelFrame(self.root, text="Hand Grasp Control", font=("Arial", 12))
        hand_frame.pack(pady=10, padx=10, fill="both")
        
        # Progress label
        self.hand_progress_label = tk.Label(hand_frame, 
                                           text=f"Progress: {self.hand_grasp_progress:.2f}", 
                                           font=("Arial", 10))
        self.hand_progress_label.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(hand_frame)
        button_frame.pack(pady=5)
        
        # Start -> Grasp button
        btn_to_grasp = tk.Button(button_frame, text="Hand Grasp →", width=20, 
                                 bg="lightcoral", font=("Arial", 10, "bold"))
        btn_name_to_grasp = "hand_grasp_forward"
        self.button_states[btn_name_to_grasp] = False
        btn_to_grasp.bind('<ButtonPress-1>', 
                         lambda e, name=btn_name_to_grasp: self._on_button_press(name))
        btn_to_grasp.bind('<ButtonRelease-1>', 
                         lambda e, name=btn_name_to_grasp: self._on_button_release(name))
        btn_to_grasp.pack(side=tk.LEFT, padx=10)
        
        # Grasp -> Start button
        btn_to_start = tk.Button(button_frame, text="← Hand Release", width=20, 
                                bg="lightseagreen", font=("Arial", 10, "bold"))
        btn_name_to_start = "hand_grasp_backward"
        self.button_states[btn_name_to_start] = False
        btn_to_start.bind('<ButtonPress-1>', 
                         lambda e, name=btn_name_to_start: self._on_button_press(name))
        btn_to_start.bind('<ButtonRelease-1>', 
                         lambda e, name=btn_name_to_start: self._on_button_release(name))
        btn_to_start.pack(side=tk.LEFT, padx=10)
    
    def _on_button_press(self, button_name):
        self.button_states[button_name] = True
        print(f"{button_name} pressed")
    
    def _on_button_release(self, button_name):
        self.button_states[button_name] = False
        print(f"{button_name} released")
    
    def _toggle_mode(self):
        self.control_mode = 'cartesian' if self.control_mode == 'joint' else 'joint'
        self.mode_label.config(text=f"Mode: {self.control_mode.upper()}")
        print(f">>> Control Mode: {self.control_mode.upper()}")
    
    def _handle_hand_grasp_control(self, pressed_buttons):
        """Handle hand grasp trajectory execution"""
        if self.predefined_hand_traj is None:
            return
        
        start_pose = self.predefined_hand_traj['start']
        grasp_pose = self.predefined_hand_traj['grasp']
        
        # Forward direction (start -> grasp)
        if 'hand_grasp_forward' in pressed_buttons:
            if self.hand_grasp_progress < 1.0:
                self.hand_grasp_progress = min(1.0, self.hand_grasp_progress + self.hand_grasp_speed)
                # Linear interpolation
                current_pose = start_pose + (grasp_pose - start_pose) * self.hand_grasp_progress
                self.hand.move(current_pose)
                self.hand_progress_label.config(text=f"Progress: {self.hand_grasp_progress:.2f}")
        
        # Backward direction (grasp -> start)
        elif 'hand_grasp_backward' in pressed_buttons:
            if self.hand_grasp_progress > 0.0:
                self.hand_grasp_progress = max(0.0, self.hand_grasp_progress - self.hand_grasp_speed)
                # Linear interpolation
                current_pose = start_pose + (grasp_pose - start_pose) * self.hand_grasp_progress
                self.hand.move(current_pose)
                self.hand_progress_label.config(text=f"Progress: {self.hand_grasp_progress:.2f}")
    
    def _handle_pose_control(self, pressed_buttons):
        """ Move to predefined poses """
        current_pose = self.robot.get_data()['position'].copy()  # 4x4 matrix
        
        for button in pressed_buttons:
            target_pose = self.predefined_poses[button].copy()
            print(target_pose, current_pose) 
            # Translation delta
            t_delta = target_pose[:3, 3] - current_pose[:3, 3]
            t_distance = np.linalg.norm(t_delta)
            
            # Rotation delta
            current_rot = Rotation.from_matrix(current_pose[:3, :3])
            target_rot = Rotation.from_matrix(target_pose[:3, :3])
            delta_rot = target_rot * current_rot.inv()
            r_delta = delta_rot.as_rotvec()
            r_distance = np.linalg.norm(r_delta)
            
            # 충분히 가까우면 스킵
            if t_distance < 0.001 and r_distance < 0.01:  # 1mm, 0.01 rad
                continue
            
            # 새로운 pose 계산
            new_pose = current_pose.copy()
            
            # Translation 보간
            if t_distance > 0.001:
                t_step_size = min(self.cart_delta / 1000, t_distance)  # cart_delta는 mm 단위
                new_pose[:3, 3] = current_pose[:3, 3] + (t_delta / t_distance) * t_step_size
            
            # Rotation 보간
            if r_distance > 0.01:
                r_step_size = min(self.angle_delta, r_distance)
                partial_rot = Rotation.from_rotvec((r_delta / r_distance) * r_step_size)
                new_rot = partial_rot * current_rot
                new_pose[:3, :3] = new_rot.as_matrix()
            
            self.robot.move(new_pose, is_servo=True)
        
    
    def _handle_joint_control(self, pressed_buttons):
        current_qpos = self.robot.get_data()['qpos'].copy()
        delta = np.zeros(6)
        
        for button in pressed_buttons:
            if "Joint 0 -" in button:
                delta[0] = -self.joint_delta
            elif "Joint 0 +" in button:
                delta[0] = self.joint_delta
            elif "Joint 1 -" in button:
                delta[1] = -self.joint_delta
            elif "Joint 1 +" in button:
                delta[1] = self.joint_delta
            elif "Joint 2 -" in button:
                delta[2] = -self.joint_delta
            elif "Joint 2 +" in button:
                delta[2] = self.joint_delta
            elif "Joint 3 -" in button:
                delta[3] = -self.joint_delta
            elif "Joint 3 +" in button:
                delta[3] = self.joint_delta
            elif "Joint 4 -" in button:
                delta[4] = -self.joint_delta
            elif "Joint 4 +" in button:
                delta[4] = self.joint_delta
            elif "Joint 5 -" in button:
                delta[5] = -self.joint_delta
            elif "Joint 5 +" in button:
                delta[5] = self.joint_delta
        
        if np.any(delta != 0):
            new_qpos = current_qpos + delta
            self.robot.move(new_qpos, is_servo=True)
    
    def _handle_cartesian_control(self, pressed_buttons):
        current_pose = self.robot.get_data()['position'].copy()
        
        t_delta = np.zeros(3)
        r_delta = np.zeros(3)
        
        for button in pressed_buttons:
            # Translation
            if "X+" in button:
                t_delta[0] = self.cart_delta
            elif "X-" in button:
                t_delta[0] = -self.cart_delta
            elif "Y-" in button:
                t_delta[1] = -self.cart_delta
            elif "Y+" in button:
                t_delta[1] = self.cart_delta
            elif "Z+" in button:
                t_delta[2] = self.cart_delta
            elif "Z-" in button:
                t_delta[2] = -self.cart_delta
            # Rotation
            elif "Roll+" in button:
                r_delta[0] = self.angle_delta
            elif "Roll-" in button:
                r_delta[0] = -self.angle_delta
            elif "Pitch-" in button:
                r_delta[1] = -self.angle_delta
            elif "Pitch+" in button:
                r_delta[1] = self.angle_delta
            elif "Yaw+" in button:
                r_delta[2] = self.angle_delta
            elif "Yaw-" in button:
                r_delta[2] = -self.angle_delta
        
        if np.any(t_delta != 0) or np.any(r_delta != 0):
            current_pose[:3, 3] += t_delta / 1000
            
            current_rot = Rotation.from_matrix(current_pose[:3, :3])
            delta_rot = Rotation.from_euler('xyz', r_delta)
            new_rot = current_rot * delta_rot
            current_pose[:3, :3] = new_rot.as_matrix()
            
            self.robot.move(current_pose, is_servo=True)
    
    def _control_loop(self):
        """눌린 버튼 상태를 계속 확인하는 루프"""
        while self.running:
            pressed_buttons = [name for name, state in self.button_states.items() if state]
            
            if pressed_buttons:
                # Hand grasp 버튼 체크 (우선순위 1)
                hand_grasp_pressed = [name for name in pressed_buttons 
                                     if 'hand_grasp' in name]
                if hand_grasp_pressed:
                    self._handle_hand_grasp_control(hand_grasp_pressed)
                # Pose 버튼 체크 (우선순위 2)
                else:
                    pose_pressed = [name for name in pressed_buttons if name in self.predefined_poses]
                    if pose_pressed:
                        self._handle_pose_control(pose_pressed)
                    elif self.control_mode == 'joint':
                        self._handle_joint_control(pressed_buttons)
                    else:
                        self._handle_cartesian_control(pressed_buttons)
            
            time.sleep(0.01)  # 100Hz
    
    def _on_exit(self):
        print("종료 중...")
        self.running = False
        try:
            self.robot.end(set_break=True)
            self.hand.end()
        except:
            pass
        self.root.destroy()
    
    def run(self):
        """GUI 메인 루프 실행"""
        self.root.mainloop()