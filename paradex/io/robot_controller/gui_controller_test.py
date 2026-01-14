import tkinter as tk
import time
import numpy as np
from threading import Thread
from scipy.spatial.transform import Rotation

class RobotGUIController:
    def __init__(self, robot_controller, hand_controller, predefined_poses=None, grasp_pose=None, 
             approach_traj=None, lift_distance=100.0, place_distance=40.0):
        """
        Args:
            robot_controller: XArmController 또는 유사한 인터페이스를 가진 로봇 컨트롤러
            hand_controller: Hand controller
            predefined_poses: dict, 미리 정의된 포즈들 (선택사항, 'base' 포함 필요)
            grasp_pose: dict with 'start', 'pregrasp', 'grasp', 'squeezed' keys containing hand poses
            approach_traj: numpy array, approach trajectory [N, 22] (xarm 6 + hand 16)
            lift_distance: float, lift distance in mm (default: 100mm)
            place_distance: float, place down distance in mm (default: 40mm)
        """
        print(">>> Initializing Robot GUI Controller...")
        self.predefined_poses = predefined_poses
        self.predefined_hand_traj = grasp_pose
        self.approach_traj = approach_traj
        self.lift_distance = lift_distance
        self.place_distance = place_distance
        
        self.robot = robot_controller
        self.hand = hand_controller
        self.target_hand_qpos = None
        
        # Control parameters
        self.joint_delta = 0.03
        self.hand_joint_delta = 0.001
        self.cart_delta = 1.0
        self.angle_delta = 0.01
        
        # Progress trackers
        self.approach_progress = 0.0
        self.hand_grasp_speed = 0.005
        self.hand_grasp_progress = 0.0
        
        # Height trackers
        self.lift_start_z = None
        self.place_start_z = None
        
        # State machines
        self.place_state = 0
        self.release_state = 0
        
        # Button states
        self.button_states = {}
        self.control_mode = 'cartesian'
        self.running = True
        
        # Control loop start
        print(">>> Starting control thread...")
        self.control_thread = Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        print(">>> Control thread started.")
        
        # GUI
        print(">>> Creating GUI...")
        self.root = tk.Tk()
        self._build_gui()
        print(">>> GUI created.")
    
    def _build_gui(self):
        self.root.title("Robot GUI Controller")
        self.root.geometry("1200x1200")
        
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
        
        # Approach, Grasp, Lift Control
        self._build_task_frame()
        
        # Hand Grasp Control
        self._build_hand_grasp_frame()
        
        # Hand Joint Control (개별 관절 제어)
        self._build_hand_joint_frame()
        
        # Place/Release Control
        self._build_place_release_frame()
        
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
            color = color_list[idx % len(color_list)]
            btn = tk.Button(pose_frame, text=label, width=15, bg=color)
            btn_name = f"{label}"
            self.button_states[btn_name] = False
            btn.bind('<ButtonPress-1>', lambda e, name=btn_name: self._on_button_press(name))
            btn.bind('<ButtonRelease-1>', lambda e, name=btn_name: self._on_button_release(name))
            btn.pack(side=tk.LEFT, padx=5)
    
    def _build_task_frame(self):
        """Approach, Grasp, Lift 버튼 프레임"""
        task_frame = tk.LabelFrame(self.root, text="Task Execution", font=("Arial", 12))
        task_frame.pack(pady=10, padx=10, fill="both")
        
        # Approach 버튼
        approach_btn = tk.Button(task_frame, text="Approach", width=15, bg="skyblue", 
                                font=("Arial", 10, "bold"))
        btn_name = "approach"
        self.button_states[btn_name] = False
        approach_btn.bind('<ButtonPress-1>', lambda e, name=btn_name: self._on_button_press(name))
        approach_btn.bind('<ButtonRelease-1>', lambda e, name=btn_name: self._on_button_release(name))
        approach_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Grasp 버튼
        grasp_btn = tk.Button(task_frame, text="Grasp", width=15, bg="lightcoral", 
                             font=("Arial", 10, "bold"))
        btn_name = "grasp"
        self.button_states[btn_name] = False
        grasp_btn.bind('<ButtonPress-1>', lambda e, name=btn_name: self._on_button_press(name))
        grasp_btn.bind('<ButtonRelease-1>', lambda e, name=btn_name: self._on_button_release(name))
        grasp_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Lift 버튼
        lift_btn = tk.Button(task_frame, text="Lift", width=15, bg="lightgreen", 
                            font=("Arial", 10, "bold"))
        btn_name = "lift"
        self.button_states[btn_name] = False
        lift_btn.bind('<ButtonPress-1>', lambda e, name=btn_name: self._on_button_press(name))
        lift_btn.bind('<ButtonRelease-1>', lambda e, name=btn_name: self._on_button_release(name))
        lift_btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    def _build_hand_grasp_frame(self):
        """Hand grasp trajectory control frame with 4 stages"""
        hand_frame = tk.LabelFrame(self.root, text="Hand Grasp Control", font=("Arial", 12))
        hand_frame.pack(pady=10, padx=10, fill="both")
        
        # Progress label with stage info
        self.hand_progress_label = tk.Label(hand_frame, text=self._get_progress_text(), 
                                           font=("Arial", 10))
        self.hand_progress_label.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(hand_frame)
        button_frame.pack(pady=5)
        
        # Start -> Squeezed button
        btn_to_grasp = tk.Button(button_frame, text="Hand Grasp →", width=20, bg="lightcoral", 
                                font=("Arial", 10, "bold"))
        btn_name_to_grasp = "hand_grasp_forward"
        self.button_states[btn_name_to_grasp] = False
        btn_to_grasp.bind('<ButtonPress-1>', lambda e, name=btn_name_to_grasp: self._on_button_press(name))
        btn_to_grasp.bind('<ButtonRelease-1>', lambda e, name=btn_name_to_grasp: self._on_button_release(name))
        btn_to_grasp.pack(side=tk.LEFT, padx=10)
        
        # Squeezed -> Start button
        btn_to_start = tk.Button(button_frame, text="← Hand Release", width=20, bg="lightseagreen", 
                                font=("Arial", 10, "bold"))
        btn_name_to_start = "hand_grasp_backward"
        self.button_states[btn_name_to_start] = False
        btn_to_start.bind('<ButtonPress-1>', lambda e, name=btn_name_to_start: self._on_button_press(name))
        btn_to_start.bind('<ButtonRelease-1>', lambda e, name=btn_name_to_start: self._on_button_release(name))
        btn_to_start.pack(side=tk.LEFT, padx=10)
    
    def _build_hand_joint_frame(self):
        """Allegro hand 각 joint 를 개별적으로 제어하는 버튼 프레임"""
        hand_joint_frame = tk.LabelFrame(self.root, text="Hand Joint Control", font=("Arial", 12))
        hand_joint_frame.pack(pady=10, padx=10, fill="both")
        
        # 16개 joint 를 4x4 그리드로 배치
        joints_per_row = 4
        for idx in range(16):
            row = idx // joints_per_row
            col = idx % joints_per_row
            
            frame = tk.Frame(hand_joint_frame)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="w")
            
            tk.Label(frame, text=f"H{idx}").pack()
            
            # - 버튼
            btn_minus = tk.Button(frame, text="-", width=4)
            btn_name_minus = f"Hand {idx} -"
            self.button_states[btn_name_minus] = False
            btn_minus.bind('<ButtonPress-1>', lambda e, name=btn_name_minus: self._on_button_press(name))
            btn_minus.bind('<ButtonRelease-1>', lambda e, name=btn_name_minus: self._on_button_release(name))
            btn_minus.pack(side=tk.LEFT)
            
            # + 버튼
            btn_plus = tk.Button(frame, text="+", width=4)
            btn_name_plus = f"Hand {idx} +"
            self.button_states[btn_name_plus] = False
            btn_plus.bind('<ButtonPress-1>', lambda e, name=btn_name_plus: self._on_button_press(name))
            btn_plus.bind('<ButtonRelease-1>', lambda e, name=btn_name_plus: self._on_button_release(name))
            btn_plus.pack(side=tk.LEFT)
    
    def _build_place_release_frame(self):
        """Place/Release sequence 버튼 프레임"""
        frame = tk.LabelFrame(self.root, text="Place & Release", font=("Arial", 12))
        frame.pack(pady=10, padx=10, fill="both")
        
        # Place 버튼 (성공 시)
        place_btn = tk.Button(frame, text="Place (Success)", width=20, bg="gold", 
                             font=("Arial", 10, "bold"))
        btn_name = "place"
        self.button_states[btn_name] = False
        place_btn.bind('<ButtonPress-1>', lambda e, name=btn_name: self._on_button_press(name))
        place_btn.bind('<ButtonRelease-1>', lambda e, name=btn_name: self._on_button_release(name))
        place_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Release 버튼 (실패 시)
        release_btn = tk.Button(frame, text="Release (Fail)", width=20, bg="orange", 
                               font=("Arial", 10, "bold"))
        btn_name = "release"
        self.button_states[btn_name] = False
        release_btn.bind('<ButtonPress-1>', lambda e, name=btn_name: self._on_button_press(name))
        release_btn.bind('<ButtonRelease-1>', lambda e, name=btn_name: self._on_button_release(name))
        release_btn.pack(side=tk.LEFT, padx=10, pady=5)
    
    def _get_progress_text(self):
        """Get progress text with current stage"""
        progress = self.hand_grasp_progress
        if progress < 1.0:
            stage = "Start → Pregrasp"
            stage_progress = progress
        elif progress < 2.0:
            stage = "Pregrasp → Grasp"
            stage_progress = progress - 1.0
        elif progress < 3.0:
            stage = "Grasp → Squeezed"
            stage_progress = progress - 2.0
        else:
            stage = "Squeezed"
            stage_progress = 1.0
        
        return f"Progress: {progress:.2f}/3.0 | Stage: {stage} ({stage_progress:.2f})"
    
    def _on_button_press(self, button_name):
        self.button_states[button_name] = True
        print(f"{button_name} pressed")
    
    def _on_button_release(self, button_name):
        self.button_states[button_name] = False
        
    
    def _toggle_mode(self):
        self.control_mode = 'cartesian' if self.control_mode == 'joint' else 'joint'
        self.mode_label.config(text=f"Mode: {self.control_mode.upper()}")
        print(f">>> Control Mode: {self.control_mode.upper()}")
    
    def _execute_approach(self):
        """Approach trajectory 실행"""
        if self.approach_traj is None:
            print("No approach trajectory loaded!")
            return
        
        # 이미 trajectory 완료
        if self.approach_progress >= len(self.approach_traj):
            print("Approach already complete!")
            return
        
        current_qpos = self.robot.get_data()['qpos'].copy()
        current_hand_qpos = self.hand.get_data()['qpos'].copy()
        current_full_qpos = np.concatenate([current_qpos, current_hand_qpos])
        
        # 현재 목표 waypoint
        target_idx = int(self.approach_progress)
        target_qpos = self.approach_traj[target_idx]
        self.hand.move(target_qpos[6:])
        
        # Calculate distance
        distance = np.linalg.norm(current_qpos - target_qpos[:6])
        
        if distance > 0.01:  # Not reached yet
            delta = target_qpos[:6] - current_qpos
            
            # Normalize and limit velocity
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 0:
                delta = delta / delta_norm * min(self.joint_delta, delta_norm)
            
            new_qpos = current_qpos + delta
            self.robot.move(new_qpos, is_servo=True)
        else:
            # Reached current waypoint, move to next
            self.approach_progress += 1
            if self.approach_progress >= len(self.approach_traj):
                print("Approach trajectory complete!")
            else:
                print(f"Waypoint {target_idx} reached, moving to {int(self.approach_progress)}")

    def _execute_grasp(self):
        """Grasp sequence 실행 (누르고 있는 동안 자동 진행)"""
        if self.predefined_hand_traj is None:
            print("No grasp trajectory loaded!")
            return
        
        start_pose = self.predefined_hand_traj['start']
        pregrasp_pose = self.predefined_hand_traj['pregrasp']
        grasp_pose = self.predefined_hand_traj['grasp']
        squeezed_pose = self.predefined_hand_traj['squeezed']
        
        if self.hand_grasp_progress < 3.0:
            self.hand_grasp_progress = min(3.0, self.hand_grasp_progress + self.hand_grasp_speed)
            
            # Determine current stage and interpolate
            progress = self.hand_grasp_progress
            if progress <= 1.0:
                # Stage 1: start -> pregrasp
                current_pose = start_pose + (pregrasp_pose - start_pose) * progress
            elif progress <= 2.0:
                # Stage 2: pregrasp -> grasp
                stage_progress = progress - 1.0
                current_pose = pregrasp_pose + (grasp_pose - pregrasp_pose) * stage_progress
            else:
                # Stage 3: grasp -> squeezed
                stage_progress = progress - 2.0
                current_pose = grasp_pose + (squeezed_pose - grasp_pose) * stage_progress
            
            self.hand.move(current_pose)
            self.hand_progress_label.config(text=self._get_progress_text())
        else:
            print("Grasp sequence complete!")
    
    def _execute_lift(self):
        """Lift 실행 - lift_distance만큼만 올라감"""
        current_pose = self.robot.get_data()['position'].copy()
        
        # 처음 버튼 누르면 시작 높이 저장
        if self.lift_start_z is None:
            self.lift_start_z = current_pose[2, 3]
        
        # 목표 높이
        target_z = self.lift_start_z + self.lift_distance / 1000.0
        
        # 아직 목표에 도달 안 했으면 계속 올라감
        if current_pose[2, 3] < target_z - 0.001:  # 1mm threshold
            new_pose = current_pose.copy()
            new_pose[2, 3] += self.cart_delta / 1000.0
            self.robot.move(new_pose, is_servo=True)
        else:
            print("Lift complete!")
    
    def _execute_place(self):
        """Place sequence 실행 (성공 시): Z- place_distance → hand release → base pose"""
        if 'base' not in self.predefined_poses:
            print("No 'base' pose defined!")
            return
        
        if self.predefined_hand_traj is None:
            print("No hand trajectory loaded!")
            return
        
        current_pose = self.robot.get_data()['position'].copy()
        
        # State 0: Lower down by place_distance
        if self.place_state == 0:
            if self.place_start_z is None:
                self.place_start_z = current_pose[2, 3]
            
            target_z = self.place_start_z - self.place_distance / 1000.0
            
            if current_pose[2, 3] > target_z + 0.001:  # Still lowering
                new_pose = current_pose.copy()
                new_pose[2, 3] -= self.cart_delta / 1000.0
                self.robot.move(new_pose, is_servo=True)
            else:
                print("Place: Lowering complete")
                self.place_state = 1
        
        # State 1: Release hand
        elif self.place_state == 1:
            start_pose = self.predefined_hand_traj['start']
            self.hand.move(start_pose)
            reached = self._move_to_pose(self.predefined_poses['base'])
            if reached:
                print("Place: Complete!")
                self.place_state = 2

    def _execute_release(self):
        """Release sequence 실행 (실패 시): hand release → base pose"""
        if 'base' not in self.predefined_poses:
            print("No 'base' pose defined!")
            return
        
        if self.predefined_hand_traj is None:
            print("No hand trajectory loaded!")
            return
        
        if self.release_state == 0: 
            start_pose = self.predefined_hand_traj['start']
            self.hand.move(start_pose)
            
            reached = self._move_to_pose(self.predefined_poses['base'])
            if reached:
                self.release_state = 1
                print("Release: Complete!")
            
    
    def _move_to_pose(self, target_pose):
        """Helper function to move to a target pose"""
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
    
    def _handle_hand_grasp_control(self, pressed_buttons):
        """Handle hand grasp trajectory execution with 4 stages"""
        if self.predefined_hand_traj is None:
            return
        
        start_pose = self.predefined_hand_traj['start']
        pregrasp_pose = self.predefined_hand_traj['pregrasp']
        grasp_pose = self.predefined_hand_traj['grasp']
        squeezed_pose = self.predefined_hand_traj['squeezed']
        
        # Forward direction (start -> pregrasp -> grasp -> squeezed)
        if 'hand_grasp_forward' in pressed_buttons:
            if self.hand_grasp_progress < 3.0:
                self.hand_grasp_progress = min(3.0, self.hand_grasp_progress + self.hand_grasp_speed)
                
                # Determine current stage and interpolate
                progress = self.hand_grasp_progress
                if progress <= 1.0:
                    # Stage 1: start -> pregrasp
                    current_pose = start_pose + (pregrasp_pose - start_pose) * progress
                elif progress <= 2.0:
                    # Stage 2: pregrasp -> grasp
                    stage_progress = progress - 1.0
                    current_pose = pregrasp_pose + (grasp_pose - pregrasp_pose) * stage_progress
                else:
                    # Stage 3: grasp -> squeezed
                    stage_progress = progress - 2.0
                    current_pose = grasp_pose + (squeezed_pose - grasp_pose) * stage_progress
                
                self.hand.move(current_pose)
                self.hand_progress_label.config(text=self._get_progress_text())
        
        # Backward direction (squeezed -> grasp -> pregrasp -> start)
        elif 'hand_grasp_backward' in pressed_buttons:
            if self.hand_grasp_progress > 0.0:
                self.hand_grasp_progress = max(0.0, self.hand_grasp_progress - self.hand_grasp_speed)
                
                # Determine current stage and interpolate
                progress = self.hand_grasp_progress
                if progress <= 1.0:
                    # Stage 1: start -> pregrasp
                    current_pose = start_pose + (pregrasp_pose - start_pose) * progress
                elif progress <= 2.0:
                    # Stage 2: pregrasp -> grasp
                    stage_progress = progress - 1.0
                    current_pose = pregrasp_pose + (grasp_pose - pregrasp_pose) * stage_progress
                else:
                    # Stage 3: grasp -> squeezed
                    stage_progress = progress - 2.0
                    current_pose = grasp_pose + (squeezed_pose - grasp_pose) * stage_progress
                
                self.hand.move(current_pose)
                self.hand_progress_label.config(text=self._get_progress_text())
    
    def _handle_pose_control(self, pressed_buttons):
        """Move to predefined poses"""
        current_pose = self.robot.get_data()['position'].copy()  # 4x4 matrix
        
        for button in pressed_buttons:
            target_pose = self.predefined_poses[button].copy()
            
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
                t_step_size = min(self.cart_delta / 1000, t_distance)
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
    
    def _handle_hand_joint_control(self, pressed_buttons):
        if self.target_hand_qpos is None:
            self.target_hand_qpos = self.approach_traj[0][6:]
    
        """Allegro hand joint qpos 를 개별적으로 제어"""
        delta = np.zeros_like(self.target_hand_qpos)  # hand joint 부분만
        
        for button in pressed_buttons:
            # 버튼 이름 형식: "Hand {idx} -" 또는 "Hand {idx} +"
            if not button.startswith("Hand "):
                continue
            try:
                parts = button.split()
                idx = int(parts[1])
                sign = parts[2]
            except Exception:
                continue
            
            if 0 <= idx < len(delta):
                if sign == "-":
                    print(f"Hand joint {idx} - pressed")
                    delta[idx] = -self.hand_joint_delta
                elif sign == "+":
                    print(f"Hand joint {idx} + pressed")    
                    delta[idx] = self.hand_joint_delta
        
        if np.any(delta != 0):
            self.target_hand_qpos = self.target_hand_qpos + delta
            self.hand.move(self.target_hand_qpos)
            print(f"Moving hand joints to: {self.target_hand_qpos}")
            
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
        iteration = 0
        
        while self.running:
            iteration += 1
            
            pressed_buttons = [name for name, state in self.button_states.items() if state]
            
            if pressed_buttons:
                # Place sequence (최우선)
                if 'place' in pressed_buttons:
                    self._execute_place()
                # Release sequence
                elif 'release' in pressed_buttons:
                    self._execute_release()
                # Approach 버튼 체크
                elif 'approach' in pressed_buttons:
                    self._execute_approach()
                # Grasp 버튼 체크
                elif 'grasp' in pressed_buttons:
                    self._execute_grasp()
                # Lift 버튼 체크
                elif 'lift' in pressed_buttons:
                    self._execute_lift()
                # Hand grasp 버튼 체크
                elif any('hand_grasp' in name for name in pressed_buttons):
                    hand_grasp_pressed = [name for name in pressed_buttons if 'hand_grasp' in name]
                    self._handle_hand_grasp_control(hand_grasp_pressed)
                # Hand joint 버튼 체크 (알레그로 개별 관절)
                elif any(name.startswith("Hand ") for name in pressed_buttons):
                    hand_joint_pressed = [name for name in pressed_buttons if name.startswith("Hand ")]
                    self._handle_hand_joint_control(hand_joint_pressed)
                # Pose 버튼 체크
                else:
                    pose_pressed = [name for name in pressed_buttons if name in self.predefined_poses]
                    if pose_pressed:
                        self._handle_pose_control(pose_pressed)
                    elif self.control_mode == 'joint':
                        self._handle_joint_control(pressed_buttons)
                    else:
                        self._handle_cartesian_control(pressed_buttons)
            
            time.sleep(0.01)  # 100Hz
        
        print(">>> Control loop exited")
    
    def _on_exit(self):
        print("종료 중...")
        self.running = False
        try:
            self.robot.end()
            self.hand.end()
        except:
            pass
        self.root.destroy()
    
    def run(self):
        """GUI 메인 루프 실행"""
        self.root.mainloop()