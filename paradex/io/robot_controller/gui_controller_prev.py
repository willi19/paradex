import tkinter as tk
import time
import numpy as np
from threading import Thread
from scipy.spatial.transform import Rotation


class RobotGUIController:
    def __init__(self, robot_controller, hand_controller, predefined_traj=None):
        """
        Args:
            robot_controller: XArmController 또는 유사한 인터페이스를 가진 로봇 컨트롤러
            hand_controller: Hand controller
            predefined_traj: dict of trajectories, each containing:
                - 'arm': np.array of shape (N, 4, 4) for arm end effector poses
                - 'hand': np.array of shape (N, 16) for hand joint positions
        """
        
        self.predefined_traj = predefined_traj if predefined_traj is not None else {}
        self.robot = robot_controller
        self.hand = hand_controller
        
        # Control parameters
        self.joint_delta = 0.01  # 관절 각도 변화량 (radian)
        self.cart_delta = 1.0    # cartesian 위치 변화량 (mm)
        self.angle_delta = 0.01  # orientation 변화량 (radian)
        
        # Trajectory playback parameters
        self.traj_speed = 0.3  # trajectory playback speed multiplier
        self.traj_progress = {}  # progress for each trajectory
        
        # Initialize progress for all trajectories
        for traj_name in self.predefined_traj.keys():
            self.traj_progress[traj_name] = 0.0
        
        # Trajectory state for each trajectory
        self.traj_state = {}  # "idle", "moving_to_start", or "playing"
        self.move_to_start_progress = {}
        self.start_arm_pose = {}  # Store arm pose when button is first pressed
        self.start_hand_pose = {}  # Store hand pose when button is first pressed
        
        for traj_name in self.predefined_traj.keys():
            self.traj_state[traj_name] = "idle"
            self.move_to_start_progress[traj_name] = 0.0
            self.start_arm_pose[traj_name] = None
            self.start_hand_pose[traj_name] = None
        
        self.move_to_start_speed = 0.005  # Speed for moving to start position
        
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
        
        # Trajectory Playback Control
        self._build_trajectory_frame()
        
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
    
    def _build_trajectory_frame(self):
        """Trajectory playback control frame"""
        traj_frame = tk.LabelFrame(self.root, text="Trajectory Playback (Hold to Play)", font=("Arial", 12))
        traj_frame.pack(pady=10, padx=10, fill="both")
        
        # Trajectory buttons with progress labels
        color_list = ["lightblue", "lightgreen", "lightyellow", "lightpink", "lightcoral"]
        
        self.traj_progress_labels = {}
        self.traj_state_labels = {}
        
        for idx, traj_name in enumerate(self.predefined_traj.keys()):
            # Container for each trajectory
            container = tk.Frame(traj_frame)
            container.pack(pady=5, padx=10, fill="x")
            
            # Button
            color = color_list[idx % len(color_list)]
            btn = tk.Button(container, text=f"▶ {traj_name}", width=20, bg=color,
                          font=("Arial", 10, "bold"))
            btn_name = f"traj_{traj_name}"
            self.button_states[btn_name] = False
            btn.bind('<ButtonPress-1>', lambda e, name=btn_name: self._on_button_press(name))
            btn.bind('<ButtonRelease-1>', lambda e, name=btn_name: self._on_button_release(name))
            btn.pack(side=tk.LEFT, padx=5)
            
            # State label
            state_label = tk.Label(container, 
                                  text=self._get_traj_state_text(traj_name),
                                  font=("Arial", 9), width=20)
            state_label.pack(side=tk.LEFT, padx=5)
            self.traj_state_labels[traj_name] = state_label
            
            # Progress label
            progress_label = tk.Label(container, 
                                     text=self._get_traj_progress_text(traj_name),
                                     font=("Arial", 9))
            progress_label.pack(side=tk.LEFT, padx=10)
            self.traj_progress_labels[traj_name] = progress_label
        
        # Control buttons
        control_frame = tk.Frame(traj_frame)
        control_frame.pack(pady=10)
        
        # Reset all button
        tk.Button(control_frame, text="Reset All Trajectories", width=25,
                 command=self._reset_all_trajectories, bg="lightyellow",
                 font=("Arial", 10, "bold")).pack()
    
    def _reset_all_trajectories(self):
        """Reset all trajectory progress to start"""
        for traj_name in self.traj_progress.keys():
            self.traj_progress[traj_name] = 0.0
            self.traj_state[traj_name] = "idle"
            self.move_to_start_progress[traj_name] = 0.0
            self.traj_progress_labels[traj_name].config(text=self._get_traj_progress_text(traj_name))
            self.traj_state_labels[traj_name].config(text=self._get_traj_state_text(traj_name))
        print("All trajectories reset to start")
    
    def _get_traj_state_text(self, traj_name):
        """Get trajectory state text"""
        state = self.traj_state.get(traj_name, "idle")
        if state == "idle":
            return "State: IDLE"
        elif state == "moving_to_start":
            progress = self.move_to_start_progress.get(traj_name, 0.0)
            return f"State: MOVING ({progress*100:.0f}%)"
        elif state == "playing":
            return "State: PLAYING"
        return "State: UNKNOWN"
    
    def _get_traj_progress_text(self, traj_name):
        """Get trajectory progress text"""
        if traj_name not in self.predefined_traj:
            return "N/A"
        
        traj = self.predefined_traj[traj_name]
        traj_length = len(traj['arm'])
        progress = self.traj_progress.get(traj_name, 0.0)
        current_idx = int(progress)
        
        return f"Frame: {current_idx}/{traj_length-1} ({progress/max(1, traj_length-1)*100:.1f}%)"
    
    def _interpolate_pose(self, pose1, pose2, alpha):
        """Interpolate between two 4x4 poses
        Args:
            pose1: 4x4 numpy array
            pose2: 4x4 numpy array
            alpha: interpolation factor (0 to 1)
        Returns:
            4x4 numpy array
        """
        result = np.eye(4)
        
        # Linear interpolation for translation
        result[:3, 3] = pose1[:3, 3] * (1 - alpha) + pose2[:3, 3] * alpha
        
        # Slerp for rotation
        rot1 = Rotation.from_matrix(pose1[:3, :3])
        rot2 = Rotation.from_matrix(pose2[:3, :3])
        
        # Compute relative rotation
        rot_relative = rot2 * rot1.inv()
        
        # Interpolate rotation
        rot_interpolated = Rotation.from_rotvec(rot_relative.as_rotvec() * alpha) * rot1
        result[:3, :3] = rot_interpolated.as_matrix()
        
        return result
    
    def _on_button_press(self, button_name):
        self.button_states[button_name] = True
        print(f"{button_name} pressed")
    
    def _on_button_release(self, button_name):
        self.button_states[button_name] = False
        print(f"{button_name} released")
        
        # Reset trajectory state when button is released
        if button_name.startswith("traj_"):
            traj_name = button_name.replace("traj_", "")
            if traj_name in self.traj_state:
                self.traj_state[traj_name] = "idle"
                self.move_to_start_progress[traj_name] = 0.0
                self.start_arm_pose[traj_name] = None
                self.start_hand_pose[traj_name] = None
                self.traj_state_labels[traj_name].config(text=self._get_traj_state_text(traj_name))
    
    def _toggle_mode(self):
        self.control_mode = 'cartesian' if self.control_mode == 'joint' else 'joint'
        self.mode_label.config(text=f"Mode: {self.control_mode.upper()}")
        print(f">>> Control Mode: {self.control_mode.upper()}")
    
    def _handle_trajectory_playback(self, pressed_buttons):
        """Handle trajectory playback while button is pressed"""
        # Extract trajectory names from pressed buttons
        traj_buttons = [name for name in pressed_buttons if name.startswith("traj_")]
        
        if not traj_buttons:
            return
        
        # Only handle the first pressed trajectory button
        traj_button = traj_buttons[0]
        traj_name = traj_button.replace("traj_", "")
        
        if traj_name not in self.predefined_traj:
            return
        
        traj = self.predefined_traj[traj_name]
        arm_traj = traj['arm']  # (N, 4, 4)
        hand_traj = traj['hand']  # (N, 16)
        traj_length = len(arm_traj)
        
        # State machine
        if self.traj_state[traj_name] == "idle":
            # First press - initialize and start moving to start position
            self.traj_state[traj_name] = "moving_to_start"
            self.move_to_start_progress[traj_name] = 0.0
            self.start_arm_pose[traj_name] = self.robot.get_data()['position'].copy()
            self.start_hand_pose[traj_name] = self.hand.get_data()['qpos'].copy()
            self.traj_state_labels[traj_name].config(text=self._get_traj_state_text(traj_name))
        
        if self.traj_state[traj_name] == "moving_to_start":
            # Move to start position
            target_arm_pose = arm_traj[0]
            target_hand_pose = hand_traj[0]
            
            current_arm_pose = self.robot.get_data()['position'].copy()
            current_hand_pose = self.hand.get_data()['qpos'].copy()
            
            # Calculate distances
            arm_trans_distance = np.linalg.norm(target_arm_pose[:3, 3] - current_arm_pose[:3, 3])
            hand_distance = np.linalg.norm(target_hand_pose - current_hand_pose)
            
            # Check if close enough to start playing
            if arm_trans_distance < 0.005 :  # 5mm, 0.01 rad
                self.traj_state[traj_name] = "playing"
                self.traj_progress[traj_name] = 0.0
                # print(f"{traj_name} - reached start position, now playing...")
                self.traj_state_labels[traj_name].config(text=self._get_traj_state_text(traj_name))
            else:
                # Continue moving to start
                self.move_to_start_progress[traj_name] = min(1.0, 
                    self.move_to_start_progress[traj_name] + self.move_to_start_speed)
                
                # Interpolate to start position
                new_arm_pose = self._interpolate_pose(
                    self.start_arm_pose[traj_name], 
                    target_arm_pose, 
                    self.move_to_start_progress[traj_name]
                )
                new_hand_pose = (self.start_hand_pose[traj_name] + 
                               (target_hand_pose - self.start_hand_pose[traj_name]) * 
                               self.move_to_start_progress[traj_name])
                # print(target_hand_pose, self.start_hand_pose[traj_name])
                self.robot.move(new_arm_pose, is_servo=True)
                self.hand.move(target_hand_pose)
                self.traj_state_labels[traj_name].config(text=self._get_traj_state_text(traj_name))
        
        elif self.traj_state[traj_name] == "playing":
            # Play trajectory forward while button is pressed
            if self.traj_progress[traj_name] < traj_length - 1:
                self.traj_progress[traj_name] = min(traj_length - 1, 
                                                   self.traj_progress[traj_name] + self.traj_speed)
                
                # Get interpolated positions
                idx = int(self.traj_progress[traj_name])
                alpha = self.traj_progress[traj_name] - idx
                
                if idx + 1 < traj_length:
                    # Interpolate between frames
                    arm_pose = self._interpolate_pose(arm_traj[idx], arm_traj[idx + 1], alpha)
                    hand_pos = hand_traj[idx] * (1 - alpha) + hand_traj[idx + 1] * alpha
                else:
                    arm_pose = arm_traj[idx]
                    hand_pos = hand_traj[idx]
                
                # Move robot and hand
                self.robot.move(arm_pose, is_servo=True)
                self.hand.move(hand_pos)
                
                # Update progress label
                self.traj_progress_labels[traj_name].config(text=self._get_traj_progress_text(traj_name))
            else:
                print(f"Trajectory {traj_name} completed!")
    
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
                # Check if trajectory button is pressed (highest priority)
                traj_buttons = [name for name in pressed_buttons if name.startswith("traj_")]
                
                if traj_buttons:
                    self._handle_trajectory_playback(pressed_buttons)
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