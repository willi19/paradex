import time
import numpy as np
from paradex.transforms.coordinate import DEVICE2WRIST,  DEVICE2GLOBAL
from paradex.retargetor.hand_regargetor import inspire, allegro
from paradex.retargetor.limiter import PoseCommandLimiter

class Retargetor(): # Input is only from Xsens
    def __init__(self, arm_name=None, hand_name=None, translation_scale=1.0,
                 limit_wrist=True):
        self.arm_name = arm_name
        self.hand_name = hand_name
        self.translation_scale = float(translation_scale)
        # Reject VIVE tracking spikes so the arm doesn't lurch on glitches.
        self.limit_wrist = bool(limit_wrist)
        self._limiter = None
        
        if arm_name not in [None, "xarm", "franka"]:
            raise ValueError("Invalid arm name")
        if hand_name not in [None, "inspire", "allegro"]:
            raise ValueError("Invalid hand name")
        
        self.hand_retargetor = None
        if self.hand_name == "inspire":
            self.hand_retargetor = inspire
        elif self.hand_name == "allegro":
            self.hand_retargetor = allegro
        else:
            self.hand_retargetor = None

        if self.arm_name is not None:
            self.device2wrist = DEVICE2WRIST[self.arm_name].copy()
            self.device2global = DEVICE2GLOBAL[self.arm_name].copy()
        else:
            self.device2wrist = DEVICE2WRIST[self.hand_name].copy()
            self.device2global = np.eye(4)

    def get_action(self, data):
        fresh_anchor = self.init_human_pose is None
        if fresh_anchor:
            self.init_human_pose = data["Right"]["wrist"].copy()
        # print(self.device2wrist.T.shape)
        delta_wrists_R = self.device2wrist[:3,:3].T @ np.linalg.inv(self.init_human_pose[:3,:3]) @ data["Right"]["wrist"][:3,:3] @ self.device2wrist[:3,:3]
        delta_wrists_t = data["Right"]["wrist"][:3,3] - self.init_human_pose[:3,3]

        # Align the VIVE-world translation delta to the robot base frame. This
        # was applied raw before (assumes VIVE world == robot base); set
        # DEVICE2GLOBAL[arm] to the actual VIVE->robot-base rotation to fix the
        # "hand forward -> robot goes sideways" mismatch.
        delta_t_robot = self.device2global[:3, :3] @ delta_wrists_t

        robot_wrist_pose = np.zeros((4,4))
        robot_wrist_pose[:3,:3] = self.init_robot_pose[:3,:3] @ delta_wrists_R

        robot_wrist_pose[:3,3] = self.translation_scale * delta_t_robot + self.init_robot_pose[:3,3]
        robot_wrist_pose[3,3] = 1

        if getattr(self, "_debug_translation", False):
            import time as _t
            if _t.time() - getattr(self, "_dbg_tt", 0.0) > 0.7:
                print(f"[retgt] VIVE dt(m)={np.round(delta_wrists_t,3)} "
                      f"-> robot dt(m)={np.round(self.translation_scale*delta_t_robot,3)}")
                self._dbg_tt = _t.time()
        
        # Slew-rate limit the commanded wrist pose: the target advances toward the
        # raw hand pose at a bounded speed, giving even velocity and keeping the
        # streamed servo target from outrunning the arm (which caused the uneven
        # speed + the post-'s' j4/j5 coast). On the first frame after start()/
        # stop() (fresh_anchor) just reanchor so there's no startup jump.
        if self._limiter is not None:
            now = time.time()
            if fresh_anchor:
                self._limiter.reanchor(robot_wrist_pose, now)
            else:
                robot_wrist_pose = self._limiter.clamp(robot_wrist_pose, now)

        self.last_arm_pose = robot_wrist_pose.copy()
        arm_action = robot_wrist_pose.copy()
        
        if self.hand_name is not None:
            ergo = (data.get("ergonomics") or {}).get("Right") if isinstance(data, dict) else None
            if self.hand_name == "inspire" and ergo:
                # MANUS ergonomics -> reliable finger + thumb command (geometric
                # inspire()'s thumb saturates outside [0,1000]).
                from paradex.retargetor.hand_regargetor import inspire_from_manus_ergonomics
                try:
                    hand_action = inspire_from_manus_ergonomics(ergo)
                except (ValueError, KeyError):
                    hand_action = self.hand_retargetor(data["Right"])
            else:
                hand_action = self.hand_retargetor(data["Right"])
        else:
            hand_action = None

        return arm_action, hand_action

    def start(self, home_pose):
        self.init_robot_pose = home_pose.copy()
        self.last_arm_pose = home_pose.copy()
        self.init_human_pose = None
        if self.limit_wrist:
            self._limiter = PoseCommandLimiter(home_pose, time.time())
        else:
            self._limiter = None

    def stop(self):
        self.init_human_pose = None
        self.init_robot_pose = self.last_arm_pose.copy()