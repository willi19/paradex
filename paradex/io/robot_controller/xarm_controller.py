import time
from scipy.spatial.transform import Rotation
import transforms3d as t3d

import numpy as np
from threading import Thread, Event, Lock
import json
import os

from xarm.wrapper import XArmAPI

from paradex.utils.log import get_logger

logger = get_logger("xarm")

action_dof = 6

def homo2cart(h):
    t = h[:3, 3] * 1000
    R = h[:3, :3]

    rpy = Rotation.from_matrix(R).as_euler("xyz")
    return np.concatenate([t, rpy])

def cart2homo(cart):
    pos = np.eye(4)
    pos[:3,3] = cart[:3] / 1000
    pos[:3, :3] = Rotation.from_euler('xyz', cart[3:]).as_matrix()
    return pos

def homo2aa(h):
    t = h[:3, 3] * 1000
    R = h[:3, :3]

    axis, angle = t3d.axangles.mat2axangle(R, unit_thresh=0.001)
    axis_angle = axis * angle
    return np.concatenate([t, axis_angle])

class XArmController:
    def __init__(self, ip):
        self.xarm_ip_address = ip
        
        self.fps = 100
        self._move_speed = None
        self.reset()

        self.lock = Lock()
        
        self.exit_event = Event()
        self.save_event = Event()
        self.error_event = Event()
        self.connect_event = Event()
        self.position_control_event = Event()
        
        self.thread = Thread(target=self.control_loop, daemon=True)
        self.thread.start()
        
        self.connect_event.set()
        
    def control_loop(self):
        with self.lock:
            self.action = np.array(self.arm.get_joint_states(is_radian=True)[1][0])[:6]
            
            
        self.is_servo = True
        self.connect_event.set()
        
        while not self.exit_event.is_set():
            # print("asdfasdf")
            start_time = time.perf_counter()
            if self.arm.has_err_warn:
                # Arm faulted (e.g. kinematics/overspeed error). Stop hammering
                # servo commands at a dead arm, and wake any move(is_servo=False)
                # waiter so it does not block forever. Recovery is left to the
                # caller (clear_error()/reset()) — auto-clearing could re-drive
                # straight back into the same singularity.
                if not self.error_event.is_set():
                    logger.error(
                        "xArm fault: error_code=%s warn_code=%s state=%s. "
                        "Halting servo output; call clear_error()/reset() to recover.",
                        self.arm.error_code, self.arm.warn_code, self.arm.state,
                    )
                self.error_event.set()
                self.position_control_event.set()
                time.sleep(0.05)
                continue

            try:
                self._control_step()
            except Exception as e:
                # An IO/SDK call raised. Do not let it silently kill this thread —
                # that would leave move()'s waiters blocked forever (the classic
                # "device errored and the whole program hangs, even Ctrl-C"). Flag
                # the error, wake any waiter, and keep the loop alive so the caller
                # can recover (clear_error()/reset()) or exit cleanly.
                logger.exception("xArm control loop IO error: %s", e)
                self.error_event.set()
                self.position_control_event.set()
                time.sleep(0.05)
                continue

            elapsed = time.perf_counter() - start_time
            time.sleep(max(0, (1 / self.fps) - elapsed))
            
        logger.info("Control loop exited.")

    def _control_step(self):
        with self.lock:
            action = self.action.copy()
            is_servo = self.is_servo
            move_speed = self._move_speed

        is_joint_value = (action.shape == (6,))

        if not is_servo and not self.finished:
            self.arm.set_mode(0)  # 0: position control, 1: servo control
            self.arm.set_state(state=0)

            if is_joint_value:
                kwargs = dict(angle=action.tolist(), is_radian=True, wait=True)
                if move_speed is not None:
                    kwargs['speed'] = move_speed
                self.arm.set_servo_angle(**kwargs)
            else:
                cart = homo2cart(action)
                self.arm.set_position(x = cart[0],
                                        y = cart[1],
                                        z = cart[2],
                                        roll = cart[3],
                                        pitch = cart[4],
                                        yaw = cart[5],
                                        speed=100,
                                        is_radian=True,
                                        wait=True) # motion_type=1 if necessary to go home but this is too dangerous

            self.arm.set_mode(1)
            self.arm.set_state(state=0)
            self.finished = True
            self.position_control_event.set()


        else:
            if is_joint_value:
                self.arm.set_servo_angle_j(angles=action.tolist(), is_radian=True)
            else:
                aa = homo2aa(action)
                self.arm.set_servo_cartesian_aa(aa, is_radian=True)

        if self.save_event.is_set():
            if is_joint_value:
                cart = self.arm.get_forward_kinematics(action.tolist(), input_is_radian=True, return_is_radian=True)[1]
                qpos = action.copy()
            else:
                cart = homo2cart(action)
                success, qpos = self.arm.get_inverse_kinematics(cart)
                qpos = np.array(qpos)
                if success != 0:
                    logger.warning("inverse kinematics failed (code=%s) for cart=%s", success, cart)
                    qpos = - np.ones(6)

            _, state = self.arm.get_joint_states(is_radian=True)
            self.data["position"].append(np.array(state[0])[:6])
            self.data["velocity"].append(np.array(state[1])[:6])
            self.data["torque"].append(np.array(state[2])[:6])
            self.data["time"].append(time.time())
            self.data["action"].append(cart)

            self.data["action_qpos"].append(qpos[:6])

    def start(self, save_path):            
        self.save_path = save_path
        os.makedirs(os.path.join(self.save_path), exist_ok=True)
        
        self.data = {
            "time":[],# np.zeros((T,1), dtype=np.float64),
            "position":[],# np.zeros((T, action_dof), dtype=np.float64),
            "velocity":[],# np.zeros((T, action_dof), dtype=np.float64),
            "torque":[],# np.zeros((T, action_dof), dtype=np.float64),
            "action":[], #np.zeros((T, 4, 4), dtype=np.float64),
            "action_qpos":[] #np.zeros((T, action_dof), dtype=np.float64) # x y z rpy
        }
        
        with self.lock:
            self.save_event.set()

    def stop(self):
        with self.lock:
            self.save_event.clear()
        for name, value in self.data.items():          
            np.save(os.path.join(self.save_path, f"{name}.npy"), np.array(value))
            self.data[name] = []
        self.save_path = None
        
    def end(self, set_break=False):
        self.exit_event.set()
        self.thread.join(timeout=2.0)
        
        if set_break:
            self.arm.motion_enable(enable=False)
        self.arm.disconnect()
        if self.save_event.is_set():
            self.stop()
    
    def move(self, action, is_servo=True, speed=None):
        assert action.shape == (4,4) or action.shape == (6,)

        # Clamp joint values to xArm limits to prevent out_of_joint_range errors
        if action.shape == (6,):
            action = np.clip(action, -2 * np.pi, 2 * np.pi)

        with self.lock:
            self.action = action.copy()
            self.is_servo = is_servo
            self._move_speed = speed

        if not is_servo:
            self.finished = False
            self.position_control_event.clear()
            # Poll with a timeout so KeyboardInterrupt (Ctrl-C) is not swallowed
            # by an uninterruptible no-timeout wait, and so a fault/exit releases
            # us instead of blocking forever.
            while not self.position_control_event.wait(timeout=0.2):
                if self.exit_event.is_set() or self.error_event.is_set():
                    break
        
    def clear_error(self):
        """Clear errors/warnings and re-enable servo mode without reconnecting."""
        if self.arm.has_err_warn:
            self.arm.clean_warn()
            self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(state=0)
        self.error_event.clear()
        time.sleep(0.1)

    def reset(self):
        self.arm = XArmAPI(self.xarm_ip_address, report_type="devlop")
        if self.arm.has_err_warn:
            self.arm.clean_warn()
            self.arm.clean_error()

        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        time.sleep(0.1)


        self.arm.set_mode(1)
        self.arm.set_state(state=0)
    
    def get_data(self):
        with self.lock:
            qpos = np.array(self.arm.get_joint_states(is_radian=True)[1][0])[:6]
            cart = np.array(self.arm.get_position(is_radian=True)[1])

        pos = cart2homo(cart)
        return {
            "qpos": qpos,
            "position": pos,
            "time": time.time()
        }
    
    def is_error(self):
        return self.error_event.is_set()