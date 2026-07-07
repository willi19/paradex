import os
import time
import numpy as np
import chime
chime.theme('pokemon')

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.io.robot_controller import get_arm, get_hand
from paradex.utils.path import shared_dir
from paradex.retargetor.state import HandStateExtractor
from paradex.retargetor.unimanual import Retargetor
from paradex.calibration.utils import save_current_camparam, save_current_C2R
from paradex.utils.system import network_info

class CaptureSession():
    """Record a multi-sensor capture session through one lifecycle.

    Fans a single ``start`` / ``stop`` / ``end`` across the camera rig, robot arm,
    robot hand, and XSens teleop, writing a fixed ``raw/`` directory tree under
    ``shared_dir``. Only the devices requested at construction are built; unused
    ones stay ``None`` and are skipped in every lifecycle method, so camera-only,
    arm-only, and camera+arm+hand+teleop sessions all flow through the same calls.

    See the :doc:`dataset-acquisition guide </dataset_acquisition>` for the mental
    model, the ``raw/`` layout, and the sync-generator ordering invariant.
    """

    def __init__(self, camera=False, arm=None, hand=None, teleop=None, hand_ip=False):
        """Build the requested devices.

        Parameters
        ----------
        camera : bool
            If True, build the remote camera controller and the UTGE900 sync
            generator. A ``TimestampMonitor`` is added only when ``arm`` or
            ``hand`` is also set (nothing to cross-sync otherwise).
        arm : str or None
            Arm name for ``get_arm`` (e.g. ``"xarm"``), or None to omit the arm.
        hand : str or None
            Hand name for ``get_hand`` (e.g. ``"allegro"``, ``"inspire"``), or None.
        teleop : str or None
            ``"xsens"`` builds the XSens receiver, retargetor, and hand-state
            extractor. Requires ``arm`` or ``hand``. ``"occulus"`` is not implemented.
        hand_ip : bool
            Passed through to ``get_hand`` (Inspire needs an IP socket).

        Raises
        ------
        ValueError
            If ``teleop`` is set but neither ``arm`` nor ``hand`` is.
        """
        if arm is None and hand is None and teleop is not None:
            raise ValueError("Teleop device requires at least one of arm or hand to be specified.")
        
        if camera:
            self.camera = remote_camera_controller(name="dataset_acquisition")
            self.sync_generator = UTGE900(**network_info["signal_generator"]["param"])
            if arm is not None or hand is not None:
                self.timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])
            else:
                self.timestamp_monitor = None
        else:
            self.camera = None
            self.timestamp_monitor = None
            self.sync_generator = None
        
        if arm is not None:
            self.arm = get_arm(arm)
        else:
            self.arm = None
        
        if hand is not None:
            self.hand = get_hand(hand, ip=hand_ip)
        else:
            self.hand = None
            
        if teleop is not None:
            if teleop == "xsens":
                from paradex.io.teleop.xsens.receiver import XSensReceiver
                self.teleop_device = XSensReceiver(**network_info["xsens"]["param"])
            
            # elif teleop == "occulus":
            #     from paradex.io.teleop.oculus.receiver import OculusReceiver
            #     self.teleop_device = OculusReceiver()
            self.retargetor = Retargetor(arm_name=arm, hand_name=hand)
            self.state_extractor = HandStateExtractor()
            
        else:
            self.teleop_device = None

        self.save_path = None
        self.stage = None

    def start(self, save_path, mode="video", fps=30, exposure_time=None, gain=None, stage=None): # Start recording on all sensors
        """Start recording on every enabled sensor.

        Creates ``<save_path>/raw[/<stage>]/`` under ``shared_dir`` and starts each
        recorder into its own subdir (``arm/``, ``hand/``, ``teleop/``,
        ``timestamps/``, plus camera ``videos/`` or ``images/``). The camera runs in
        sync mode and the UTGE900 trigger is fired **last**, after every trigger
        consumer (camera + timestamp monitor) is armed.

        Parameters
        ----------
        save_path : str
            Session directory **relative to** ``shared_dir`` (e.g.
            ``"capture/<pipeline>/<name>/<ts>"``).
        mode : str
            ``"image"`` captures stills; any other value (default ``"video"``) arms
            the camera for continuous recording into a video sink.
        fps : int
            Frame rate; also passed to the sync generator.
        exposure_time, gain : optional
            None uses the per-camera ``camera.json`` baseline (recommended).
        stage : str or None
            Optional sub-bucket: data lands under ``raw/<stage>/…`` instead of
            ``raw/…`` (the layout the upload processor expects when set).
        """
        self.save_path = save_path
        self.stage = stage
        # raw_rel is the directory under save_path that holds per-session data.
        # When stage is provided, layout matches what the upload processor
        # expects: raw/{stage}/videos vs raw/{stage}/arm vs ...
        raw_rel = os.path.join("raw", stage) if stage else "raw"
        os.makedirs(os.path.join(shared_dir, save_path, raw_rel), exist_ok=True)

        if self.arm is not None:
            self.arm.start(os.path.join(shared_dir, save_path, raw_rel, "arm"))

        if self.hand is not None:
            self.hand.start(os.path.join(shared_dir, save_path, raw_rel, "hand"))

        if self.teleop_device is not None:
            self.teleop_device.start(os.path.join(shared_dir, save_path, raw_rel, "teleop"))
            self.state_hist = []
            self.state_time = []

        if self.camera is not None:
            cam_rel = os.path.join(save_path, raw_rel)
            if mode == "image":
                self.camera.start("image", True, cam_rel, fps=fps,
                                  exposure_time=exposure_time, gain=gain)
            else:  # continuous recording: arm + video sink
                self.camera.arm(True, fps=fps, exposure_time=exposure_time, gain=gain)
                self.camera.set_record(cam_rel, on=True)
            if self.timestamp_monitor is not None:
                self.timestamp_monitor.start(os.path.join(shared_dir, save_path, raw_rel, "timestamps"))
            self.sync_generator.start(fps=fps)
        
    def stop(self):
        """Stop all sensors and flush per-session artifacts.

        Teleop sessions dump ``state/state_hist.npy`` + ``state/state_time.npy``.
        Camera sessions stop the timestamp monitor and the sync generator (in that
        order), then snapshot the current calibration next to the data via
        ``save_current_camparam`` + ``save_current_C2R`` so each dataset carries the
        calibration it was shot with. Clears ``save_path``/``stage``, re-arming the
        session for another ``start``.
        """
        if self.arm is not None:
            self.arm.stop()
        if self.hand is not None:
            self.hand.stop()
            
        if self.teleop_device is not None:
            self.teleop_device.stop()
            raw_rel = os.path.join("raw", self.stage) if self.stage else "raw"
            os.makedirs(os.path.join(shared_dir, self.save_path, raw_rel, "state"), exist_ok=True)
            np.save(os.path.join(shared_dir, self.save_path, raw_rel, "state", "state_hist.npy"), np.array(self.state_hist))
            np.save(os.path.join(shared_dir, self.save_path, raw_rel, "state", "state_time.npy"), np.array(self.state_time))

        if self.camera is not None:
            self.camera.stop()
            if self.timestamp_monitor is not None:
                self.timestamp_monitor.stop()
            self.sync_generator.stop()
        
            save_current_camparam(os.path.join(shared_dir, self.save_path))
            save_current_C2R(os.path.join(shared_dir, self.save_path))

        self.save_path = None
        self.stage = None

    def end(self):
        """Release every device. ``camera.end()`` frees the daemon lock.

        Call once per session lifetime — without it the camera lock is only freed
        after the daemon's idle timeout.
        """
        if self.arm is not None:
            self.arm.end()
        if self.hand is not None:
            self.hand.end()
        if self.teleop_device is not None:
            self.teleop_device.end()
        
        if self.camera is not None:
            self.camera.end()
            if self.timestamp_monitor is not None:
                self.timestamp_monitor.end()
            self.sync_generator.end()
    
    def teleop(self):
        """Run the blocking XSens retarget loop until the operator stops or exits.

        The operator's **left** hand drives a state machine: state 0 retargets and
        moves the arm/hand, 1 pauses, 2 is stop-hold, 3 is exit-hold. Holding stop
        or exit for more than ~90 ticks (~0.9 s at 10 ms) ends the loop. State is
        logged to ``state_hist``/``state_time`` only while a ``save_path`` is set
        (i.e. during the record phase, not the prepare phase).

        Returns
        -------
        str
            ``"exit"`` or ``"stop"`` depending on which hold gesture ended the loop.

        Raises
        ------
        ValueError
            If the session has no teleop device.
        """
        if self.teleop_device is None:
            raise ValueError("No teleop device initialized.")

        chime.warning(sync=True)
        exit_counter = 0
        stop_counter = 0

        home_pose = self.arm.get_data()["position"] if self.arm is not None else np.eye(4)
        
        self.retargetor.start(home_pose)

        while True:
            data = self.teleop_device.get_data()
            if data["Right"] is None:
                continue
            state = self.state_extractor.get_state(data['Left'])
            if self.save_path is not None:
                self.state_hist.append(state)
                self.state_time.append(time.time())
                
            if state == 0:
                wrist_pose, hand_action = self.retargetor.get_action(data)
                if self.hand is not None:
                    self.hand.move(hand_action)

                if self.arm is not None:
                    self.arm.move(wrist_pose.copy())

            if state == 1:
                self.retargetor.stop()
            
            if state == 2:
                self.retargetor.stop()
                stop_counter += 1
            
            else:
                stop_counter = 0
                
            if state == 3:
                exit_counter += 1
            
            else:
                exit_counter = 0
                
            if exit_counter > 90:
                chime.success(sync=True)
                return "exit"
        
            if stop_counter > 90:
                chime.info(sync=True)
                return "stop"
            time.sleep(0.01)
        
    
    def move(self, action_dict):
        """Scripted single-step control (alternative to :meth:`teleop`).

        Parameters
        ----------
        action_dict : dict
            May contain ``"arm"`` (wrist pose → ``arm.move``) and/or ``"hand"``
            (hand action → ``hand.move``). Keys are optional, but a key for a device
            the session did not build will raise ``AttributeError``.
        """
        if "arm" in action_dict:
            self.arm.move(action_dict["arm"])
        if "hand" in action_dict:
            self.hand.move(action_dict["hand"])
