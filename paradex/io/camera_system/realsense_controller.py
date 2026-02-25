from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Event, Thread

import cv2
import numpy as np
import pyrealsense2 as rs


class realsense_controller:
    def __init__(
        self,
        color_size=(640, 360),
        depth_size=(320, 240),
        #     color_size=(1920, 1080),
        # depth_size=(1024, 768),
        default_fps=30,
        warmup_frames=30,
    ):
        self.color_size = color_size
        self.depth_size = depth_size
        self.default_fps = default_fps
        self.warmup_frames = warmup_frames

        self.pipeline = None
        self.capture_thread = None
        self.stop_event = Event()

        self.rgb_writer = None
        self.depth_writer = None

        self.use_depth = True
        self.color_format = rs.format.bgr8
        self.serial = None
        self.fps = default_fps
        self.save_dir = None
        self.session_prefix = None
        self.started = False
        
        self.ctx = rs.context()

    def _has_profile(self, device, stream_type, fmt, size, fps):
        for sensor in device.sensors:
            for profile in sensor.get_stream_profiles():
                vsp = profile.as_video_stream_profile()
                if vsp.stream_type() != stream_type:
                    continue
                if vsp.format() != fmt:
                    continue
                if (vsp.width(), vsp.height(), vsp.fps()) == (size[0], size[1], fps):
                    return True
        return False

    def _select_device_and_color_format(self, devices):
        for device in devices:
            if self.use_depth and not self._has_profile(
                device, rs.stream.depth, rs.format.z16, self.depth_size, self.fps
            ):
                continue
            if self._has_profile(device, rs.stream.color, rs.format.bgr8, self.color_size, self.fps):
                return device, rs.format.bgr8
            if self._has_profile(device, rs.stream.color, rs.format.rgb8, self.color_size, self.fps):
                return device, rs.format.rgb8
        return None, None

    def start(self, save_path, fps=30, use_depth=True):
        if self.started:
            raise RuntimeError("RealSense controller is already recording.")

        self.fps = fps
        self.use_depth = use_depth
        self.save_dir = Path(save_path)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ctx = rs.context()
        devices = self.ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense device connected.")

        device, self.color_format = self._select_device_and_color_format(devices)
        if device is None:
            if self.use_depth:
                raise RuntimeError(
                    "Couldn't find a connected device supporting "
                    f"RGB {self.color_size[0]}x{self.color_size[1]}@{self.fps} and "
                    f"Depth {self.depth_size[0]}x{self.depth_size[1]}@{self.fps}."
                )
            raise RuntimeError(
                "Couldn't find a connected device supporting "
                f"RGB {self.color_size[0]}x{self.color_size[1]}@{self.fps}."
            )

        self.serial = device.get_info(rs.camera_info.serial_number)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color,
            self.color_size[0],
            self.color_size[1],
            self.color_format,
            self.fps,
        )
        if self.use_depth:
            config.enable_stream(
                rs.stream.depth,
                self.depth_size[0],
                self.depth_size[1],
                rs.format.z16,
                self.fps,
            )

        self.pipeline.start(config)

        self._open_writers()
        self.stop_event.clear()
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.started = True

        print(f"RealSense started: serial={self.serial}, fps={self.fps}, depth={self.use_depth}")

    def _open_writers(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        rgb_path = self.save_dir / f"RealSense_rgb.mp4"
        self.rgb_writer = cv2.VideoWriter(str(rgb_path), fourcc, self.fps, self.color_size)
        if not self.rgb_writer.isOpened():
            raise RuntimeError(f"Failed to open RGB writer: {rgb_path}")

        if self.use_depth:
            depth_path = self.save_dir / f"Realsense_depth_vis.mp4"
            self.depth_writer = cv2.VideoWriter(str(depth_path), fourcc, self.fps, self.depth_size)
            if not self.depth_writer.isOpened():
                raise RuntimeError(f"Failed to open depth writer: {depth_path}")

    def _capture_loop(self):
        while not self.stop_event.is_set():
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            if self.color_format == rs.format.rgb8:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            self.rgb_writer.write(color_image)

            if self.use_depth:
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                self.depth_writer.write(depth_vis)

    def stop(self):
        if not self.started:
            return

        self.stop_event.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=3.0)
            self.capture_thread = None

        if self.rgb_writer is not None:
            self.rgb_writer.release()
            self.rgb_writer = None
        if self.depth_writer is not None:
            self.depth_writer.release()
            self.depth_writer = None

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None

        self.started = False
        print("RealSense stopped.")

    def end(self):
        self.stop()
