import ipaddress
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paradex.io.camera_system.aravis_addressing import (
    CameraAddressing,
    CameraRecord,
    NicSubnet,
    gvcp_forceip_packet,
)
from paradex.io.camera_system.aravis_gstreamer import (
    AravisGStreamerError,
    AravisGStreamerCameraLoader,
    AravisGStreamerSettings,
    trigger_features,
)
from paradex.io.camera_system.camera_server_daemon import camera_server_daemon


class FakeCamera:
    def __init__(self, name, fail_start=False):
        self.name = name
        self.fail_start = fail_start
        self.calls = []

    def start(self, mode, sync_mode, save_path, fps):
        self.calls.append(("start", mode, sync_mode, save_path, fps))
        if self.fail_start:
            raise RuntimeError("synthetic start failure")

    def stop(self):
        self.calls.append(("stop",))

    def end(self):
        self.calls.append(("end",))

    def get_error(self):
        return False, (None, None)

    def get_status(self):
        return {"name": self.name, "state": "READY"}


class FakeLoader:
    def __init__(self):
        self.calls = []

    def start(self, mode, sync_mode, save_path, fps):
        self.calls.append(("start", mode, sync_mode, save_path, fps))

    def stop(self):
        self.calls.append(("stop",))

    def end(self):
        self.calls.append(("end",))

    def get_status_list(self):
        return []

    def get_all_errors(self):
        return {}


class AravisCaptureTests(unittest.TestCase):
    def test_forceip_packet_has_expected_gvcp_fields(self):
        packet = gvcp_forceip_packet("2c:dd:a3:7d:a6:9c", "11.0.3.100", request_id=42)

        self.assertEqual(len(packet), 64)
        self.assertEqual(packet[:2], b"B\x01")
        self.assertEqual(struct.unpack_from(">H", packet, 2)[0], 4)
        self.assertEqual(struct.unpack_from(">H", packet, 4)[0], 56)
        self.assertEqual(struct.unpack_from(">H", packet, 6)[0], 42)
        self.assertEqual(packet[10:16], bytes.fromhex("2cdda37da69c"))
        self.assertEqual(packet[28:32], bytes((11, 0, 3, 100)))

    def test_forceip_plan_prefers_discovered_physical_nic(self):
        left = NicSubnet("enp5s0", "11.0.1.1", ipaddress.ip_network("11.0.1.0/24"))
        right = NicSubnet("enp6s0", "11.0.2.1", ipaddress.ip_network("11.0.2.0/24"))
        addressing = CameraAddressing([left, right])
        record = CameraRecord("cam-a", "FLIR-cam-a", "169.254.1.2", "2c:dd:a3:7d:a6:9c")
        addressing._seen = {record.serial: record}
        addressing._mac_to_nic = {record.mac: right}

        plan = addressing._plan_force_ips()

        self.assertEqual(plan["cam-a"], (right, "11.0.2.100"))

    def test_trigger_features_rearm_line0_after_caps_negotiation(self):
        settings = AravisGStreamerSettings(trigger_source="Line0", trigger_activation="RisingEdge")

        self.assertEqual(trigger_features(settings, False), "TriggerMode=Off")
        self.assertEqual(
            trigger_features(settings, True),
            "TriggerSelector=FrameStart TriggerSource=Line0 "
            "TriggerActivation=RisingEdge TriggerOverlap=ReadOut TriggerMode=On",
        )

    def test_loader_preserves_existing_avi_output_layout(self):
        cameras = {}

        def factory(serial):
            camera = FakeCamera(serial)
            cameras[serial] = camera
            return camera

        with tempfile.TemporaryDirectory() as temp_dir:
            capture_roots = [str(Path(temp_dir) / "captures1"), str(Path(temp_dir) / "captures2")]
            with patch("paradex.io.camera_system.aravis_gstreamer.capture_path_list", capture_roots):
                loader = AravisGStreamerCameraLoader(
                    serial_list=["cam-a", "cam-b", "cam-c"],
                    camera_factory=factory,
                    reconcile_addresses=False,
                )
                loader.start("video", True, "dataset/session/raw", 30)

            self.assertEqual(
                cameras["cam-a"].calls[0],
                ("start", "video", True, str(Path(capture_roots[0]) / "dataset/session/raw/videos/cam-a.avi"), 30),
            )
            self.assertEqual(
                cameras["cam-b"].calls[0],
                ("start", "video", True, str(Path(capture_roots[1]) / "dataset/session/raw/videos/cam-b.avi"), 30),
            )
            self.assertEqual(
                cameras["cam-c"].calls[0],
                ("start", "video", True, str(Path(capture_roots[0]) / "dataset/session/raw/videos/cam-c.avi"), 30),
            )

    def test_loader_stops_all_cameras_after_partial_start_failure(self):
        first = FakeCamera("cam-a")
        second = FakeCamera("cam-b", fail_start=True)
        cameras = iter((first, second))
        loader = AravisGStreamerCameraLoader(
            serial_list=["cam-a", "cam-b"],
            camera_factory=lambda _serial: next(cameras),
            reconcile_addresses=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("paradex.io.camera_system.aravis_gstreamer.capture_path_list", [temp_dir]):
                with self.assertRaisesRegex(RuntimeError, "synthetic start failure"):
                    loader.start("video", True, "session/raw", 30)

        self.assertIn(("stop",), first.calls)
        self.assertIn(("stop",), second.calls)

    def test_loader_rejects_pyspin_only_shared_memory_modes(self):
        loader = AravisGStreamerCameraLoader(
            serial_list=["cam-a"],
            camera_factory=FakeCamera,
            reconcile_addresses=False,
        )

        with self.assertRaisesRegex(AravisGStreamerError, "pyspin agent"):
            loader.start("stream", False, fps=30)

    def test_server_reports_ready_only_after_loader_start_returns(self):
        loader = FakeLoader()
        server = camera_server_daemon(loader=loader, start_threads=False)
        try:
            self.assertEqual(
                server.execute_command({"action": "register", "controller_name": "main"})["status"],
                "ok",
            )
            response = server.execute_command(
                {
                    "action": "start",
                    "controller_name": "main",
                    "mode": "video",
                    "syncMode": True,
                    "save_path": "session/raw",
                    "fps": 30,
                }
            )
            self.assertEqual(response, {"status": "ok", "msg": "ready"})
            self.assertEqual(loader.calls, [("start", "video", True, "session/raw", 30)])
        finally:
            server.ctx.term()


if __name__ == "__main__":
    unittest.main()
