import contextlib
import io
import ipaddress
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paradex.io.camera_system.aravis_addressing import (
    CameraAddressing,
    CameraRecord,
    NicSubnet,
    _camera_subnet,
    _is_deployed_11_camera_address,
    gvcp_forceip_packet,
)
from paradex.io.camera_system.aravis_gstreamer import (
    AravisGStreamerError,
    AravisGStreamerCamera,
    AravisGStreamerCameraLoader,
    AravisGStreamerSettings,
    _write_optional_feature,
    camera_caps,
    trigger_features,
)
from paradex.io.camera_system.camera_server_daemon import camera_server_daemon


class FakeCamera:
    def __init__(self, name, fail_start=False):
        self.name = name
        self.fail_start = fail_start
        self.calls = []
        self.frame_count = 17

    def start(self, mode, sync_mode, save_path, fps):
        self.calls.append(("start", mode, sync_mode, save_path, fps))
        if self.fail_start:
            raise RuntimeError("synthetic start failure")

    def prepare_hardware(self, fps=30, sync_mode=True):
        self.calls.append(("prepare_hardware", fps, sync_mode))

    def prepare(self, mode, sync_mode, save_path, fps):
        self.calls.append(("prepare", mode, sync_mode, save_path, fps))
        if self.fail_start:
            raise RuntimeError("synthetic start failure")

    def start_prepared(self):
        self.calls.append(("start_prepared",))

    def request_playing(self):
        self.calls.append(("request_playing",))

    def confirm_playing(self):
        self.calls.append(("confirm_playing",))

    def abort(self):
        self.calls.append(("abort",))

    def stop(self):
        self.calls.append(("stop",))

    def wait_for_first_frame(self, timeout_seconds=None):
        self.calls.append(("wait_for_first_frame", timeout_seconds))

    def end(self):
        self.calls.append(("end",))

    def get_error(self):
        return False, (None, None)

    def get_status(self):
        return {
            "name": self.name,
            "state": "READY",
            "frame_id": self.frame_count,
            "frame_count": self.frame_count,
        }


class FakeLoader:
    def __init__(self):
        self.calls = []

    def start(self, mode, sync_mode, save_path, fps):
        self.calls.append(("start", mode, sync_mode, save_path, fps))

    def stop(self):
        self.calls.append(("stop",))

    def wait_for_first_frames(self, timeout_seconds=None):
        self.calls.append(("validate", timeout_seconds))

    def end(self):
        self.calls.append(("end",))

    def get_status_list(self):
        return []

    def get_all_errors(self):
        return {}


class FakeTwoPhaseLoader(FakeLoader):
    def prepare(self, mode, sync_mode, save_path, fps):
        self.calls.append(("prepare", mode, sync_mode, save_path, fps))

    def activate(self):
        self.calls.append(("activate",))

    def abort(self):
        self.calls.append(("abort",))


class FakeAddressing:
    def __init__(self):
        self.nic_subnets = [
            NicSubnet("enp6s0f0", "11.0.1.1", ipaddress.ip_network("11.0.1.0/24"))
        ]
        self.seen = {
            "cam-a": CameraRecord(
                "cam-a",
                "FLIR-Blackfly-S-cam-a",
                "11.0.1.100",
                "2c:dd:a3:7d:a6:9c",
            )
        }

    def reconcile(self, expected_serials):
        return list(expected_serials)


class FakeOptionalFeatureDevice:
    def __init__(self, available=()):
        self.available = set(available)
        self.values = {}

    def set_boolean_feature_value(self, name, value):
        if name not in self.available:
            raise RuntimeError("node '{}' not found".format(name))
        self.values[name] = value

    def get_boolean_feature_value(self, name):
        return self.values[name]


class FakeAravisBufferFactory:
    @staticmethod
    def new_allocate(payload):
        return ("buffer", payload)


class FakeAravisModule:
    Buffer = FakeAravisBufferFactory

    class AcquisitionMode:
        CONTINUOUS = "continuous"


class FakeAravisStream:
    def __init__(self):
        self.buffers = []

    def push_buffer(self, buffer):
        self.buffers.append(buffer)


class FakeAravisCameraSession:
    def __init__(self, payload=4096):
        self.payload = payload
        self.stream = FakeAravisStream()
        self.mode = None

    def create_stream(self, callback, user_data):
        self.create_args = (callback, user_data)
        return self.stream

    def get_payload(self):
        return self.payload

    def set_acquisition_mode(self, mode):
        self.mode = mode


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

    def test_legacy_11_network_uses_its_physical_link_as_a_24(self):
        subnet = _camera_subnet(
            "enp5s0", "11.0.3.1", 16, explicitly_selected=False
        )

        self.assertEqual(
            subnet,
            NicSubnet("enp5s0", "11.0.3.1", ipaddress.ip_network("11.0.3.0/24")),
        )

    def test_deployed_11_camera_address_is_trusted(self):
        self.assertTrue(_is_deployed_11_camera_address("11.0.4.1"))
        self.assertFalse(_is_deployed_11_camera_address("192.168.0.1"))
        self.assertFalse(_is_deployed_11_camera_address("11.0.4.2"))

    def test_trigger_features_rearm_line0_after_caps_negotiation(self):
        settings = AravisGStreamerSettings(trigger_source="Line0", trigger_activation="RisingEdge")

        self.assertEqual(trigger_features(settings, False), "TriggerMode=Off")
        self.assertEqual(
            trigger_features(settings, True),
            "TriggerSelector=FrameStart TriggerSource=Line0 "
            "TriggerActivation=RisingEdge TriggerOverlap=ReadOut TriggerMode=On",
        )

    def test_hardware_trigger_appsrc_caps_include_avi_rate(self):
        settings = AravisGStreamerSettings()

        self.assertEqual(
            camera_caps(settings, 30, True),
            "video/x-bayer,format=rggb,width=2048,height=1536,framerate=30/1",
        )
        self.assertEqual(
            camera_caps(settings, 30, False),
            "video/x-bayer,format=rggb,width=2048,height=1536,framerate=30/1",
        )

    def test_direct_aravis_stream_uses_reusable_buffers_and_continuous_mode(self):
        settings = AravisGStreamerSettings(aravis_buffer_count=4)
        camera = AravisGStreamerCamera(
            "cam-a",
            settings=settings,
            camera_config={"cam-a": {}},
        )
        session = FakeAravisCameraSession(payload=8192)
        camera._aravis = FakeAravisModule
        camera._aravis_camera = session

        camera._create_aravis_stream()

        self.assertIs(camera._aravis_stream, session.stream)
        self.assertEqual(session.create_args, (None, None))
        self.assertEqual(session.stream.buffers, [("buffer", 8192)] * 4)
        self.assertEqual(session.mode, "continuous")

    def test_session_teardown_retains_prepared_aravis_hardware(self):
        camera = AravisGStreamerCamera(
            "cam-a", camera_config={"cam-a": {}}, device_id="device-a"
        )
        aravis = object()
        session = object()
        stream = object()
        camera._aravis = aravis
        camera._aravis_camera = session
        camera._aravis_stream = stream

        camera._teardown_pipeline(release_hardware=False)

        self.assertIs(camera._aravis, aravis)
        self.assertIs(camera._aravis_camera, session)
        self.assertIs(camera._aravis_stream, stream)

        camera._teardown_pipeline(release_hardware=True)
        self.assertIsNone(camera._aravis)
        self.assertIsNone(camera._aravis_camera)
        self.assertIsNone(camera._aravis_stream)

    def test_drain_returns_completed_buffers_before_next_session(self):
        class FakeStream:
            def __init__(self):
                self.completed = ["old-a", "old-b"]
                self.returned = []

            def try_pop_buffer(self):
                return self.completed.pop(0) if self.completed else None

            def push_buffer(self, buffer):
                self.returned.append(buffer)

        camera = AravisGStreamerCamera(
            "cam-a", camera_config={"cam-a": {}}, device_id="device-a"
        )
        stream = FakeStream()
        camera._aravis_stream = stream

        camera._drain_aravis_output()

        self.assertEqual(stream.returned, ["old-a", "old-b"])

    def test_hardware_sync_configuration_matches_paraoffice_feature_order(self):
        device = object()
        session = MagicMock()
        session.get_device.return_value = device
        aravis = MagicMock()
        aravis.Camera.new.return_value = session
        camera = AravisGStreamerCamera(
            "cam-a",
            camera_config={"cam-a": {"gain": 1.0, "exposure": 2200.0}},
            device_id="FLIR-Blackfly-S-cam-a",
        )

        with patch(
            "paradex.io.camera_system.aravis_gstreamer._load_aravis",
            return_value=aravis,
        ), patch("paradex.io.camera_system.aravis_gstreamer._write_feature") as write, patch(
            "paradex.io.camera_system.aravis_gstreamer._write_optional_feature"
        ) as optional:
            camera._configure_camera(30, True)

        feature_order = [entry.args[1] for entry in write.call_args_list]
        self.assertEqual(
            feature_order,
            [
                "GevHeartbeatTimeout",
                "PixelFormat",
                "GevSCPSPacketSize",
                "TriggerMode",
                "TriggerSelector",
                "TriggerSource",
                "TriggerActivation",
                "TriggerOverlap",
                "ExposureAuto",
                "ExposureTime",
                "GainAuto",
                "Gain",
                "TriggerMode",
            ],
        )
        self.assertEqual(
            [entry.args for entry in optional.call_args_list],
            [
                (device, ("AcquisitionFrameRateAuto",), "Off"),
                (
                    device,
                    ("AcquisitionFrameRateEnable", "AcquisitionFrameRateEnabled"),
                    True,
                ),
                (
                    device,
                    ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"),
                    30.0,
                ),
            ],
        )

    def test_loader_validates_first_frame_for_every_camera(self):
        first = FakeCamera("cam-a")
        second = FakeCamera("cam-b")
        cameras = iter((first, second))
        loader = AravisGStreamerCameraLoader(
            serial_list=["cam-a", "cam-b"],
            camera_factory=lambda _serial: next(cameras),
            reconcile_addresses=False,
        )
        loader._capture_active = True

        loader.wait_for_first_frames(2.5)

        self.assertIn(("wait_for_first_frame", 2.5), first.calls)
        self.assertIn(("wait_for_first_frame", 2.5), second.calls)

    def test_optional_frame_rate_enable_uses_alias_or_skips(self):
        aliased = FakeOptionalFeatureDevice({"AcquisitionFrameRateEnabled"})
        missing = FakeOptionalFeatureDevice()

        self.assertEqual(
            _write_optional_feature(
                aliased,
                ("AcquisitionFrameRateEnable", "AcquisitionFrameRateEnabled"),
                True,
            ),
            "AcquisitionFrameRateEnabled",
        )
        self.assertIsNone(
            _write_optional_feature(
                missing,
                ("AcquisitionFrameRateEnable", "AcquisitionFrameRateEnabled"),
                True,
            )
        )

    def test_loader_preserves_existing_avi_output_layout(self):
        cameras = {}

        def factory(serial):
            camera = FakeCamera(serial)
            cameras[serial] = camera
            return camera

        with tempfile.TemporaryDirectory() as temp_dir:
            capture_roots = [str(Path(temp_dir) / "captures1"), str(Path(temp_dir) / "captures2")]
            with patch("paradex.io.camera_system.aravis_gstreamer.capture_path_list", capture_roots), patch(
                "paradex.io.camera_system.aravis_gstreamer._load_aravis"
            ):
                loader = AravisGStreamerCameraLoader(
                    serial_list=["cam-a", "cam-b", "cam-c"],
                    camera_factory=factory,
                    reconcile_addresses=False,
                )
                loader.start("video", True, "dataset/session/raw", 30)

            self.assertEqual(
                cameras["cam-a"].calls[1],
                ("prepare", "video", True, str(Path(capture_roots[0]) / "dataset/session/raw/videos/cam-a.avi"), 30),
            )
            self.assertEqual(
                cameras["cam-b"].calls[1],
                ("prepare", "video", True, str(Path(capture_roots[1]) / "dataset/session/raw/videos/cam-b.avi"), 30),
            )
            self.assertEqual(
                cameras["cam-c"].calls[1],
                ("prepare", "video", True, str(Path(capture_roots[0]) / "dataset/session/raw/videos/cam-c.avi"), 30),
            )

    def test_loader_prints_discovered_camera_inventory(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            AravisGStreamerCameraLoader(
                serial_list=["cam-a"],
                addressing=FakeAddressing(),
                camera_factory=FakeCamera,
            )

        text = output.getvalue()
        self.assertIn("Aravis/GStreamer cameras loaded (1/1 configured)", text)
        self.assertIn("serial=cam-a", text)
        self.assertIn("ip=11.0.1.100", text)
        self.assertIn("nic=enp6s0f0", text)

    def test_loader_uses_exact_discovered_device_id(self):
        loader = AravisGStreamerCameraLoader(
            serial_list=["cam-a"],
            addressing=FakeAddressing(),
            prewarm_hardware=False,
        )

        self.assertEqual(loader.cameralist[0].device_id, "FLIR-Blackfly-S-cam-a")

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
            with patch("paradex.io.camera_system.aravis_gstreamer.capture_path_list", [temp_dir]), patch(
                "paradex.io.camera_system.aravis_gstreamer._load_aravis"
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic start failure"):
                    loader.start("video", True, "session/raw", 30)

        self.assertIn(("abort",), first.calls)
        self.assertIn(("abort",), second.calls)

    def test_loader_prints_frame_counts_once_when_capture_stops(self):
        camera = FakeCamera("cam-a")
        loader = AravisGStreamerCameraLoader(
            serial_list=["cam-a"],
            camera_factory=lambda _serial: camera,
            reconcile_addresses=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "paradex.io.camera_system.aravis_gstreamer.capture_path_list", [temp_dir]
        ), patch("paradex.io.camera_system.aravis_gstreamer._load_aravis"):
            loader.start("video", True, "session/raw", 30)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                loader.stop()
                loader.stop()

        self.assertEqual(output.getvalue().count("serial=cam-a: 17 frames"), 1)
        self.assertEqual(output.getvalue().count("Total captured frames: 17"), 1)

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
            server.close()

    def test_server_separates_aravis_prepare_from_activation(self):
        loader = FakeTwoPhaseLoader()
        server = camera_server_daemon(backend="aravis", loader=loader, start_threads=False)
        try:
            server.execute_command({"action": "register", "controller_name": "main"})
            prepare = server.execute_command(
                {
                    "action": "prepare",
                    "controller_name": "main",
                    "mode": "video",
                    "syncMode": True,
                    "save_path": "session/raw",
                    "fps": 30,
                }
            )
            self.assertEqual(prepare, {"status": "ok", "msg": "prepared"})
            self.assertEqual(
                loader.calls,
                [("prepare", "video", True, "session/raw", 30)],
            )

            start = server.execute_command(
                {"action": "start", "controller_name": "main"}
            )
            self.assertEqual(start, {"status": "ok", "msg": "ready"})
            self.assertEqual(loader.calls[-1], ("activate",))
        finally:
            server.close()

    def test_server_close_ends_loader_and_is_idempotent(self):
        loader = FakeLoader()
        server = camera_server_daemon(loader=loader, start_threads=False)

        server.close()
        server.close()

        self.assertEqual(loader.calls, [("end",)])
        self.assertEqual(server.state, "closed")


if __name__ == "__main__":
    unittest.main()
