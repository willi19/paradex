import unittest
from unittest.mock import patch

from paradex.io.camera_system.aravis_timestamp import AravisTimestampMonitor


class FakeBuffer:
    def __init__(self, frame_id=17, status="success"):
        self.frame_id = frame_id
        self.status = status

    def get_status(self):
        return self.status

    def get_frame_id(self):
        return self.frame_id


class FakeStream:
    def __init__(self):
        self.input = []
        self.output = []
        self.timeout_us = None

    def push_buffer(self, buffer):
        self.input.append(buffer)

    def try_pop_buffer(self):
        return self.output.pop(0) if self.output else None

    def timeout_pop_buffer(self, timeout_us):
        self.timeout_us = timeout_us
        return self.output.pop(0) if self.output else None

    def pop_buffer(self):
        return self.output.pop(0) if self.output else None


class FakeDevice:
    def __init__(self):
        self.values = {}

    def __getattr__(self, name):
        if name.startswith("set_") and name.endswith("_feature_value"):
            return lambda feature, value: self.values.__setitem__(feature, value)
        if name.startswith("get_") and name.endswith("_feature_value"):
            return lambda feature: self.values[feature]
        raise AttributeError(name)


class FakeCamera:
    def __init__(self):
        self.device = FakeDevice()
        self.stream = FakeStream()
        self.started = 0
        self.stopped = 0
        self.mode = None

    def get_device(self):
        return self.device

    def set_acquisition_mode(self, mode):
        self.mode = mode

    def create_stream(self, callback, data):
        self.create_args = (callback, data)
        return self.stream

    def get_payload(self):
        return 1024

    def start_acquisition(self):
        self.started += 1

    def stop_acquisition(self):
        self.stopped += 1


class FakeAravis:
    class AcquisitionMode:
        CONTINUOUS = "continuous"

    class BufferStatus:
        SUCCESS = "success"

    class Buffer:
        @staticmethod
        def new_allocate(payload):
            return ("buffer", payload)

    def __init__(self):
        self.session = FakeCamera()
        self.Camera = self

    def update_device_list(self):
        pass

    def get_n_devices(self):
        return 1

    def get_device_serial_nbr(self, index):
        return "22684253"

    def get_device_id(self, index):
        return "FLIR-Blackfly-S-22684253"

    def new(self, device_id):
        self.opened_id = device_id
        return self.session


class AravisTimestampTests(unittest.TestCase):
    def make_monitor(self):
        aravis = FakeAravis()
        with patch(
            "paradex.io.camera_system.aravis_timestamp._load_aravis",
            return_value=aravis,
        ), patch(
            "paradex.io.camera_system.aravis_timestamp.get_camera_config",
            return_value={"22684253": {"gain": 2.0, "exposure": 2200.0}},
        ):
            monitor = AravisTimestampMonitor("22684253", buffer_count=3)
        return monitor, aravis

    def test_configures_trigger_and_reuses_small_buffer_pool(self):
        monitor, aravis = self.make_monitor()

        self.assertEqual(aravis.opened_id, "FLIR-Blackfly-S-22684253")
        self.assertEqual(aravis.session.mode, "continuous")
        self.assertEqual(aravis.session.device.values["TriggerMode"], "On")
        self.assertEqual(aravis.session.device.values["TriggerSource"], "Line0")
        self.assertEqual(
            aravis.session.stream.input,
            [("buffer", 1024), ("buffer", 1024), ("buffer", 1024)],
        )
        self.assertIs(monitor.stream, aravis.session.stream)

    def test_returns_only_frame_metadata_and_recycles_buffer(self):
        monitor, aravis = self.make_monitor()
        monitor.start()
        frame = FakeBuffer(frame_id=91)
        aravis.session.stream.output.append(frame)

        result = monitor.get_timestamp(timeout_ms=500)

        self.assertEqual(result["frameID"], 91)
        self.assertIn("pc_time", result)
        self.assertEqual(aravis.session.stream.timeout_us, 500000)
        self.assertIs(aravis.session.stream.input[-1], frame)
        monitor.stop()
        self.assertEqual(aravis.session.started, 1)
        self.assertEqual(aravis.session.stopped, 1)

    def test_timeout_is_normal_while_waiting_for_utg(self):
        monitor, _ = self.make_monitor()
        monitor.start()
        self.assertIsNone(monitor.get_timestamp(timeout_ms=10))


if __name__ == "__main__":
    unittest.main()
