import numpy as np
import pytest

from paradex.io.tactile.human_tactile_recorder import HumanTactileRecorder


def test_human_tactile_recorder_parses_26_value_packet():
    recorder = HumanTactileRecorder(port="unused", baud_rate=115200)
    packet = ",".join(str(value) for value in range(26))

    assert recorder._parse_line(packet) == list(range(26))


def test_human_tactile_recorder_reader_stores_26_value_packet():
    recorder = HumanTactileRecorder(port="unused", baud_rate=115200)

    class SerialWithOnePacket:
        def readline(self):
            recorder.exit_event.set()
            return b"0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25\n"

    recorder.ser = SerialWithOnePacket()
    recorder._read_loop()

    _, _, sample = recorder.get_latest()
    assert np.array_equal(sample, np.arange(26, dtype=np.int32))


@pytest.mark.parametrize(
    "packet",
    [
        ",".join(str(value) for value in range(25)),
        ",".join(str(value) for value in range(27)),
        ",".join(["0"] * 25 + ["not-an-int"]),
    ],
)
def test_human_tactile_recorder_rejects_invalid_packet(packet):
    recorder = HumanTactileRecorder(port="unused", baud_rate=115200)

    with pytest.raises(ValueError):
        recorder._parse_line(packet)


def test_empty_recording_uses_26_channel_array(tmp_path):
    recorder = HumanTactileRecorder(port="unused", baud_rate=115200)
    recorder.save_path = tmp_path
    recorder.data = {"time": [], "tactile": []}

    recorder.stop()

    assert np.load(tmp_path / "tactile.npy").shape == (0, 26)


def test_realtime_plot_is_refreshed_by_the_owner_thread():
    recorder = HumanTactileRecorder(port="unused", baud_rate=115200, plot_realtime=True)
    recorder.plot_enabled = True
    recorder.latest_sample_id = 1
    recorder.latest_timestamp = 0.0
    recorder.latest_sample = np.arange(26, dtype=np.int32)
    plotted = []
    recorder._update_plot = lambda sample_id, sample: plotted.append((sample_id, sample))

    recorder.refresh_plot()
    recorder.refresh_plot()

    assert recorder.plot_thread is None
    assert len(plotted) == 1
    assert plotted[0][0] == 1
