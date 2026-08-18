from paradex.io.streamdeck_pedal import LeftRightPedalState, MiddlePedalState


class FakePedal:
    def __init__(self):
        self.callback = None
        self.connected_value = True
        self.connected_calls = 0
        self.opened = False
        self.poll_hz = None

    def open(self):
        self.opened = True

    def set_poll_frequency(self, poll_hz):
        self.poll_hz = poll_hz

    def set_key_callback(self, callback):
        self.callback = callback

    def connected(self):
        self.connected_calls += 1
        return self.connected_value

    def is_open(self):
        return self.opened

    def close(self):
        self.opened = False


def test_middle_pedal_is_a_fail_safe_deadman_switch():
    device = FakePedal()
    pedal = MiddlePedalState(
        device=device,
        poll_hz=100,
        connection_check_hz=0.01,
        verbose=False,
    )

    assert device.opened
    assert device.poll_hz == 100
    assert pedal.get_state() == 1

    device.callback(device, 0, True)
    assert pedal.get_state() == 1

    device.callback(device, 1, True)
    assert pedal.get_state() == 0

    device.callback(device, 1, False)
    assert pedal.get_state() == 1

    device.callback(device, 1, True)
    device.connected_value = False
    pedal._check_physical_connection()
    assert pedal.get_state() == 1

    pedal.close()
    assert not device.opened


def test_get_state_does_not_query_usb_connection():
    device = FakePedal()
    pedal = MiddlePedalState(
        device=device,
        connection_check_hz=0.01,
        verbose=False,
    )

    device.callback(device, 1, True)
    for _ in range(1000):
        assert pedal.get_state() == 0

    assert device.connected_calls == 0
    pedal.close()


def test_outer_pedals_report_signed_motion_and_fail_safe_hold():
    device = FakePedal()
    pedal = LeftRightPedalState(
        device=device,
        connection_check_hz=0.01,
        verbose=False,
    )

    assert pedal.get_direction() == 0
    device.callback(device, 0, True)
    assert pedal.get_direction() == 1
    device.callback(device, 2, True)
    assert pedal.get_direction() == 0
    device.callback(device, 0, False)
    assert pedal.get_direction() == -1
    device.connected_value = False
    pedal._check_physical_connection()
    assert pedal.get_direction() == 0
    pedal.close()


def test_outer_pedal_reader_also_exposes_the_shared_middle_deadman_switch():
    device = FakePedal()
    pedal = LeftRightPedalState(
        device=device,
        connection_check_hz=0.01,
        verbose=False,
    )

    assert pedal.get_state() == 1
    device.callback(device, 1, True)
    assert pedal.get_state() == 0
    device.callback(device, 1, False)
    assert pedal.get_state() == 1
    pedal.close()
