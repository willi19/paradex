import time
from threading import Event, Thread


class MiddlePedalState:
    """Expose the Stream Deck middle pedal as a teleoperation state.

    State 0 means the middle pedal is held. State 1 is the fail-safe state used
    when the pedal is released or the device is disconnected.
    """

    MIDDLE_KEY = 1

    def __init__(
        self,
        device=None,
        poll_hz=100,
        connection_check_hz=2,
        verbose=True,
    ):
        if connection_check_hz <= 0:
            raise ValueError("connection_check_hz must be positive")

        self._pressed = Event()
        self._connected = Event()
        self._connection_monitor_stop = Event()
        self._connection_check_interval = 1.0 / float(connection_check_hz)
        self._verbose = verbose
        self._device = device if device is not None else self._find_device()
        self._device.open()
        self._device.set_poll_frequency(poll_hz)
        self._device.set_key_callback(self._on_key_change)
        self._connected.set()
        self._connection_monitor = Thread(
            target=self._monitor_connection,
            daemon=True,
        )
        self._connection_monitor.start()

        if self._verbose:
            print("Middle pedal control ready: hold=state 0, release=state 1")

    @staticmethod
    def _find_device():
        try:
            from StreamDeck.DeviceManager import DeviceManager
        except ImportError as exc:
            raise RuntimeError(
                "The streamdeck package is required for bimanual pedal control."
            ) from exc

        devices = [
            device
            for device in DeviceManager().enumerate()
            if device.deck_type() == "Stream Deck Pedal"
        ]
        if not devices:
            raise RuntimeError(
                "Stream Deck Pedal not found (expected USB VID:PID 0fd9:0086)."
            )
        return devices[0]

    def _on_key_change(self, _device, key, pressed):
        if key != self.MIDDLE_KEY:
            return

        if pressed:
            self._pressed.set()
        else:
            self._pressed.clear()

    def _mark_disconnected(self):
        self._connected.clear()
        self._pressed.clear()

    def _check_physical_connection(self):
        try:
            connected = self._device.connected()
        except Exception:
            connected = False

        if not connected:
            self._mark_disconnected()
        return connected

    def _monitor_connection(self):
        next_physical_check = time.monotonic() + self._connection_check_interval
        while not self._connection_monitor_stop.wait(0.01):
            try:
                is_open = self._device.is_open()
            except Exception:
                is_open = False

            if not is_open:
                self._mark_disconnected()
                return

            now = time.monotonic()
            if now >= next_physical_check:
                if not self._check_physical_connection():
                    return
                next_physical_check = now + self._connection_check_interval

    def get_state(self):
        if self._connected.is_set() and self._pressed.is_set():
            return 0
        return 1

    def close(self):
        self._connection_monitor_stop.set()
        self._mark_disconnected()
        try:
            if self._device.is_open():
                self._device.close()
        finally:
            self._connection_monitor.join(timeout=1.0)


class LeftRightPedalState(MiddlePedalState):
    """Expose the outer Stream Deck Pedal switches as a signed input.

    The outer switches provide a signed interpolation direction and the
    inherited middle switch remains available as a deadman state.  This lets
    one physical Stream Deck Pedal control both a hand interpolation and an
    xArm without opening the USB device twice.
    """

    LEFT_KEY = 0
    RIGHT_KEY = 2

    def __init__(self, *args, **kwargs):
        self._left_pressed = Event()
        self._right_pressed = Event()
        kwargs.setdefault("verbose", False)
        super().__init__(*args, **kwargs)
        if self._verbose:
            print("Outer pedal control ready: left=+, right=-, both/released=hold")

    def _on_key_change(self, _device, key, pressed):
        if key == self.MIDDLE_KEY:
            super()._on_key_change(_device, key, pressed)
            return
        pressed_event = (
            self._left_pressed
            if key == self.LEFT_KEY
            else self._right_pressed
            if key == self.RIGHT_KEY
            else None
        )
        if pressed_event is None:
            return
        if pressed:
            pressed_event.set()
        else:
            pressed_event.clear()

    def _mark_disconnected(self):
        super()._mark_disconnected()
        self._left_pressed.clear()
        self._right_pressed.clear()

    def get_direction(self):
        """Return +1 for left, -1 for right, and 0 for safe hold."""
        if not self._connected.is_set():
            return 0
        left = self._left_pressed.is_set()
        right = self._right_pressed.is_set()
        if left == right:
            return 0
        return 1 if left else -1
