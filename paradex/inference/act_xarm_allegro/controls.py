"""Testable deadman state plus a global keyboard listener."""

from __future__ import annotations

from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class DeadmanSnapshot:
    held: bool
    aborted: bool
    rearm_requested: bool
    enable_generation: int


class DeadmanState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held = False
        self._aborted = False
        self._rearm_requested = False
        self._enable_generation = 0

    def press(self, key: str) -> None:
        with self._lock:
            if key == "esc":
                self._held = False
                self._aborted = True
            elif key == "r" and self._aborted:
                self._rearm_requested = True

    def release(self, key: str) -> None:
        del key

    def consume_rearm(self, checks_passed: bool) -> bool:
        with self._lock:
            requested = self._rearm_requested
            self._rearm_requested = False
            if requested and checks_passed:
                self._aborted = False
                self._enable_generation += 1
                return True
            return False

    def snapshot(self) -> DeadmanSnapshot:
        with self._lock:
            return DeadmanSnapshot(
                self._held,
                self._aborted,
                self._rearm_requested,
                self._enable_generation,
            )


class KeyboardDeadman:
    def __init__(self, state: DeadmanState | None = None) -> None:
        self.state = state or DeadmanState()
        self._listener = None

    @staticmethod
    def _name(key: object) -> str | None:
        from pynput import keyboard

        if key == keyboard.Key.esc:
            return "esc"
        char = getattr(key, "char", None)
        return "r" if isinstance(char, str) and char.lower() == "r" else None

    def start(self) -> None:
        from pynput import keyboard

        def on_press(key: object) -> None:
            name = self._name(key)
            if name is not None:
                self.state.press(name)

        def on_release(key: object) -> None:
            name = self._name(key)
            if name is not None:
                self.state.release(name)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener.join(timeout=1.0)
            self._listener = None
