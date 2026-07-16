import os
import time


class UTGE900:
    """UNI-T UTG900E controller using its documented SCPI interface.

    The device is accessed through the Linux USBTMC character device.  Signal
    programming intentionally matches ParaOffice's known-good hardware trigger
    contract: High-Z load, 0--5 V square wave, requested frequency and 50 %
    duty cycle.
    """

    VERIFY_NUMERIC_TOL = 0.01
    LOAD_OHMS = 10000
    AMPLITUDE_VPP = 5.0
    OFFSET_V = 2.5
    DUTY_CYCLE = 50.0

    def __init__(self, addr):
        self.device = addr
        if not os.path.exists(addr):
            raise FileNotFoundError(f"Device {addr} not found")
        if not os.access(addr, os.R_OK | os.W_OK):
            raise PermissionError(
                f"No permission for {addr}. Configure a udev rule granting read/write access."
            )
        self.ch = [False, False]

    def write(self, cmd):
        """Write one newline-terminated SCPI command."""

        with open(self.device, "wb") as device:
            device.write((cmd + "\n").encode("ascii"))
            device.flush()

    def query(self, cmd, strip=False):
        """Write one SCPI query and return its response."""

        with open(self.device, "w+b", buffering=0) as device:
            device.write((cmd + "\n").encode("ascii"))
            time.sleep(0.05)
            response = device.read(4096).decode("ascii").strip()
        return response.rstrip() if strip else response

    def _write_verify(self, cmd, value):
        """Write a signal parameter and verify it through the query form."""

        self.write(f"{cmd} {value}")
        raw = self.query(f"{cmd}?").strip()
        if isinstance(value, str):
            expected = value.upper()
            actual = raw.upper()
            valid = actual.startswith(expected[:3]) or expected.startswith(actual[:3])
        else:
            try:
                actual = float(raw)
            except ValueError as exc:
                raise RuntimeError(
                    f"UTG900E {cmd} returned non-numeric value {raw!r}"
                ) from exc
            valid = abs(actual - float(value)) <= self.VERIFY_NUMERIC_TOL
        if not valid:
            raise RuntimeError(
                f"UTG900E {cmd} write failed: requested {value!r}, device reports {raw!r}"
            )

    @staticmethod
    def _waveform_name(wave):
        names = {
            "square": "SQUare",
            "squ": "SQUare",
            "sine": "SINe",
            "sin": "SINe",
        }
        normalized = str(wave).strip()
        return names.get(normalized.lower(), normalized)

    def generate(self, ch=1, wave="square", freq=30, amp=AMPLITUDE_VPP):
        """Program a ParaOffice-compatible trigger signal without enabling it."""

        ch = int(ch)
        if ch not in (1, 2):
            raise ValueError(f"Unsupported UTG900E channel: {ch}")
        prefix = f":CHANnel{ch}"
        self._write_verify(f"{prefix}:LOAD", self.LOAD_OHMS)
        self._write_verify(f"{prefix}:BASE:WAVE", self._waveform_name(wave))
        self._write_verify(f"{prefix}:BASE:AMPLitude", float(amp))
        self._write_verify(f"{prefix}:BASE:OFFSet", self.OFFSET_V)
        self._write_verify(f"{prefix}:BASE:FREQuency", float(freq))
        self._write_verify(f"{prefix}:BASE:DUTY", self.DUTY_CYCLE)

    def start(self, fps, ch=1):
        """Program the trigger and enable the selected channel output."""

        ch = int(ch)
        if ch not in (1, 2):
            raise ValueError(f"Unsupported UTG900E channel: {ch}")
        if self.ch[ch - 1]:
            return
        prefix = f":CHANnel{ch}"
        self.write(f"{prefix}:OUTPut OFF")
        self.generate(ch=ch, wave="square", freq=fps, amp=self.AMPLITUDE_VPP)
        self.write(f"{prefix}:OUTPut ON")
        self.ch[ch - 1] = True
        time.sleep(0.1)

    def stop(self, ch=1):
        """Disable the selected channel output."""

        ch = int(ch)
        if ch not in (1, 2):
            raise ValueError(f"Unsupported UTG900E channel: {ch}")
        if not self.ch[ch - 1]:
            return
        self.write(f":CHANnel{ch}:OUTPut OFF")
        self.ch[ch - 1] = False

    def end(self):
        """Best-effort shutdown of every output enabled by this instance."""

        for ch in (1, 2):
            if self.ch[ch - 1]:
                self.stop(ch)

    def getName(self):
        return self.query("*IDN?")
