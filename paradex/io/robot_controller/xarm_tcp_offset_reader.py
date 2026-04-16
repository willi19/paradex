import time

import numpy as np

from xarm.wrapper import XArmAPI


class XArmTcpOffsetReader:
    """Connect to xArm and read TCP offset into `self.tcp_offset`."""

    def __init__(self, ip, auto_enable=False):
        self.xarm_ip_address = ip
        self.arm = None
        self.tcp_offset = None

        self.connect(auto_enable=auto_enable)
        self.read_tcp_offset()

    def connect(self, auto_enable=False):
        self.arm = XArmAPI(self.xarm_ip_address, report_type="devlop")

        if self.arm.has_err_warn:
            self.arm.clean_warn()
            self.arm.clean_error()

        if auto_enable:
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(state=0)
            time.sleep(0.1)

    def _normalize_offset_result(self, result):
        # Typical xArm SDK pattern: (code, data)
        if isinstance(result, tuple) and len(result) >= 2:
            code, data = result[0], result[1]
            if code == 0 and data is not None:
                return np.asarray(data, dtype=np.float64)

        # Some APIs may directly return a list/ndarray
        if isinstance(result, (list, tuple, np.ndarray)):
            arr = np.asarray(result, dtype=np.float64)
            if arr.size >= 6:
                return arr

        return None

    def read_tcp_offset(self):
        # SDK 1.13.x exposes tcp offset as an attribute.
        if hasattr(self.arm, "tcp_offset"):
            try:
                offset = np.asarray(self.arm.tcp_offset, dtype=np.float64)
                if offset.size >= 6:
                    self.tcp_offset = offset[:6]
                    return self.tcp_offset
            except Exception:
                pass

        # Fallback for older/newer APIs exposing explicit getter methods.
        candidates = [
            "get_tcp_offset",
            "get_tcp_pose_offset",
            "get_tool_offset",
        ]
        for name in candidates:
            if not hasattr(self.arm, name):
                continue
            fn = getattr(self.arm, name)
            try:
                offset = self._normalize_offset_result(fn())
            except Exception:
                offset = None
            if offset is not None:
                self.tcp_offset = offset[:6]
                return self.tcp_offset

        raise RuntimeError(
            "Failed to read TCP offset: neither `arm.tcp_offset` nor known getter methods worked."
        )

    def end(self):
        if self.arm is not None:
            self.arm.disconnect()
            self.arm = None


if __name__ == "__main__":
    # Quick manual test
    from paradex.utils.system import network_info

    reader = None
    try:
        ip = network_info["xarm"]["param"]["ip"]
        reader = XArmTcpOffsetReader(ip=ip, auto_enable=False)
        print("tcp_offset:", np.array2string(reader.tcp_offset, precision=6))
    finally:
        if reader is not None:
            reader.end()
