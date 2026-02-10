import argparse
import time
from typing import Optional

import numpy as np
import serial

from paradex.io.robot_controller.inspire_f1_controller import (
    ADDR_ACTUAL_ANGLE,
    _build_read_frame,
    _parse_read_resp,
    _read_frame,
    _i16_from_le,
)


def _read_group_6(ser: serial.Serial, hand_id: int, addr: int) -> Optional[np.ndarray]:
    frame = _build_read_frame(hand_id, addr, 12)
    ser.write(frame)
    resp = _read_frame(ser, 0.5)
    if resp is None:
        return None
    data = _parse_read_resp(resp)
    if data is None or len(data) != 12:
        return None
    out = np.zeros(6, dtype=np.int16)
    for i in range(6):
        out[i] = _i16_from_le(data[i * 2:i * 2 + 2])
    return out


def _scan_ids(ser: serial.Serial, max_id: int, tries: int, sleep_s: float) -> int:
    found = []
    for hid in range(1, max_id + 1):
        ok = False
        for _ in range(tries):
            ser.reset_input_buffer()
            raw_angle = _read_group_6(ser, hid, ADDR_ACTUAL_ANGLE)
            if raw_angle is not None:
                ok = True
                break
            time.sleep(sleep_s)
        if ok:
            found.append(hid)
    if not found:
        print("[FAIL] no hand_id responded in scan range.")
        return 1
    print("[OK] responded hand_id(s): " + ", ".join(str(h) for h in found))
    return 0


def _scan_baud_and_ids(port: str, baud_list: str, max_id: int, tries: int, sleep_s: float) -> int:
    bauds = []
    for b in baud_list.split(","):
        b = b.strip()
        if not b:
            continue
        try:
            bauds.append(int(b))
        except ValueError:
            print(f"[WARN] skip invalid baud: {b}")
    if not bauds:
        print("[FAIL] no valid baud in list.")
        return 2

    any_ok = False
    for baud in bauds:
        try:
            ser = serial.Serial(port, baud, timeout=0.05)
        except Exception as exc:
            print(f"[WARN] baud {baud}: serial open failed: {exc}")
            continue
        try:
            rc = _scan_ids(ser, max_id, tries, sleep_s)
            if rc == 0:
                any_ok = True
            else:
                print(f"[INFO] baud {baud}: no response")
        finally:
            try:
                ser.close()
            except Exception:
                pass
    return 0 if any_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspire F1 connection diagnostic")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1, help="Hand ID (default: 1)")
    parser.add_argument("--baud-rate", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--tries", type=int, default=5, help="Number of read attempts")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep between attempts (sec)")
    parser.add_argument("--scan-ids", type=int, default=0, help="Scan hand IDs 1..N and report responders")
    parser.add_argument(
        "--scan-baud-ids",
        type=str,
        default="",
        help="Comma-separated baud list to scan with IDs 1..N, e.g. 115200,57600,38400",
    )
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud_rate, timeout=0.05)
    except Exception as exc:
        print(f"[FAIL] serial open failed: {exc}")
        return 2

    try:
        if args.scan_baud_ids:
            return _scan_baud_and_ids(args.port, args.scan_baud_ids, args.scan_ids or 10, args.tries, args.sleep)

        if args.scan_ids and args.scan_ids > 0:
            rc = _scan_ids(ser, args.scan_ids, args.tries, args.sleep)
            return rc

        ok = False
        last = None
        for _ in range(args.tries):
            ser.reset_input_buffer()
            raw_angle = _read_group_6(ser, args.hand_id, ADDR_ACTUAL_ANGLE)
            if raw_angle is not None:
                ok = True
                last = raw_angle
                break
            time.sleep(args.sleep)

        if not ok:
            print("[FAIL] no response. check port/hand_id/baud/cable/power.")
            return 1

        print("[OK] response received.")
        print(f"raw_angle: {last.tolist()}")
        return 0
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
