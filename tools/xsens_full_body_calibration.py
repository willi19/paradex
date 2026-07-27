#!/usr/bin/env python3
"""Run an Xsens MVN full-body calibration through the XME Python SDK.

This follows the workflow in the SDK's ``calibrate_and_record`` example while
making the full-body configuration and the essential body dimensions explicit.

Example:
    python tools/xsens_full_body_calibration.py \
        --height 1.78 \
        --foot-size 0.27 \
        --pose Npose \
        --channel 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from threading import Event, Lock
from typing import Any


QUALITY_EXIT_OK = {"good", "acceptable"}


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def awinda_channel(value: str) -> int:
    parsed = int(value)
    if not 11 <= parsed <= 25:
        raise argparse.ArgumentTypeError("must be in the range 11-25")
    return parsed


def positive_seconds(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate an Xsens MVN suit in the FullBody configuration."
    )
    parser.add_argument(
        "--height",
        type=positive_float,
        required=True,
        metavar="METERS",
        help="subject body height in meters, for example 1.78",
    )
    parser.add_argument(
        "--foot-size",
        type=positive_float,
        required=True,
        metavar="METERS",
        help="subject foot length in meters, for example 0.27",
    )
    parser.add_argument(
        "--pose",
        default="Npose",
        choices=("Npose", "Tpose"),
        help="dynamic calibration procedure (default: Npose)",
    )
    parser.add_argument(
        "--channel",
        type=awinda_channel,
        default=15,
        metavar="11-25",
        help="Awinda radio channel; ignored by non-Awinda systems (default: 15)",
    )
    parser.add_argument(
        "--license-host",
        default="127.0.0.1",
        help="Xsens License Manager host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=positive_seconds,
        default=120.0,
        metavar="SECONDS",
        help="maximum time to wait for all required trackers (default: 120)",
    )
    parser.add_argument(
        "--calibration-timeout",
        type=positive_seconds,
        default=180.0,
        metavar="SECONDS",
        help="maximum time to wait for calibration completion (default: 180)",
    )
    parser.add_argument(
        "--save-calibration",
        type=Path,
        metavar="PATH",
        help="optionally save the completed calibration to this file",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="start calibration without waiting for Enter",
    )
    return parser.parse_args(argv)


def as_text(value: Any) -> str:
    return str(value)


def make_callbacks(xme: Any, channel: int) -> Any:
    section_names = {
        xme.XmeCalibrationRecordingSection_Static: "static pose",
        xme.XmeCalibrationRecordingSection_Dynamic: "dynamic movement",
        xme.XmeCalibrationRecordingSection_RightArmRaise: "right arm raise",
        xme.XmeCalibrationRecordingSection_LeftArmRaise: "left arm raise",
        xme.XmeCalibrationRecordingSection_RightLegRaise: "right leg raise",
        xme.XmeCalibrationRecordingSection_LeftLegRaise: "left leg raise",
    }

    class CalibrationCallbacks(xme.XmeCallback):
        def __init__(self) -> None:
            super().__init__()
            self.hardware_ready = Event()
            self.hardware_disconnected = Event()
            self.calibration_processed = Event()
            self.calibration_complete = Event()
            self.calibration_aborted = Event()
            self.current_section: int | None = None
            self.last_hardware_error = ""
            self._lock = Lock()

        def reset_calibration_state(self) -> None:
            self.calibration_processed.clear()
            self.calibration_complete.clear()
            self.calibration_aborted.clear()
            with self._lock:
                self.current_section = None

        def onHardwareReady(self, dev: Any) -> None:
            del dev
            print("\n[hardware] Full-body hardware is ready.")
            self.hardware_ready.set()

        def onHardwareDisconnected(self, dev: Any) -> None:
            del dev
            print("\n[hardware] Hardware disconnected.")
            self.hardware_disconnected.set()
            self.hardware_ready.clear()

        def onHardwareError(self, dev: Any) -> None:
            suit = dev.status().suitStatus()
            message = as_text(suit.m_hardwareStatusText)
            if message != self.last_hardware_error:
                print(f"\n[hardware] {message}", file=sys.stderr)
                missing = list(suit.m_missingSensors)
                if missing:
                    names = ", ".join(
                        f"{segment_id} ({as_text(dev.segmentName(segment_id))})"
                        for segment_id in missing
                    )
                    print(f"[hardware] Missing trackers: {names}", file=sys.stderr)
                self.last_hardware_error = message

            master_id = suit.m_masterDevice.m_deviceId
            if channel and master_id.isAwindaX() and suit.m_wirelessChannel != channel:
                print(f"[hardware] Setting Awinda radio channel to {channel}.")
                xme.XmeControl.setRadioChannel(master_id, channel)

        def onCalibrationProcessed(self, dev: Any) -> None:
            del dev
            print("\n[calibration] Recorded data processed.")
            self.calibration_processed.set()

        def onCalibrationComplete(self, dev: Any) -> None:
            del dev
            print("\n[calibration] Calibration update complete.")
            self.calibration_complete.set()

        def onCalibrationAborted(self, dev: Any) -> None:
            del dev
            print("\n[calibration] Calibration aborted.", file=sys.stderr)
            self.calibration_aborted.set()

        def onCalibrationRecordingSectionChanged(
            self, dev: Any, calibration_section: int
        ) -> None:
            del dev
            with self._lock:
                self.current_section = calibration_section
            name = section_names.get(calibration_section, str(calibration_section))
            print(f"\n[calibration] Stage: {name}")

        def onProgressUpdate(self, dev: Any, percentage: int, category: Any) -> None:
            del dev
            print(f"\n[calibration] {as_text(category)}: {percentage}%")

    return CalibrationCallbacks()


def wait_for_hardware(
    xme: Any,
    control: Any,
    callbacks: Any,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    obr_disable_requested = False

    while time.monotonic() < deadline:
        status = control.status()
        if status.isConnected() and status.isInObrMode():
            if not obr_disable_requested:
                master_id = status.suitStatus().m_masterDevice.m_deviceId
                print("[hardware] Disabling on-body recording mode.")
                control.setObrMode(master_id, xme.XOBRM_Disabled)
                obr_disable_requested = True
        elif status.isConnected() and callbacks.hardware_ready.is_set():
            return
        time.sleep(0.05)

    detail = callbacks.last_hardware_error or "no complete MVN system was detected"
    raise TimeoutError(f"hardware was not ready within {timeout:g}s: {detail}")


def set_body_dimensions(
    control: Any,
    callbacks: Any,
    height: float,
    foot_size: float,
) -> None:
    supported = {as_text(label) for label in control.bodyDimensionLabelList()}
    required = {"bodyHeight", "footSize"}
    missing = sorted(required - supported)
    if missing:
        raise RuntimeError(
            "SDK configuration does not expose required body dimensions: "
            + ", ".join(missing)
        )

    callbacks.reset_calibration_state()
    print(
        "[calibration] Applying body dimensions: "
        f"height={height:.3f} m, footSize={foot_size:.3f} m"
    )
    control.setBodyDimension("bodyHeight", height)
    control.setBodyDimension("footSize", foot_size)

    if not callbacks.calibration_complete.wait(timeout=10.0):
        raise TimeoutError("body-dimension update did not complete within 10s")
    callbacks.reset_calibration_state()


def print_calibration_instructions(control: Any, pose: str) -> None:
    times_ms = [int(value) for value in control.calibrationRecordingTimesRemaining()]
    print(f"\nCalibration procedure: {pose}")
    if not times_ms:
        print("  The SDK did not report stage durations.")
        return

    labels = ("hold the calibration pose", "move naturally")
    for index, duration_ms in enumerate(times_ms):
        label = labels[index] if index < len(labels) else f"stage {index + 1}"
        print(f"  {index + 1}. {label}: {duration_ms / 1000:.1f} seconds")


def run_calibration(
    xme: Any,
    control: Any,
    callbacks: Any,
    pose: str,
    timeout: float,
    assume_yes: bool,
) -> Any:
    available = [as_text(label) for label in control.calibrationLabelList()]
    if pose not in available:
        raise RuntimeError(
            f"{pose} is unavailable; SDK reported: {', '.join(available)}"
        )

    control.initializeCalibration(pose)
    print_calibration_instructions(control, pose)

    if not assume_yes:
        input("\nAssume the requested pose, then press Enter to start... ")

    callbacks.reset_calibration_state()
    control.startCalibration()
    deadline = time.monotonic() + timeout
    last_report: tuple[int | None, int | None] | None = None

    while time.monotonic() < deadline:
        if callbacks.calibration_aborted.is_set():
            raise RuntimeError("XME aborted the calibration")
        if (
            callbacks.calibration_complete.is_set()
            and not control.status().isCalibrating()
        ):
            return control.calibrationResult(pose)

        with callbacks._lock:
            section = callbacks.current_section
        remaining = [
            int(value) for value in control.calibrationRecordingTimesRemaining()
        ]
        seconds = (remaining[0] + 999) // 1000 if remaining else None
        report = (section, seconds)
        if report != last_report:
            if seconds is None:
                print("\r[calibration] Processing...                    ", end="")
            else:
                print(
                    f"\r[calibration] Current stage: {seconds:>3}s remaining",
                    end="",
                    flush=True,
                )
            last_report = report
        time.sleep(0.1)

    control.abortCalibration()
    raise TimeoutError(f"calibration did not complete within {timeout:g}s")


def quality_name(xme: Any, quality: int) -> str:
    return {
        xme.XCalQ_Unknown: "unknown",
        xme.XCalQ_Good: "good",
        xme.XCalQ_Acceptable: "acceptable",
        xme.XCalQ_Poor: "poor",
        xme.XCalQ_Failed: "failed",
    }.get(quality, f"unexpected({quality})")


def cleanup_xme(xme: Any, control: Any, callbacks: Any) -> None:
    if control is None:
        return

    try:
        control.setScanMode(False)
    except Exception:
        pass

    try:
        if control.status().isConnected():
            control.disconnectHardware()
            deadline = time.monotonic() + 5.0
            while control.status().isConnected() and time.monotonic() < deadline:
                time.sleep(0.05)
    except Exception:
        pass

    if callbacks is not None:
        try:
            control.removePyCallbackHandler(callbacks)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import xme
    except ImportError as exc:
        print(
            "Unable to import xme. Install the wheel matching this Python version:\n"
            "  python -m pip install "
            "./xme-2026.4.0-cp312-none-manylinux_2_31_x86_64.whl",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    license_handle = None
    control = None
    callbacks = None

    try:
        version = as_text(xme.xmeGetDllVersion().toSimpleString())
        print(f"XME runtime version: {version}")

        license_handle = xme.XmeLicense(args.license_host)
        if not xme.XmeLicense.isConstructed():
            raise RuntimeError(
                "no XME license connection; install/run Xsens License Manager "
                "and make an SDK-Dev license available"
            )
        print(f"XME license: {as_text(xme.XmeLicense.getCurrentLicense())}")

        control = xme.XmeControl()
        callbacks = make_callbacks(xme, args.channel)
        control.addCallbackHandler(callbacks)

        print("[hardware] Selecting FullBody configuration.")
        control.setConfiguration("FullBody")
        print("[hardware] Scanning for the MVN system...")
        control.setScanMode(True)
        wait_for_hardware(
            xme=xme,
            control=control,
            callbacks=callbacks,
            timeout=args.connect_timeout,
        )
        control.setScanMode(False)

        set_body_dimensions(
            control=control,
            callbacks=callbacks,
            height=args.height,
            foot_size=args.foot_size,
        )
        result = run_calibration(
            xme=xme,
            control=control,
            callbacks=callbacks,
            pose=args.pose,
            timeout=args.calibration_timeout,
            assume_yes=args.yes,
        )

        quality = quality_name(xme, result.m_quality)
        warnings = [as_text(warning) for warning in result.m_warnings]
        print(f"\nCalibration quality: {quality}")
        for warning in warnings:
            print(f"  warning: {warning}")

        if args.save_calibration is not None:
            destination = args.save_calibration.expanduser().resolve()
            if not destination.parent.is_dir():
                raise FileNotFoundError(
                    f"calibration output directory does not exist: {destination.parent}"
                )
            control.saveCurrentCalibration(str(destination))
            print(f"Saved calibration: {destination}")

        if quality not in QUALITY_EXIT_OK:
            print(
                "Calibration should be repeated before capture.",
                file=sys.stderr,
            )
            return 3
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted; aborting calibration.", file=sys.stderr)
        if control is not None:
            try:
                if control.status().isCalibrating():
                    control.abortCalibration()
            except Exception:
                pass
        return 130
    except Exception as exc:
        print(f"\nXME calibration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        cleanup_xme(xme, control, callbacks)
        callbacks = None
        control = None
        license_handle = None
        xme.xmeTerminate()


if __name__ == "__main__":
    raise SystemExit(main())
