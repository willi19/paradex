import argparse
import datetime
import json
import os
import posixpath
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

sys.path.append(str(Path(__file__).parents[3]))


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _remote_shared_path(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").strip("/")
    return posixpath.join("shared_data", normalized)


def _selected_pc_list(args: argparse.Namespace) -> List[str]:
    from paradex.utils.system import get_pc_list

    if args.pc_list:
        pc_list = list(args.pc_list)
    else:
        excluded = set(args.exclude_pc or [])
        pc_list = [pc for pc in get_pc_list() if pc not in excluded]

    if not pc_list:
        raise ValueError("No capture PCs selected.")
    return pc_list


def _expected_cameras(pc_list: List[str]) -> List[str]:
    from paradex.utils.system import get_camera_list

    serials = []
    for pc in pc_list:
        try:
            serials.extend(get_camera_list(pc))
        except KeyError:
            print(f"[WARN] PC '{pc}' is not in pc config; expected camera list unknown.")
    return sorted(set(serials))


def _list_saved_images(images_dir: str) -> List[str]:
    if not os.path.isdir(images_dir):
        return []

    serials = []
    for name in os.listdir(images_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
            serials.append(stem)
    return sorted(serials)


def _wait_for_images(images_dir: str, expected: List[str], timeout_s: float) -> Tuple[List[str], List[str]]:
    deadline = time.time() + max(0.0, float(timeout_s))
    expected_set = set(expected)

    while True:
        saved = _list_saved_images(images_dir)
        if not expected_set or expected_set.issubset(saved):
            return saved, sorted(expected_set - set(saved))
        if time.time() >= deadline:
            return saved, sorted(expected_set - set(saved))
        time.sleep(0.25)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _capture_one_state(
    rcc: Any,
    args: argparse.Namespace,
    pc_list: List[str],
    expected_cameras: List[str],
    session_rel_path: str,
    session_abs_path: str,
    state_index: int,
    state_label: str,
    notes: str,
) -> dict:
    from paradex.calibration.utils import save_current_camparam

    state_id = f"{state_index:0{args.index_digits}d}"
    state_rel_path = os.path.join(session_rel_path, state_id)
    state_abs_path = os.path.join(session_abs_path, state_id)
    raw_rel_path = os.path.join(state_rel_path, "raw")
    raw_remote_path = _remote_shared_path(raw_rel_path)
    images_abs_dir = os.path.join(state_abs_path, "raw", "images")

    os.makedirs(state_abs_path, exist_ok=True)
    save_current_camparam(state_abs_path)

    print(f"[{state_id}] Capturing state '{state_label}' -> {state_abs_path}")
    rcc.start("image", False, raw_remote_path)
    rcc.stop()

    saved_cameras, missing_cameras = _wait_for_images(
        images_abs_dir,
        expected_cameras,
        timeout_s=args.wait_timeout,
    )

    now = datetime.datetime.now().isoformat(timespec="seconds")
    metadata = {
        "object_name": args.object_name,
        "session_name": os.path.basename(session_abs_path),
        "state_id": state_id,
        "state_index": state_index,
        "state_label": state_label,
        "state_kind": args.state_kind,
        "notes": notes,
        "created_at": now,
        "capture_mode": "image",
        "raw_images_dir": os.path.join(state_abs_path, "raw", "images"),
        "relative_raw_images_dir": os.path.join(state_rel_path, "raw", "images"),
        "cam_param_dir": os.path.join(state_abs_path, "cam_param"),
        "pc_list": pc_list,
        "expected_cameras": expected_cameras,
        "saved_cameras": saved_cameras,
        "missing_cameras": missing_cameras,
    }
    _write_json(os.path.join(state_abs_path, "metadata.json"), metadata)

    if missing_cameras:
        print(f"[WARN] {state_id}: missing images for cameras: {missing_cameras}")
    else:
        print(f"[{state_id}] Saved {len(saved_cameras)} camera images.")

    return metadata


def _planned_labels(args: argparse.Namespace) -> Optional[List[str]]:
    if args.state_labels:
        return args.state_labels
    return None


def _label_for_state(state_index: int, args: argparse.Namespace) -> str:
    default_label = f"state_{state_index:0{args.index_digits}d}"
    if not args.prompt_labels:
        return default_label

    label = input(f"State label [{default_label}]: ").strip()
    if not label:
        label = default_label
    return label


def _notes_for_state(args: argparse.Namespace) -> str:
    notes = ""
    if args.prompt_notes:
        notes = input("Notes for this state [optional]: ").strip()
    return notes


def _read_single_key() -> str:
    if not sys.stdin.isatty():
        return input().strip()[:1]

    if os.name == "nt":
        import msvcrt

        return msvcrt.getwch()

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _wait_for_capture_or_exit(label: str) -> bool:
    print(f"Prepare state '{label}', then press c to capture or q to finish: ", end="", flush=True)
    while True:
        cmd = _read_single_key().strip().lower()
        if not cmd:
            continue
        if cmd == "c":
            print("c")
            return True
        if cmd in {"q", "quit", "exit"}:
            print("q")
            return False
        print(f"\nUnknown command: {cmd}")
        print(f"Prepare state '{label}', then press c to capture or q to finish: ", end="", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture raw multiview snapshots for articulated/separable object states."
    )
    parser.add_argument("--object-name", required=True, help="Object/session subject name.")
    parser.add_argument(
        "--capture-root",
        default=os.path.join("capture", "articulated_object"),
        help="Relative root under shared_data.",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Session folder name. Defaults to current timestamp.",
    )
    parser.add_argument(
        "--state-kind",
        default="unspecified",
        choices=["unspecified", "base", "articulated", "separated", "hidden_surface"],
        help="Coarse state type stored in metadata.",
    )
    parser.add_argument(
        "--state-labels",
        nargs="*",
        default=None,
        help="Optional planned labels. If omitted, states are named state_000, state_001, ...",
    )
    parser.add_argument(
        "--prompt-labels",
        action="store_true",
        help="Prompt for each state label before waiting for c. Defaults to state_000, state_001, ...",
    )
    parser.add_argument("--prompt-notes", action="store_true", help="Prompt for per-state notes.")
    parser.add_argument("--notes", default="", help="Session-level notes.")
    parser.add_argument("--pc-list", nargs="*", default=None, help="Specific capture PCs to use.")
    parser.add_argument("--exclude-pc", nargs="*", default=None, help="Capture PCs to exclude.")
    parser.add_argument("--wait-timeout", type=float, default=10.0, help="Seconds to wait for images to appear.")
    parser.add_argument("--index-digits", type=int, default=3, help="Digits for state folder IDs.")
    parser.add_argument(
        "--min-states",
        type=int,
        default=3,
        help="Recommended minimum state count for later joint fitting warnings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from paradex.calibration.utils import save_current_camparam
    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
    from paradex.utils.path import shared_dir

    pc_list = _selected_pc_list(args)
    expected_cameras = _expected_cameras(pc_list)

    session_name = args.session_name or _timestamp()
    session_rel_path = os.path.join(args.capture_root, args.object_name, session_name)
    session_abs_path = os.path.join(shared_dir, session_rel_path)
    os.makedirs(session_abs_path, exist_ok=True)

    save_current_camparam(session_abs_path)
    session_payload = {
        "object_name": args.object_name,
        "session_name": session_name,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "capture_root": args.capture_root,
        "session_path": session_abs_path,
        "relative_session_path": session_rel_path,
        "notes": args.notes,
        "pc_list": pc_list,
        "expected_cameras": expected_cameras,
        "recommended_min_states": args.min_states,
        "states": [],
    }
    _write_json(os.path.join(session_abs_path, "session.json"), session_payload)

    print(f"Session path: {session_abs_path}")
    print(f"Capture PCs: {pc_list}")
    if expected_cameras:
        print(f"Expected cameras: {expected_cameras}")
    else:
        print("[WARN] No expected cameras found in pc config; captures will still run.")

    rcc = remote_camera_controller("articulated_object_capture", pc_list=pc_list)
    captured_states = []
    print("Keyboard control: c=capture current state, q=finish")

    try:
        labels = _planned_labels(args)
        if labels is not None:
            for idx, label in enumerate(labels):
                notes = _notes_for_state(args)
                if not _wait_for_capture_or_exit(label):
                    break
                captured_states.append(
                    _capture_one_state(
                        rcc,
                        args,
                        pc_list,
                        expected_cameras,
                        session_rel_path,
                        session_abs_path,
                        idx,
                        label,
                        notes,
                    )
                )
        else:
            print("Prepare each static object state, release occluders, then press c.")
            print("Use q to finish.")
            idx = 0
            while True:
                label = _label_for_state(idx, args)
                notes = _notes_for_state(args)
                if not _wait_for_capture_or_exit(label):
                    break
                captured_states.append(
                    _capture_one_state(
                        rcc,
                        args,
                        pc_list,
                        expected_cameras,
                        session_rel_path,
                        session_abs_path,
                        idx,
                        label,
                        notes,
                    )
                )
                idx += 1
    finally:
        rcc.end()

    session_payload["states"] = [
        {
            "state_id": state["state_id"],
            "state_label": state["state_label"],
            "state_kind": state["state_kind"],
            "path": os.path.join(session_abs_path, state["state_id"]),
            "missing_cameras": state["missing_cameras"],
        }
        for state in captured_states
    ]
    session_payload["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    _write_json(os.path.join(session_abs_path, "session.json"), session_payload)

    print(f"Captured {len(captured_states)} states.")
    if len(captured_states) < args.min_states:
        print(
            f"[WARN] Captured {len(captured_states)} states; "
            f"{args.min_states}+ states are recommended for stable joint fitting."
        )
    print(f"Session metadata: {os.path.join(session_abs_path, 'session.json')}")


if __name__ == "__main__":
    main()
