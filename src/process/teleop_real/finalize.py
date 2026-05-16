"""
Batch-finalize all teleop sessions under a save_path (use after --no_collect):

  1. rsync raw .avi from capture PCs -> shared, transcode to .mp4
  2. postprocess_session  (synthesize_camera_timeline -> arm/hand frame-aligned)
  3. (optional) delete raw .avi originals  -- DESTRUCTIVE, needs --delete_raw

A session is considered DONE when {session}/videos/*.mp4 exists AND
{session}/arm/position.npy length == that mp4's frame count. By default only
not-done sessions are (re)processed; pass --all to force every session.

Usage:
  python src/process/teleop_real/finalize.py --save_path teleop_test/01
  python src/process/teleop_real/finalize.py --save_path teleop_test/01 --delete_raw
  python src/process/teleop_real/finalize.py --save_path teleop_test/01 --all
"""
import os
import sys
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import collect_videos          # noqa: E402
import delete_raw_avi          # noqa: E402

from paradex.utils.path import shared_dir              # noqa: E402
from paradex.utils.system import get_pc_list           # noqa: E402
from paradex.dataset_acqusition.match_sync import postprocess_session  # noqa: E402


def _video_frames(session_dir):
    import cv2
    for p in sorted(glob.glob(os.path.join(session_dir, "videos", "*.mp4")) +
                    glob.glob(os.path.join(session_dir, "videos", "*.avi"))):
        cap = cv2.VideoCapture(p)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    return 0


def is_done(session_dir):
    """Done = has mp4 AND arm/position.npy already resampled to frame count."""
    if not glob.glob(os.path.join(session_dir, "videos", "*.mp4")):
        return False
    arm_pos = os.path.join(session_dir, "arm", "position.npy")
    if not os.path.exists(arm_pos):
        return False
    nframes = _video_frames(session_dir)
    if nframes == 0:
        return False
    return len(np.load(arm_pos)) == nframes


def list_sessions(save_path):
    root = os.path.join(shared_dir, save_path)
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        sdir = os.path.join(root, name)
        if os.path.isdir(os.path.join(sdir, "raw")):
            out.append(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_path", required=True, help="e.g. teleop_test/01")
    ap.add_argument("--all", action="store_true",
                    help="process every session, not just not-done ones")
    ap.add_argument("--move_all", action="store_true",
                    help="also rsync raw .avi (not just mp4) even on success")
    ap.add_argument("--delete_raw", action="store_true",
                    help="DESTRUCTIVE: delete raw .avi (shared+capture) after "
                         "mp4 confirmed. Default: dry-run preview only.")
    args = ap.parse_args()

    sessions = list_sessions(args.save_path)
    if not sessions:
        print(f"[finalize] no sessions under {shared_dir}/{args.save_path}")
        return

    if args.all:
        todo = sessions
    else:
        todo = [s for s in sessions
                if not is_done(os.path.join(shared_dir, args.save_path, s))]

    print(f"[finalize] {len(sessions)} session(s); processing {len(todo)}: {todo}")
    if not todo:
        print("[finalize] nothing to do (all done). Use --all to force.")
    else:
        # 1. collect (rsync + mp4) — scoped to the not-done sessions
        collect_videos.collect(args.save_path, do_rsync=True, do_mp4=True,
                                pc_list=get_pc_list(), sessions=todo,
                                move_all=args.move_all)
        # 2. postprocess each (synthesize_camera_timeline kicks in)
        for s in todo:
            sdir = os.path.join(shared_dir, args.save_path, s)
            print(f"[finalize] postprocess {sdir}")
            postprocess_session(sdir)

    # 3. delete raw originals (guarded: only where shared mp4 exists)
    print(f"\n[finalize] raw .avi cleanup "
          f"({'DELETE' if args.delete_raw else 'DRY-RUN'}):")
    delete_raw_avi.delete_shared(args.save_path, args.delete_raw)
    delete_raw_avi.delete_capture(args.save_path, args.delete_raw, get_pc_list())

    print("[finalize] done")


if __name__ == "__main__":
    main()
