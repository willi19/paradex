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
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import collect_videos          # noqa: E402
import delete_raw_avi          # noqa: E402

from paradex.utils.path import shared_dir, local_shared_dir, home_path    # noqa: E402
from paradex.utils.system import get_pc_list           # noqa: E402
from paradex.dataset_acqusition.match_sync import postprocess_session  # noqa: E402


def sync_local_robot_data(save_path, from_local):
    """Move locally-staged robot data (arm/hand/teleop/state/cam_param/...) to the
    NAS shared_dir so finalize can find + postprocess the sessions. Videos are
    NOT here (they come from the capture PCs via collect_videos)."""
    src = os.path.join(home_path, from_local, save_path)
    if not os.path.isdir(src):
        print(f"[finalize] no local robot data at {src} (skip)")
        return
    dst = os.path.join(shared_dir, save_path)
    os.makedirs(dst, exist_ok=True)
    print(f"[finalize] robot data local -> NAS: {src}/ -> {dst}/")
    subprocess.run(["rsync", "-a", src + "/", dst + "/"], check=False)


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


def list_sessions(root):
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
                    help="[--dest nas] DESTRUCTIVE: delete raw .avi (shared+capture) "
                         "after mp4 confirmed. Default: dry-run preview only.")
    ap.add_argument("--keep_avi", action="store_true",
                    help="[--dest local] Do NOT delete capture-PC avi. Default for "
                         "--dest local is to delete each camera's capture-PC avi (+temp "
                         "mp4) once its mp4 is confirmed on the main PC with a sane size.")
    ap.add_argument("--dest", choices=["local", "nas"], default="local",
                    help='"local" (default): operate on the main-PC data2 store '
                         '(local_shared_dir) where capture wrote everything -- no NAS. '
                         '"nas": operate on the shared NAS (slow).')
    ap.add_argument("--from_local", type=str, default=None,
                    help="[--dest nas only] Local staging folder name under $HOME where "
                         "robot data was saved during capture; rsync it to the NAS first.")
    args = ap.parse_args()

    root = local_shared_dir if args.dest == "local" else shared_dir
    root_sp = os.path.join(root, args.save_path)

    # --from_local only makes sense when finalizing to the NAS (data2 capture data
    # is already under local_shared_dir for --dest local).
    if args.from_local and args.dest == "nas":
        sync_local_robot_data(args.save_path, args.from_local)

    sessions = list_sessions(root_sp)
    if not sessions:
        print(f"[finalize] no sessions under {root_sp}")
        return

    # 1. collect videos for ALL sessions -- idempotent (transcode/rsync skip
    # existing mp4), so already-pulled sessions still get their capture-PC avi
    # reclaimed. dest=local: pull mp4 to data2 and (unless --keep_avi) delete each
    # camera's capture-PC avi+temp-mp4 once its mp4 is confirmed on the main PC.
    # dest=nas: push to NAS (avi cleanup handled in step 3).
    print(f"[finalize] dest={args.dest}  {len(sessions)} session(s); "
          f"collect+delete_avi={args.dest == 'local' and not args.keep_avi}")
    collect_videos.collect(args.save_path, do_rsync=True, do_mp4=True,
                            pc_list=get_pc_list(), sessions=sessions,
                            move_all=args.move_all, dest=args.dest,
                            delete_avi=(args.dest == "local" and not args.keep_avi))

    # 2. postprocess only the NOT-done sessions (the expensive step).
    if args.all:
        todo = sessions
    else:
        todo = [s for s in sessions if not is_done(os.path.join(root_sp, s))]
    print(f"[finalize] postprocess {len(todo)}: {todo}")
    for s in todo:
        sdir = os.path.join(root_sp, s)
        print(f"[finalize] postprocess {sdir}")
        postprocess_session(sdir)

    # 3. NAS-only raw cleanup (guarded: only where mp4 exists). For --dest local
    # the capture-PC avi was already deleted per-serial in step 1 (or kept with
    # --keep_avi), so there is nothing to do here.
    if args.dest == "nas":
        print(f"\n[finalize] raw .avi cleanup "
              f"({'DELETE' if args.delete_raw else 'DRY-RUN'}):")
        delete_raw_avi.delete_shared(args.save_path, args.delete_raw)
        delete_raw_avi.delete_capture(args.save_path, args.delete_raw, get_pc_list())

    print("[finalize] done")


if __name__ == "__main__":
    main()
