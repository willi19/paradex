"""Mirror the CURRENT camera calibration from the shared NAS to the local data2
store, so capture-time save_current_camparam / save_current_C2R never touch the
slow NAS.

- cam_param: full copy (small -- just intrinsics/extrinsics json).
- handeye_calibration: only the ONE C2R.npy that load_current_C2R actually reads
  (handeye/<latest>/<first-index>/C2R.npy, 256 bytes). The rest of the handeye
  dir (per-camera images/detection/debug, ~GB) is NOT needed at capture time, so
  we deliberately skip it -- a full mirror would re-copy GBs from the slow NAS on
  every recalibration.

Run once, and again after every recalibration (the capture log prints which
calibration name it is using, so you can tell if the mirror is stale).

    python src/process/teleop_real/sync_local_calib.py
"""
import os
import shutil

from paradex.calibration.utils import (
    cam_param_dir, handeye_calib_path,
    local_cam_param_dir, local_handeye_calib_path,
)
from paradex.utils.file_io import find_latest_directory


def mirror_camparam(nas_dir, local_dir):
    if not os.path.isdir(nas_dir):
        print(f"[sync] SKIP cam_param: NAS source missing ({nas_dir})")
        return
    name = find_latest_directory(nas_dir)
    if name is None:
        print(f"[sync] SKIP cam_param: NAS source empty ({nas_dir})")
        return
    src = os.path.join(nas_dir, name)
    dst = os.path.join(local_dir, name)
    if os.path.isdir(dst):
        print(f"[sync] cam_param: '{name}' already mirrored -> {dst}")
        return
    os.makedirs(local_dir, exist_ok=True)
    print(f"[sync] cam_param: copying '{name}' -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"[sync] cam_param: done ({name})")


def mirror_handeye_c2r(nas_dir, local_dir):
    """Copy ONLY handeye/<latest>/<first-index>/C2R.npy -- the single file
    load_current_C2R reads (it uses sorted(os.listdir(...))[0])."""
    if not os.path.isdir(nas_dir):
        print(f"[sync] SKIP handeye: NAS source missing ({nas_dir})")
        return
    name = find_latest_directory(nas_dir)
    if name is None:
        print(f"[sync] SKIP handeye: NAS source empty ({nas_dir})")
        return
    src_name = os.path.join(nas_dir, name)
    indices = sorted(os.listdir(src_name))
    if not indices:
        print(f"[sync] SKIP handeye: no index dirs under {src_name}")
        return
    idx = indices[0]                       # exactly what load_current_C2R picks
    src = os.path.join(src_name, idx, "C2R.npy")
    dst = os.path.join(local_dir, name, idx, "C2R.npy")
    if not os.path.exists(src):
        print(f"[sync] SKIP handeye: C2R.npy missing at {src}")
        return
    if os.path.exists(dst):
        print(f"[sync] handeye: '{name}/{idx}/C2R.npy' already mirrored -> {dst}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[sync] handeye: copied C2R.npy ('{name}/{idx}') -> {dst}")


if __name__ == "__main__":
    mirror_camparam(cam_param_dir, local_cam_param_dir)
    mirror_handeye_c2r(handeye_calib_path, local_handeye_calib_path)
    print("[sync] local calibration mirror ready under data2")
