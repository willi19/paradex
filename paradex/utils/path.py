import os

home_path = os.path.expanduser("~")
pc_name = os.path.basename(home_path)
# Shared-data root folder name (under $HOME). Override with env
# PARADEX_SHARED_DIR (e.g. "shared_data_local") to redirect robot captures AND
# collect_videos to a different shared folder. Default keeps existing behavior.
shared_dir = os.path.join(home_path, os.environ.get("PARADEX_SHARED_DIR", "shared_data"))
# Fast local store on the big data2 disk (7.3T) instead of the slow NAS. Capture
# writes robot data here, collect_videos --dest local pulls videos here, and the
# camera calibration is mirrored here (see sync_local_calib.py) so 's' never
# touches the NAS. Override with PARADEX_LOCAL_DIR.
local_shared_dir = os.environ.get(
    "PARADEX_LOCAL_DIR", os.path.join(home_path, "data2", "zerodex_data"))
capture_path_list = [os.path.join(home_path, f"captures{i}") for i in range(1,3)]

download_dir = os.path.join(home_path, "download")
model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "model")

rsc_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "rsc",
)
