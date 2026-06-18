import json
import os
from pathlib import Path


def _candidate_config_dirs():
    env_dir = os.environ.get("PARADEX_CONFIG_DIR")
    if env_dir:
        yield Path(env_dir).expanduser()

    package_root = Path(__file__).resolve().parents[2]
    yield package_root / "system" / "current"

    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        yield parent / "system" / "current"


def _find_config_dir():
    seen = set()
    for path in _candidate_config_dirs():
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if (path / "pc.json").is_file() and (path / "network.json").is_file():
            return path
    searched = "\n  ".join(str(p) for p in seen)
    raise FileNotFoundError(
        "Could not find paradex system config directory containing "
        "pc.json and network.json.\n"
        "Set PARADEX_CONFIG_DIR=/path/to/system/current or run from a "
        f"repo checkout.\nSearched:\n  {searched}"
    )


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


config_dir = str(_find_config_dir())
pc_info = _load_json(os.path.join(config_dir, "pc.json"))
network_info = _load_json(os.path.join(config_dir, "network.json"))
pc_name = os.path.basename(os.path.expanduser("~"))

def get_pc_ip(pc_name: str) -> str:
    return pc_info[pc_name]["ip"]

def get_camera_list(pc_name: str = pc_name) -> list:
    if pc_name not in pc_info:
        raise KeyError(
            f"PC '{pc_name}' is not configured in {os.path.join(config_dir, 'pc.json')}. "
            f"Configured camera PCs: {list(pc_info.keys())}. "
            "Run camera-local tools on the capture PC that owns the cameras."
        )
    return pc_info[pc_name]["cam_list"]

def get_pc_list() -> list:
    return list(pc_info.keys())

def get_camera_config() -> dict:
    cam_config_path = os.path.join(config_dir, "camera.json")
    if os.path.exists(cam_config_path):
        cam_info = json.load(open(cam_config_path, "r"))
    else:
        cam_info = {}
    return cam_info
