"""miyungpa robot-demo processing, on the paradex.process framework.

One :class:`Job` per demo (`capture/miyungpa/<obj>/<index>`). Each job:
    download videos (NAS->local) -> match_sync sensors -> render robot overlay
    -> upload overlay/merged back to NAS.

The overlay frame loop reports ``ctx.status(frame=i, total=N)`` so the dashboard
shows "overlay 340/1200 @28fps • ETA 1m35s" per demo, aggregated across PCs.

Run:
    python src/process/miyungpa/worker.py            # distributed (this PC's shard)
    python src/process/miyungpa/worker.py --local    # this machine only

miyungpa does its own whole-directory rsync and reads/writes NAS paths directly
(the "local IO in meta" pattern), so jobs declare no framework inputs/outputs —
just a ``done`` predicate (merged.mp4 on NAS) for skip-if-done.
"""

import os
import sys

import numpy as np
import cv2
import tqdm

from paradex.process import Job, run_jobs, serve_jobs, shard
from paradex.utils.path import shared_dir, home_path
from paradex.dataset_acqusition.match_sync import get_synced_data, fill_framedrop
from paradex.image.image_dict import ImageDict
from paradex.calibration.utils import load_camparam, load_c2r
from paradex.visualization.robot import RobotModule
from paradex.robot.utils import get_robot_urdf_path
from paradex.image.merge import merge_image
from paradex.utils.upload_file import rsync_copy
from paradex.video.util import convert_avi_to_mp4
from paradex.robot.inspire import parse_inspire

MIYUNGPA_ROOT = "capture/miyungpa"


# --------------------------------------------------------------------------- #
# discover
# --------------------------------------------------------------------------- #
def _merged_done(job):
    return os.path.exists(os.path.join(shared_dir, job.meta["demo_path"], "merged.mp4"))


def discover():
    """One Job per `capture/miyungpa/<obj>/<index>` demo found on NAS."""
    jobs = []
    root = os.path.join(shared_dir, MIYUNGPA_ROOT)
    if not os.path.isdir(root):
        return jobs
    for obj_name in sorted(os.listdir(root)):
        obj_dir = os.path.join(root, obj_name)
        if not os.path.isdir(obj_dir):
            continue
        for index in sorted(os.listdir(obj_dir)):
            demo_path = os.path.join(MIYUNGPA_ROOT, obj_name, index)
            jobs.append(Job(
                id=f"{obj_name}/{index}",
                meta={"demo_path": demo_path},
                done=_merged_done,
            ))
    return jobs


# --------------------------------------------------------------------------- #
# per-stage helpers (NAS-relative IO, unchanged from the original client)
# --------------------------------------------------------------------------- #
def download_dir(demo_path):
    video_dir = os.path.join(shared_dir, demo_path, "videos")
    if not os.path.exists(video_dir):
        return
    dst = os.path.join(home_path, "paradex_download", demo_path)
    os.makedirs(dst, exist_ok=True)
    rsync_copy(video_dir + "/", dst + "/", checksum=True, resume=True, verbose=False)


def match_sync(demo_path):
    root_dir = os.path.join(shared_dir, demo_path)
    sensor_data_path = os.path.join(root_dir, "raw")

    ts_dir = os.path.join(sensor_data_path, "timestamps")
    if not os.path.exists(os.path.join(ts_dir, "timestamp.npy")) \
       or not os.path.exists(os.path.join(ts_dir, "frame_id.npy")):
        return

    frameid = np.load(os.path.join(ts_dir, "frame_id.npy"))
    pc_time = np.load(os.path.join(ts_dir, "timestamp.npy"))
    pc_time, frameid = fill_framedrop(frameid, pc_time)

    for sensor_name in ["arm", "hand"]:
        sensor_path = os.path.join(sensor_data_path, sensor_name)
        if not os.path.isdir(sensor_path):
            continue
        time_path = os.path.join(sensor_path, "time.npy")
        if not os.path.exists(time_path):
            continue
        os.makedirs(os.path.join(root_dir, sensor_name), exist_ok=True)
        sensor_timestamps = np.load(time_path)
        for data_name in os.listdir(sensor_path):
            if data_name == "time.npy":
                continue
            data_path = os.path.join(sensor_path, data_name)
            if not os.path.isfile(data_path):
                continue
            synced_data = get_synced_data(pc_time, np.load(data_path), sensor_timestamps)
            np.save(os.path.join(root_dir, sensor_name, data_name), synced_data)


def overlay(demo_path, ctx=None):
    """Render the robot-mesh overlay per camera + a merged video.

    Reports frame-level progress through ``ctx`` (paradex.process) so the
    dashboard can show frame-of-N and a precise ETA.
    """
    root_dir = os.path.join(home_path, "paradex_download", demo_path)
    videos_dir = os.path.join(root_dir, "videos")
    if not os.path.exists(videos_dir):
        return

    video_cap = {name.split(".")[0]: cv2.VideoCapture(os.path.join(videos_dir, name))
                 for name in os.listdir(videos_dir)}

    # already fully rendered? (merged length matches the longest source)
    if os.path.exists(os.path.join(root_dir, "merged.avi")):
        merged_len = int(cv2.VideoCapture(os.path.join(root_dir, "merged.avi")).get(cv2.CAP_PROP_FRAME_COUNT))
        src_len = max(int(cv2.VideoCapture(os.path.join(videos_dir, n)).get(cv2.CAP_PROP_FRAME_COUNT))
                      for n in os.listdir(videos_dir))
        if merged_len == src_len:
            for cap in video_cap.values():
                cap.release()
            return

    for name in list(video_cap.keys()):
        if int(video_cap[name].get(cv2.CAP_PROP_FRAME_COUNT)) == 0:
            video_cap[name].release()
            del video_cap[name]
    if not video_cap:
        return

    video_open = {name: True for name in video_cap}
    max_length = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in video_cap.values())
    os.makedirs(os.path.join(root_dir, "overlay"), exist_ok=True)

    first = video_cap[next(iter(video_cap))]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_shape = (int(first.get(cv2.CAP_PROP_FRAME_WIDTH)), int(first.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = first.get(cv2.CAP_PROP_FPS)

    out_cap = {name: cv2.VideoWriter(os.path.join(root_dir, "overlay", f"{name}.avi"),
                                     fourcc, fps, frame_shape) for name in video_cap}
    merged_vid = cv2.VideoWriter(os.path.join(root_dir, "merged.avi"), fourcc, fps, frame_shape)

    intrinsics, extrinsics = load_camparam(os.path.join(shared_dir, demo_path))
    c2r = load_c2r(os.path.join(shared_dir, demo_path))
    rm = RobotModule(get_robot_urdf_path(arm_name="xarm", hand_name="inspire"))
    hand_state = parse_inspire(np.load(os.path.join(shared_dir, demo_path, "hand", "position.npy")))
    arm_state = np.load(os.path.join(shared_dir, demo_path, "arm", "position.npy"))

    _iter = range(max_length) if ctx is not None else tqdm.tqdm(range(max_length))
    for idx in _iter:
        if ctx is not None:
            ctx.status("overlay", frame=idx, total=max_length)
        frame_dict = {}
        for name, cap in video_cap.items():
            if not video_open[name]:
                continue
            ret, frame = cap.read()
            if not ret:
                video_open[name] = False
                continue
            frame_dict[name] = frame
        if not frame_dict:
            break

        part_intr = {n: intrinsics[n] for n in frame_dict}
        part_extr = {n: extrinsics[n] for n in frame_dict}
        imgdict = ImageDict(frame_dict, part_intr, part_extr, path=None)
        rm.update_cfg(np.concatenate([arm_state[idx], hand_state[idx]]))
        robot_mesh = rm.get_robot_mesh()
        robot_mesh.apply_transform(c2r)

        overlayed = imgdict.project_mesh(robot_mesh, color=(0, 255, 0))
        for name, out in out_cap.items():
            if video_open[name]:
                out.write(overlayed.images[name])
        merged_frame = cv2.resize(merge_image(overlayed.images), frame_shape)
        merged_vid.write(merged_frame)

    for cap in video_cap.values():
        cap.release()
    for out in out_cap.values():
        out.release()
    merged_vid.release()
    convert_avi_to_mp4(os.path.join(root_dir, "merged.avi"), os.path.join(root_dir, "merged.mp4"))


def upload_output(demo_path):
    root_dir = os.path.join(home_path, "paradex_download", demo_path)
    rsync_copy(os.path.join(root_dir, "overlay") + "/",
               os.path.join(shared_dir, demo_path) + "/",
               checksum=True, resume=True, verbose=False)
    rsync_copy(os.path.join(root_dir, "merged.mp4"),
               os.path.join(shared_dir, demo_path, "merged.mp4"),
               checksum=True, resume=True, verbose=False)


# --------------------------------------------------------------------------- #
# process one job
# --------------------------------------------------------------------------- #
def process(job, ctx):
    demo_path = job.meta["demo_path"]
    ctx.status("downloading videos", progress=0.0)
    download_dir(demo_path)
    ctx.status("syncing sensors")
    match_sync(demo_path)
    overlay(demo_path, ctx)                       # frame-level status inside
    ctx.status("uploading outputs")
    upload_output(demo_path)
    ctx.status("done", progress=1.0)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    jobs = discover()
    # overlay is GPU/mem-heavy (RobotModule + projection); keep the pool small.
    if "--local" in sys.argv:
        run_jobs(jobs, process, num_workers=1)
    else:
        serve_jobs(shard(jobs), process, num_workers=2)
