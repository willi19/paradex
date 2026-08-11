"""
Collect camera videos for a given save_path.

Transcode happens ON THE CAPTURE PC, then only the (small) .mp4 is rsynced to
the shared NAS by default -- much less network traffic than shipping raw .avi.
The raw .avi is also rsynced ONLY when:
  - capture-PC mp4 transcode failed / ffmpeg unavailable  (per-file fallback)
  - --move_all is given                                    (always ship avi too)

Any fallback .avi that lands on the NAS without an mp4 is transcoded on the
main PC afterwards (idempotent safety net).

NON-DESTRUCTIVE: capture-PC files are kept. Use delete_raw_avi.py to reclaim.

Layout assumption (from CaptureSession + camera_loader):
    capture PC : ~/captures{1,2}/{save_path}/{session}/raw/videos/{serial}.avi
    shared NAS : ~/shared_data/{save_path}/{session}/videos/{serial}.mp4
                 ~/shared_data/{save_path}/{session}/videos/{serial}.avi (only if above)
"""
import os
import sys
import glob
import argparse
import subprocess
import threading

from paradex.utils.system import get_pc_list, get_pc_ip
from paradex.utils.path import shared_dir, local_shared_dir

SSH_PORT = 77
# Shared-data folder name on the capture PCs (follows PARADEX_SHARED_DIR).
SHARED_NAME = os.path.basename(shared_dir.rstrip("/"))

# An mp4 is only treated as "properly produced" (skip re-transcode, safe to
# delete the source avi) once it is at least this many bytes. Real captures are
# 11 MB+, a failed/aborted transcode is a few KB, so 1 MB cleanly separates them.
MIN_MP4_BYTES = 1_000_000


def _good_mp4s(videos_dir, min_bytes=MIN_MP4_BYTES):
    """mp4 files in videos_dir that are at least min_bytes (i.e. not corrupt/partial)."""
    return [f for f in glob.glob(os.path.join(videos_dir, "*.mp4"))
            if os.path.getsize(f) >= min_bytes]


def _remote_rsync_script(save_path, sessions=None, move_all=False, delete_avi=False, crf=23):
    # Runs on the capture PC: transcode avi->mp4 locally, rsync the mp4.
    # rsync the avi too only on transcode failure (fallback) or move_all.
    # --chmod 644 so main PC (diff NFS uid) can read; --no-o/--no-g avoids
    # the chgrp "Operation not permitted" on the NAS.
    if sessions:
        globs = " ".join(
            f"$HOME/captures1/{save_path}/{s}/raw/videos/*.avi "
            f"$HOME/captures2/{save_path}/{s}/raw/videos/*.avi"
            for s in sessions
        )
    else:
        globs = (f"$HOME/captures1/{save_path}/*/raw/videos/*.avi "
                 f"$HOME/captures2/{save_path}/*/raw/videos/*.avi")
    move_all_flag = "1" if move_all else "0"
    del_avi_flag = "1" if delete_avi else "0"
    return f'''set -u
SD="$HOME/{SHARED_NAME}"
RS="rsync -rt --no-o --no-g --chmod=D755,F644"
nmp4=0; navi=0; ndel=0
for src in {globs}; do
  [ -f "$src" ] || continue
  base=${{src#$HOME/captures1/}}
  base=${{base#$HOME/captures2/}}
  rel=${{base/\\/raw\\/videos\\//\\/videos\\/}}
  dst_avi=$SD/$rel
  dst_mp4=$SD/${{rel%.avi}}.mp4
  if [ -s "$dst_mp4" ] && ( [ "{move_all_flag}" != "1" ] || [ -f "$dst_avi" ] ); then
    # Already collected. Still reclaim the capture-PC raw avi when --delete_avi
    # is set (mp4 is confirmed on the NAS), otherwise re-running to clean up
    # would skip forever and orphan the avi.
    if [ "{del_avi_flag}" = "1" ] && [ -f "$src" ]; then
      rm -f "$src" "${{src%.avi}}.mp4" && ndel=$((ndel+1)) && echo "  avi DELETED (already collected, mp4 on NAS): $src"
    else
      echo "  skip (already collected): ${{dst_mp4#$SD/}}"
    fi
    continue
  fi
  mkdir -p "$(dirname "$dst_avi")"
  mp4_local=${{src%.avi}}.mp4
  ok=0
  if command -v ffmpeg >/dev/null 2>&1; then
    if ffmpeg -nostdin -loglevel error -y -i "$src" -c:v libx264 -preset fast -crf {crf} -an "$mp4_local"; then
      ok=1
    fi
  fi
  if [ "$ok" = "1" ]; then
    if $RS "$mp4_local" "$dst_mp4"; then
      nmp4=$((nmp4+1)); echo "  mp4 OK ${{dst_mp4#$SD/}}"
      if [ "{move_all_flag}" = "1" ]; then
        $RS "$src" "$dst_avi" && navi=$((navi+1)) && echo "  avi OK (move_all) ${{dst_avi#$SD/}}"
      fi
      if [ "{del_avi_flag}" = "1" ]; then
        # mp4 is safely on the NAS -> reclaim the capture-PC raw avi (+ temp mp4)
        rm -f "$src" "$mp4_local" && ndel=$((ndel+1)) && echo "  avi DELETED (mp4 on NAS): $src"
      fi
    else
      echo "  mp4 rsync FAIL (avi kept): $dst_mp4"
    fi
  else
    echo "  mp4 FAIL -> fallback avi: $src"
    $RS "$src" "$dst_avi" && navi=$((navi+1)) && echo "  avi OK (fallback) ${{dst_avi#$SD/}}"
  fi
done
echo "[collect $(hostname)] mp4=$nmp4 avi=$navi deleted_avi=$ndel"
'''


def rsync_from_capture_pcs(save_path, pc_list, sessions=None, move_all=False, delete_avi=False, crf=23):
    for pc in pc_list:
        ip = get_pc_ip(pc)
        print(f"[collect] {pc} ({ip}) <- {save_path}"
              + (f" sessions={sessions}" if sessions else " (all sessions)")
              + (" [move_all]" if move_all else "")
              + (" [delete_avi]" if delete_avi else "")
              + (f" [crf={crf}]" if crf != 23 else ""))
        try:
            subprocess.run(
                ["ssh", "-p", str(SSH_PORT), "-o", "ConnectTimeout=5", f"{pc}@{ip}", "bash -s"],
                input=_remote_rsync_script(save_path, sessions, move_all, delete_avi, crf),
                text=True, check=False, timeout=1800,
            )
        except subprocess.TimeoutExpired:
            print(f"[collect] {pc}: TIMEOUT (still may continue server-side)")


def transcode_to_mp4(save_path, sessions=None, crf=23):
    if sessions:
        avis = []
        for s in sessions:
            avis += glob.glob(os.path.join(shared_dir, save_path, s, "videos", "*.avi"))
        avis = sorted(avis)
        pattern = f"{save_path}/{{{','.join(sessions)}}}/videos/*.avi"
    else:
        pattern = os.path.join(shared_dir, save_path, "*", "videos", "*.avi")
        avis = sorted(glob.glob(pattern))
    if not avis:
        print(f"[mp4] no .avi found under {pattern}")
        return
    print(f"[mp4] {len(avis)} avi -> mp4")
    for avi in avis:
        mp4 = avi[:-4] + ".mp4"
        if os.path.exists(mp4) and os.path.getsize(mp4) > 0:
            print(f"[mp4] skip (exists): {os.path.basename(mp4)}")
            continue
        print(f"[mp4] {os.path.relpath(avi, shared_dir)} -> mp4")
        r = subprocess.run(
            ["ffmpeg", "-i", avi, "-c:v", "libx264", "-preset", "fast",
             "-crf", str(crf), "-an", "-y", mp4],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if r.returncode != 0 or not os.path.exists(mp4):
            print(f"[mp4] FAIL: {avi}")
        else:
            print(f"[mp4] OK   {os.path.basename(mp4)} "
                  f"({os.path.getsize(mp4) // (1024*1024)}MB)")


def push_local_to_nas(save_path, sessions=None, delete_local=False):
    """Move main-PC LOCAL capture data (arm/hand/teleop/state/cam_param/C2R,
    written by CaptureSession to local_shared_dir for fast 's') onto the NAS.

    Runs on the main PC; both paths are local here (local disk -> NFS mount).
    Non-destructive by default (keeps the local copy). Videos are NOT here --
    they come from the capture PCs via rsync_from_capture_pcs(), and both land
    under the same shared_dir/{save_path}/{session}/, so no --delete on rsync.
    """
    if os.path.abspath(local_shared_dir) == os.path.abspath(shared_dir):
        return  # capture already wrote straight to the NAS; nothing to move
    if sessions:
        srcs = [os.path.join(local_shared_dir, save_path, s) for s in sessions]
    else:
        srcs = sorted(glob.glob(os.path.join(local_shared_dir, save_path, "*")))
    moved = 0
    for src in srcs:
        if not os.path.isdir(src):
            print(f"[nas] skip (no local data): {os.path.relpath(src, local_shared_dir)}")
            continue
        rel = os.path.relpath(src, local_shared_dir)
        dst = os.path.join(shared_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # --no-o/--no-g/--chmod: NAS is a different NFS uid/gid (same reason the
        # capture-PC rsync uses these). No --delete: don't clobber videos that
        # the capture-PC rsync already put under the same session dir.
        cmd = ["rsync", "-rt", "--no-o", "--no-g", "--chmod=D755,F644",
               src.rstrip("/") + "/", dst.rstrip("/") + "/"]
        if delete_local:
            cmd.insert(1, "--remove-source-files")
        r = subprocess.run(cmd, check=False)
        if r.returncode == 0:
            moved += 1
            print(f"[nas] local->NAS OK: {rel}" + (" (local removed)" if delete_local else ""))
        else:
            print(f"[nas] local->NAS FAIL (local kept): {rel}")
    print(f"[nas] pushed {moved} session(s) local->NAS")


def _remote_transcode_script(save_path, sessions=None, crf=23, min_bytes=MIN_MP4_BYTES):
    # Runs on the capture PC: transcode avi->mp4 IN PLACE (next to the avi).
    # No NAS write -- the main PC pulls the mp4 afterwards (dest="local").
    # Skips re-transcoding only when a big-enough mp4 already exists (a tiny
    # leftover from a previous failed transcode is redone, not trusted).
    if sessions:
        globs = " ".join(
            f"$HOME/captures1/{save_path}/{s}/raw/videos/*.avi "
            f"$HOME/captures2/{save_path}/{s}/raw/videos/*.avi"
            for s in sessions
        )
    else:
        globs = (f"$HOME/captures1/{save_path}/*/raw/videos/*.avi "
                 f"$HOME/captures2/{save_path}/*/raw/videos/*.avi")
    return f'''set -u
H=$(hostname)
tot=0; for src in {globs}; do [ -f "$src" ] && tot=$((tot+1)); done
echo "[transcode $H] START: $tot avi to check (skip if mp4 already >= {min_bytes} bytes)"
n=0; e=0; f=0; i=0
for src in {globs}; do
  [ -f "$src" ] || continue
  i=$((i+1))
  mp4=${{src%.avi}}.mp4
  if [ -f "$mp4" ] && [ "$(stat -c%s "$mp4" 2>/dev/null || echo 0)" -ge {min_bytes} ]; then e=$((e+1)); continue; fi
  # 2>/dev/null: drop benign mjpeg "overread" warnings; we gate on the exit code.
  if command -v ffmpeg >/dev/null 2>&1 && ffmpeg -nostdin -loglevel error -y -i "$src" -c:v libx264 -preset fast -crf {crf} -an "$mp4" 2>/dev/null; then
    n=$((n+1)); echo "[transcode $H] $i/$tot ok $(basename "$src")"
  else
    f=$((f+1)); echo "[transcode $H] $i/$tot FAIL (avi will be pulled+transcoded on main): $(basename "$src")"
  fi
done
echo "[transcode $H] DONE new_mp4=$n already=$e fail=$f (of $tot)"
'''


def _remote_video_dirs(pc, ip, save_path, sessions):
    """(cdir, session) pairs that STILL have a raw/videos dir on the capture PC.

    Used to skip rsyncing a source that was already collected+deleted (idempotent
    re-runs otherwise spew 'No such file or directory' + broken-pipe from rsync).
    Returns None on probe failure -> caller falls back to trying both cdirs."""
    lines = "".join(
        f'for c in captures1 captures2; do '
        f'[ -d "$HOME/$c/{save_path}/{s}/raw/videos" ] && echo "$c {s}"; done\n'
        for s in sessions
    )
    try:
        out = subprocess.run(
            ["ssh", "-p", str(SSH_PORT), "-o", "ConnectTimeout=5", f"{pc}@{ip}", "bash -s"],
            input=lines, capture_output=True, text=True, timeout=60,
        )
        return {tuple(p.split()) for p in out.stdout.splitlines() if len(p.split()) == 2}
    except Exception as exc:
        print(f"[pull] {pc}: video-dir probe failed ({exc}); trying both cdirs")
        return None


def _remote_list_sessions(pc, ip, save_path):
    try:
        out = subprocess.run(
            ["ssh", "-p", str(SSH_PORT), "-o", "ConnectTimeout=5", f"{pc}@{ip}",
             f"ls -1 $HOME/captures1/{save_path} $HOME/captures2/{save_path} 2>/dev/null | sort -u"],
            capture_output=True, text=True, timeout=30,
        )
        return [s.strip() for s in out.stdout.splitlines() if s.strip()]
    except Exception as exc:
        print(f"[pull] {pc}: session list failed: {exc}")
        return []


def pull_videos_to_local(save_path, pc_list, sessions=None, crf=23,
                         delete_avi=False, move_all=False, min_bytes=MIN_MP4_BYTES):
    """dest="local": transcode on the capture PC, then PULL the mp4 to this main
    PC's local_shared_dir (avoids the slow NAS entirely). Final layout matches
    the NAS: local_shared_dir/{save_path}/{session}/videos/{serial}.mp4, i.e.
    right next to the robot data CaptureSession already wrote locally.

    delete_avi: once a serial's mp4 is confirmed on the main PC with a sane size,
    delete THAT serial's capture-PC raw avi (+ temp mp4). Per-serial, never a
    blanket session wipe, so a camera that failed to pull keeps its avi."""
    # Phase 1: transcode avi->mp4 on ALL capture PCs IN PARALLEL (the slow ffmpeg
    # step). Each PC's ffmpeg is independent, so run them concurrently instead of
    # capture4-then-capturenew. Pull (phase 2) stays sequential -- it's just small
    # mp4 rsync, and keeping it serial avoids races on the shared session videos/.
    def _transcode(pc, ip):
        try:
            subprocess.run(
                ["ssh", "-p", str(SSH_PORT), "-o", "ConnectTimeout=5", f"{pc}@{ip}", "bash -s"],
                input=_remote_transcode_script(save_path, sessions, crf, min_bytes),
                text=True, check=False, timeout=1800,
            )
        except subprocess.TimeoutExpired:
            print(f"[pull] {pc}: transcode TIMEOUT (pulling whatever exists)")

    print(f"[pull] transcoding avi->mp4 on {len(pc_list)} capture PC(s) in PARALLEL"
          + (f" sessions={sessions}" if sessions else " (all sessions)"))
    tthreads = []
    for pc in pc_list:
        t = threading.Thread(target=_transcode, args=(pc, get_pc_ip(pc)), daemon=True)
        t.start()
        tthreads.append(t)
    for t in tthreads:
        t.join()
    print("[pull] transcode phase done -> pulling mp4")

    # Phase 2: pull + per-serial delete, sequential per PC.
    for pc in pc_list:
        ip = get_pc_ip(pc)
        sess_list = sessions or _remote_list_sessions(pc, ip, save_path)
        if not sess_list:
            print(f"[pull] {pc}: no sessions under {save_path}")
            continue

        # Which (cdir, session) actually still have videos on this PC, so we never
        # rsync a source that was already collected+deleted (that spews rsync
        # 'No such file' + broken-pipe noise on idempotent re-runs).
        existing = _remote_video_dirs(pc, ip, save_path, sess_list)

        for s in sess_list:
            if existing is None:
                cdirs_here = ["captures1", "captures2"]
            else:
                cdirs_here = [c for c in ("captures1", "captures2") if (c, s) in existing]
            if not cdirs_here:
                continue  # nothing left on this PC for this session (already done)

            dst = os.path.join(local_shared_dir, save_path, s, "videos")
            os.makedirs(dst, exist_ok=True)
            # primary: pull only the (small) mp4 from the capture dirs that have it
            for cdir in cdirs_here:
                src = f"{pc}@{ip}:{cdir}/{save_path}/{s}/raw/videos/"
                subprocess.run(
                    ["rsync", "-rt", "--include=*.mp4", "--exclude=*",
                     "-e", f"ssh -p {SSH_PORT}", src, dst + "/"], check=False)

            # safety net: capture PC couldn't transcode (no ffmpeg) -> pull the
            # avi and transcode HERE (fast local ffmpeg), then drop the local avi.
            if not _good_mp4s(dst, min_bytes):
                print(f"[pull] {s}: no mp4 -> fallback pull avi + local transcode")
                for cdir in cdirs_here:
                    src = f"{pc}@{ip}:{cdir}/{save_path}/{s}/raw/videos/"
                    subprocess.run(
                        ["rsync", "-rt", "--include=*.avi", "--exclude=*",
                         "-e", f"ssh -p {SSH_PORT}", src, dst + "/"], check=False)
                for avi in sorted(glob.glob(os.path.join(dst, "*.avi"))):
                    mp4 = avi[:-4] + ".mp4"
                    r = subprocess.run(
                        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", avi,
                         "-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-an", mp4],
                        check=False)
                    if r.returncode == 0 and os.path.exists(mp4):
                        os.remove(avi)  # only pulled to transcode; reclaim space
            elif move_all:
                for cdir in cdirs_here:
                    src = f"{pc}@{ip}:{cdir}/{save_path}/{s}/raw/videos/"
                    subprocess.run(
                        ["rsync", "-rt", "--include=*.avi", "--exclude=*",
                         "-e", f"ssh -p {SSH_PORT}", src, dst + "/"], check=False)

            good = _good_mp4s(dst, min_bytes)
            good_serials = [os.path.splitext(os.path.basename(f))[0] for f in good]
            n_mp4 = len(glob.glob(os.path.join(dst, "*.mp4")))
            n_avi = len(glob.glob(os.path.join(dst, "*.avi")))
            print(f"[pull] {pc}:{s} -> {os.path.relpath(dst, local_shared_dir)}  "
                  f"mp4={n_mp4} (ok={len(good_serials)}) avi={n_avi}")

            if delete_avi and good_serials:
                # Per-serial: delete on the capture PC ONLY the avi (+ temp mp4)
                # whose mp4 is confirmed on the main PC with a sane size. A camera
                # that didn't pull keeps its avi (no blanket session wipe).
                rm = "; ".join(
                    f"rm -f $HOME/{cdir}/{save_path}/{s}/raw/videos/{ser}.avi "
                    f"$HOME/{cdir}/{save_path}/{s}/raw/videos/{ser}.mp4"
                    for cdir in cdirs_here for ser in good_serials)
                # Then remove the now-empty videos/ raw/ session/ dirs -- ONLY if
                # empty (rmdir), and only up to the session level, so a sibling
                # session or the save_path prefix is never touched.
                rmdirs = "; ".join(
                    f'rmdir --ignore-fail-on-non-empty '
                    f'"$HOME/{cdir}/{save_path}/{s}/raw/videos" '
                    f'"$HOME/{cdir}/{save_path}/{s}/raw" '
                    f'"$HOME/{cdir}/{save_path}/{s}" 2>/dev/null'
                    for cdir in cdirs_here)
                subprocess.run(["ssh", "-p", str(SSH_PORT), f"{pc}@{ip}",
                                rm + "; " + rmdirs], check=False)
                print(f"[pull] {pc}:{s} capture-PC avi+mp4 DELETED for "
                      f"{len(good_serials)} serial(s) (+empty dirs cleaned)")


def collect(save_path, do_rsync=True, do_mp4=True, pc_list=None,
            sessions=None, move_all=False, delete_avi=False, crf=23,
            push_local=True, delete_local=False, dest="nas",
            min_bytes=MIN_MP4_BYTES):
    if pc_list is None:
        pc_list = get_pc_list()
    if dest == "local":
        # Keep everything on the main PC's local disk (robot data is already
        # there via CaptureSession). No NAS I/O at all.
        pull_videos_to_local(save_path, pc_list, sessions, crf, delete_avi,
                             move_all, min_bytes)
        return
    if push_local:
        # First move the fast-local robot data (arm/hand/teleop/cam_param/C2R)
        # onto the NAS, so the session dir on the NAS is complete before videos.
        push_local_to_nas(save_path, sessions, delete_local=delete_local)
    if do_rsync:
        rsync_from_capture_pcs(save_path, pc_list, sessions, move_all, delete_avi, crf)
    if do_mp4:
        # Safety net: transcode any fallback .avi that arrived without an mp4.
        transcode_to_mp4(save_path, sessions, crf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True,
                        help="e.g. teleop_test/01 (the prefix)")
    parser.add_argument("--sessions", nargs="*", default=None,
                        help="session leaf names to limit to (e.g. 2026-05-15_21-02-24). "
                             "Default: all sessions under save_path.")
    parser.add_argument("--move_all", action="store_true",
                        help="also rsync raw .avi (not just mp4) even on success")
    parser.add_argument("--delete_avi", action="store_true",
                        help="on successful transcode+mp4 upload, DELETE the capture-PC "
                             "raw .avi (and temp mp4) to reclaim space. Destructive but "
                             "safe: only deletes after the mp4 is on the NAS.")
    parser.add_argument("--no_rsync", action="store_true",
                        help="skip capture-PC collect, only main-PC fallback transcode")
    parser.add_argument("--no_mp4", action="store_true",
                        help="skip main-PC fallback transcode")
    parser.add_argument("--no_push_local", action="store_true",
                        help="skip moving the main-PC LOCAL capture data "
                             "(arm/hand/teleop/cam_param/C2R in local_shared_dir) to the NAS")
    parser.add_argument("--delete_local", action="store_true",
                        help="after a successful local->NAS move, delete the local copy "
                             "(local_shared_dir) to reclaim disk. Safe: only removes files "
                             "that rsync confirmed transferred.")
    parser.add_argument("--crf", type=int, default=23,
                        help="H.264 quality for the mp4 (lower=better/bigger). "
                             "18~=visually lossless, 23=default, 28=small. Range ~18-30.")
    parser.add_argument("--dest", choices=["local", "nas"], default="local",
                        help='"local" (default): transcode on the capture PC and PULL the '
                             "mp4 to this main PC's data2 store (no NAS). "
                             '"nas": push everything to the shared NAS (slow, needs NAS '
                             "write perms on the capture PCs).")
    args = parser.parse_args()

    collect(args.save_path, do_rsync=not args.no_rsync, do_mp4=not args.no_mp4,
            sessions=args.sessions, move_all=args.move_all, delete_avi=args.delete_avi,
            crf=args.crf, push_local=not args.no_push_local,
            delete_local=args.delete_local, dest=args.dest)
    print("[collect_videos] done")
