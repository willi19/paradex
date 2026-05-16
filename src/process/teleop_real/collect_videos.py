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

from paradex.utils.system import get_pc_list, get_pc_ip
from paradex.utils.path import shared_dir

SSH_PORT = 77


def _remote_rsync_script(save_path, sessions=None, move_all=False):
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
    return f'''set -u
RS="rsync -rt --no-o --no-g --chmod=D755,F644"
nmp4=0; navi=0
for src in {globs}; do
  [ -f "$src" ] || continue
  base=${{src#$HOME/captures1/}}
  base=${{base#$HOME/captures2/}}
  rel=${{base/\\/raw\\/videos\\//\\/videos\\/}}
  dst_avi=$HOME/shared_data/$rel
  dst_mp4=$HOME/shared_data/${{rel%.avi}}.mp4
  mkdir -p "$(dirname "$dst_avi")"
  mp4_local=${{src%.avi}}.mp4
  ok=0
  if command -v ffmpeg >/dev/null 2>&1; then
    if ffmpeg -nostdin -loglevel error -y -i "$src" -c:v libx264 -preset fast -crf 23 -an "$mp4_local"; then
      ok=1
    fi
  fi
  if [ "$ok" = "1" ]; then
    $RS "$mp4_local" "$dst_mp4" && nmp4=$((nmp4+1)) && echo "  mp4 OK ${{dst_mp4#$HOME/shared_data/}}"
    if [ "{move_all_flag}" = "1" ]; then
      $RS "$src" "$dst_avi" && navi=$((navi+1)) && echo "  avi OK (move_all) ${{dst_avi#$HOME/shared_data/}}"
    fi
  else
    echo "  mp4 FAIL -> fallback avi: $src"
    $RS "$src" "$dst_avi" && navi=$((navi+1)) && echo "  avi OK (fallback) ${{dst_avi#$HOME/shared_data/}}"
  fi
done
echo "[collect $(hostname)] mp4=$nmp4 avi=$navi"
'''


def rsync_from_capture_pcs(save_path, pc_list, sessions=None, move_all=False):
    for pc in pc_list:
        ip = get_pc_ip(pc)
        print(f"[collect] {pc} ({ip}) <- {save_path}"
              + (f" sessions={sessions}" if sessions else " (all sessions)")
              + (" [move_all]" if move_all else ""))
        try:
            subprocess.run(
                ["ssh", "-p", str(SSH_PORT), "-o", "ConnectTimeout=5", f"{pc}@{ip}", "bash -s"],
                input=_remote_rsync_script(save_path, sessions, move_all),
                text=True, check=False, timeout=1800,
            )
        except subprocess.TimeoutExpired:
            print(f"[collect] {pc}: TIMEOUT (still may continue server-side)")


def transcode_to_mp4(save_path, sessions=None):
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
             "-crf", "23", "-an", "-y", mp4],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if r.returncode != 0 or not os.path.exists(mp4):
            print(f"[mp4] FAIL: {avi}")
        else:
            print(f"[mp4] OK   {os.path.basename(mp4)} "
                  f"({os.path.getsize(mp4) // (1024*1024)}MB)")


def collect(save_path, do_rsync=True, do_mp4=True, pc_list=None,
            sessions=None, move_all=False):
    if pc_list is None:
        pc_list = get_pc_list()
    if do_rsync:
        rsync_from_capture_pcs(save_path, pc_list, sessions, move_all)
    if do_mp4:
        # Safety net: transcode any fallback .avi that arrived without an mp4.
        transcode_to_mp4(save_path, sessions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True,
                        help="e.g. teleop_test/01 (the prefix)")
    parser.add_argument("--sessions", nargs="*", default=None,
                        help="session leaf names to limit to (e.g. 2026-05-15_21-02-24). "
                             "Default: all sessions under save_path.")
    parser.add_argument("--move_all", action="store_true",
                        help="also rsync raw .avi (not just mp4) even on success")
    parser.add_argument("--no_rsync", action="store_true",
                        help="skip capture-PC collect, only main-PC fallback transcode")
    parser.add_argument("--no_mp4", action="store_true",
                        help="skip main-PC fallback transcode")
    args = parser.parse_args()

    collect(args.save_path, do_rsync=not args.no_rsync, do_mp4=not args.no_mp4,
            sessions=args.sessions, move_all=args.move_all)
    print("[collect_videos] done")
