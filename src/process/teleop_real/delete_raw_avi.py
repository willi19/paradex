"""
Reclaim disk by deleting .avi files for a given save_path.

DESTRUCTIVE. Dry-run by default; pass --yes to actually delete.

Safety guard: an .avi is only deletable if a non-empty .mp4 sibling exists in
the shared NAS videos/ dir (so the footage is never lost, only the bulky avi).

Targets:
    shared   : ~/shared_data/{save_path}/{session}/videos/{serial}.avi
    capture  : ~/captures{1,2}/{save_path}/{session}/raw/videos/{serial}.avi
               (deleted only if the shared .mp4 for that serial/session exists)
"""
import os
import glob
import argparse
import subprocess

from paradex.utils.system import get_pc_list, get_pc_ip
from paradex.utils.path import shared_dir

SSH_PORT = 77


def _shared_mp4_exists(avi_path):
    mp4 = avi_path[:-4] + ".mp4"
    return os.path.exists(mp4) and os.path.getsize(mp4) > 0


def delete_shared(save_path, do_it):
    pattern = os.path.join(shared_dir, save_path, "*", "videos", "*.avi")
    avis = sorted(glob.glob(pattern))
    n = 0
    for avi in avis:
        if not _shared_mp4_exists(avi):
            print(f"[shared] SKIP (no mp4): {os.path.relpath(avi, shared_dir)}")
            continue
        if do_it:
            os.remove(avi)
            print(f"[shared] DELETED {os.path.relpath(avi, shared_dir)}")
        else:
            print(f"[shared] would delete {os.path.relpath(avi, shared_dir)}")
        n += 1
    print(f"[shared] {'deleted' if do_it else 'would delete'} {n} file(s)")


def _remote_delete_script(save_path, do_it):
    # Delete capture-PC raw .avi only when the shared mp4 for the same
    # session/serial exists. mode dry-run unless do_it.
    rm = 'rm -f "$src"' if do_it else 'echo "  would delete $src"'
    return f'''set -u
n=0
for src in $HOME/captures1/{save_path}/*/raw/videos/*.avi $HOME/captures2/{save_path}/*/raw/videos/*.avi; do
  [ -f "$src" ] || continue
  base=${{src#$HOME/captures1/}}
  base=${{base#$HOME/captures2/}}
  rel=${{base/\\/raw\\/videos\\//\\/videos\\/}}
  mp4=$HOME/shared_data/${{rel%.avi}}.mp4
  if [ -s "$mp4" ]; then
    {rm} && n=$((n+1))
  else
    echo "  SKIP (no shared mp4): $src"
  fi
done
echo "[capture $(hostname)] {'deleted' if do_it else 'would delete'} $n file(s)"
'''


def delete_capture(save_path, do_it, pc_list):
    for pc in pc_list:
        ip = get_pc_ip(pc)
        print(f"[capture] {pc} ({ip})")
        subprocess.run(
            ["ssh", "-p", str(SSH_PORT), "-o", "ConnectTimeout=5", f"{pc}@{ip}", "bash -s"],
            input=_remote_delete_script(save_path, do_it),
            text=True, check=False, timeout=600,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True,
                        help="e.g. teleop_test/01")
    parser.add_argument("--target", choices=["shared", "capture", "both"],
                        default="shared",
                        help="which avi copies to delete (default: shared)")
    parser.add_argument("--yes", action="store_true",
                        help="actually delete. Without this, dry-run only.")
    args = parser.parse_args()

    if not args.yes:
        print("=== DRY RUN (pass --yes to actually delete) ===")

    if args.target in ("shared", "both"):
        delete_shared(args.save_path, args.yes)
    if args.target in ("capture", "both"):
        delete_capture(args.save_path, args.yes, get_pc_list())

    print("[delete_raw_avi] done")
