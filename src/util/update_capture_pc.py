"""Main-PC tool: sync every capture PC's ``~/paradex`` checkout to origin.

Capture PCs run their own checkout, so main-PC edits to capture-PC code
(``client.py`` scripts, ``pyspin.py``, ``camera.py``, ``camera_loader.py``, ...)
do nothing until each PC pulls. This pulls and reports the commit each PC actually
ended up on, so a silent failure can't masquerade as an update.

The pull is ``git fetch`` + ``git reset --hard origin/<branch>`` + ``git clean -fd``
on the capture PC — it takes what is on **origin**, not what is in your working
tree. Commit and push first; the script refuses to run if the local branch has
commits origin doesn't have (override with ``--force``).

Scripts launched per run via ``run_script`` (the ``*_client.py`` files) pick the new
code up on their next launch. A long-lived ``server_daemon.py`` keeps running the
code it started with — pass ``--restart`` when a change touches daemon-side camera
code and you need it live now.

Usage:
    python src/util/update_capture_pc.py                       # pull, all PCs
    python src/util/update_capture_pc.py --branch dev
    python src/util/update_capture_pc.py --pc_list capture1 capture3
    python src/util/update_capture_pc.py --restart             # also kill/relaunch the daemons
    python src/util/update_capture_pc.py --force               # skip the unpushed-commit check
"""
import argparse
import subprocess
import time

from paradex.io.capture_pc.ssh import run_script, ssh_port, repo_path
from paradex.utils.system import get_pc_list, get_pc_ip

DAEMON = "src/camera/server_daemon.py"
MONITOR = "src/camera/monitor_daemon.py"


def _sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def local_branch():
    r = _sh("git rev-parse --abbrev-ref HEAD")
    return r.stdout.strip() or "main"


def check_pushed(branch):
    """Warn (and return False) if what's on origin isn't what you're testing.

    Capture PCs do ``git reset --hard origin/<branch>``: uncommitted edits and
    unpushed commits both stay behind on this machine, which reads as "I fixed it
    but nothing changed". Check both.
    """
    ok = True

    dirty = _sh("git status --porcelain --untracked-files=all").stdout.strip().splitlines()
    # Only source matters; build artifacts and submodule pointers are noise here.
    dirty = [l for l in dirty if not l.split(maxsplit=1)[-1].startswith(("paradex.egg-info/",))]
    if dirty:
        print(f"[warn] {len(dirty)} uncommitted change(s) in the working tree — capture PCs "
              f"hard-reset to origin, so these will NOT arrive:")
        for line in dirty[:10]:
            print(f"           {line}")
        if len(dirty) > 10:
            print(f"           ... and {len(dirty) - 10} more")
        print(f"[warn] commit + push them first.")
        ok = False

    _sh("git fetch origin --quiet")
    r = _sh(f"git rev-list --count origin/{branch}..{branch}")
    if r.returncode != 0:
        print(f"[warn] cannot compare against origin/{branch}: {r.stderr.strip()}")
        return False
    ahead = int(r.stdout.strip() or 0)
    if ahead:
        print(f"[warn] local {branch} is {ahead} commit(s) ahead of origin/{branch}. "
              f"Capture PCs pull from origin, so those commits will NOT arrive.")
        print(f"[warn] push first:  git push origin {branch}")
        ok = False
    return ok


def pull(pc_list, branch):
    """git fetch + hard reset to origin/<branch> on each capture PC."""
    ok = {}
    for pc_name in pc_list:
        ip = get_pc_ip(pc_name)
        remote_cmd = (
            f"cd {repo_path} && "
            f"git fetch origin --quiet && "
            f"git reset --hard origin/{branch} --quiet && "
            f"git clean -fdq && "
            f"git log -1 --format='%h %s'"
        )
        r = _sh(f"ssh -p {ssh_port} {pc_name}@{ip} \"{remote_cmd}\"")
        if r.returncode == 0:
            head = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
            print(f"[{pc_name}] -> {head}")
            ok[pc_name] = head
        else:
            print(f"[{pc_name}] FAILED: {r.stderr.strip() or r.stdout.strip()}")
            ok[pc_name] = None
    return ok


def restart_daemons(pc_list):
    """SIGKILL + relaunch the camera daemons (same path as reset_cameras.py)."""
    for pc_name in pc_list:
        ip = get_pc_ip(pc_name)
        remote_cmd = (f"pkill -9 -f 'python.*{DAEMON}'; "
                      f"pkill -9 -f 'python.*{MONITOR}'; sleep 0.5")
        r = _sh(f"ssh -p {ssh_port} {pc_name}@{ip} \"{remote_cmd}\"")
        # pkill returns 1 when nothing matched — not an error for us.
        print(f"[{pc_name}] killed camera daemons (rc={r.returncode})")

    time.sleep(1.0)
    print("[update] relaunching server_daemon.py ...")
    run_script("python src/camera/server_daemon.py", pc_list)


def verify_running(pc_list):
    for pc_name in pc_list:
        ip = get_pc_ip(pc_name)
        r = _sh(f"ssh -p {ssh_port} {pc_name}@{ip} \"pgrep -fc 'python.*{DAEMON}'\"")
        n = (r.stdout or "0").strip()
        print(f"[{pc_name}] server_daemon processes: {n}")


def main():
    parser = argparse.ArgumentParser(description="Pull latest code on every capture PC.")
    parser.add_argument("--pc_list", type=str, nargs="+", default=None,
                        help="Capture PCs to update (default: all from system config).")
    parser.add_argument("--branch", type=str, default=None,
                        help="Branch to reset capture PCs to (default: current local branch).")
    parser.add_argument("--restart", action="store_true",
                        help="Also kill/relaunch server_daemon.py (needed for daemon-side camera code).")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even if the local branch has unpushed commits.")
    args = parser.parse_args()

    pc_list = args.pc_list if args.pc_list else get_pc_list()
    branch = args.branch or local_branch()
    print(f"[update] branch: {branch} | capture PCs: {pc_list}")

    if not check_pushed(branch) and not args.force:
        print("[update] aborted. Push your commits, or re-run with --force.")
        return

    results = pull(pc_list, branch)
    updated = [pc for pc, head in results.items() if head]
    if not updated:
        print("[update] nothing pulled successfully.")
        return

    if not args.restart:
        print("[update] done. Per-run scripts pick this up on their next launch; a "
              "running server_daemon.py keeps its old code (--restart to reload it).")
        return

    restart_daemons(updated)
    time.sleep(2.0)
    verify_running(updated)
    print("[update] done. Daemon logs: ~/shared_data/logs/camera_daemon_<hostname>.log")


if __name__ == "__main__":
    main()
