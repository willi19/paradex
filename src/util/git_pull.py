"""Main-PC tool: sync every capture PC's ``~/paradex`` checkout to origin.

Capture PCs run their own checkout, so main-PC edits reach them only after a pull.
This pulls and reports the commit each PC actually ended up on, so a silent failure
can't masquerade as an update.

The pull is ``git fetch`` + ``git reset --hard origin/<branch>`` + ``git clean -fd``
on the capture PC — it takes what is on **origin**, not what is in your working
tree. Uncommitted edits and unpushed commits both stay behind on this machine, so
the script checks for them and aborts (override with ``--force``).

Pull only. Starting/restarting capture-PC processes is a separate concern:
``src/camera/reset_cameras.py`` for the camera daemons.

Usage:
    python src/util/git_pull.py                       # all PCs, current branch
    python src/util/git_pull.py --branch dev
    python src/util/git_pull.py --pc_list capture1 capture3
    python src/util/git_pull.py --force               # ignore the dirty/unpushed check
"""
import argparse
import subprocess

from paradex.io.capture_pc.ssh import ssh_port, repo_path
from paradex.utils.system import get_pc_list, get_pc_ip


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

    # --ignore-submodules=all: a submodule pointer/worktree diff is not code the
    # capture PCs would receive, and it otherwise blocks every pull forever.
    dirty = _sh("git status --porcelain --untracked-files=all "
                "--ignore-submodules=all").stdout.strip().splitlines()
    # Build artifacts are noise too.
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


def main():
    parser = argparse.ArgumentParser(description="Pull latest code on every capture PC.")
    parser.add_argument("--pc_list", type=str, nargs="+", default=None,
                        help="Capture PCs to update (default: all from system config).")
    parser.add_argument("--branch", type=str, default=None,
                        help="Branch to reset capture PCs to (default: current local branch).")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even with uncommitted/unpushed local changes.")
    args = parser.parse_args()

    pc_list = args.pc_list if args.pc_list else get_pc_list()
    branch = args.branch or local_branch()
    print(f"[pull] branch: {branch} | capture PCs: {pc_list}")

    if not check_pushed(branch) and not args.force:
        print("[pull] aborted. Commit + push first, or re-run with --force.")
        return

    pull(pc_list, branch)


if __name__ == "__main__":
    main()
