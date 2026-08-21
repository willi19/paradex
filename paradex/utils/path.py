import os

home_path = os.path.expanduser("~")
pc_name = os.path.basename(home_path)
shared_dir = os.path.join(home_path, "shared_data")
capture_path_list = [os.path.join(home_path, f"captures{i}") for i in range(1,3)]

download_dir = os.path.join(home_path, "download")
model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "model")

rsc_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "rsc",
)


def assert_shared_data_mounted(what="this run"):
    """Abort if ``shared_dir`` is an fstab mount point that is not mounted.

    ``shared_data`` is NFS here. When the mount is down the directory still exists
    and stays writable, so every write silently lands on the local disk instead of
    the NAS — invisible again the moment someone remounts. A 25-pose extrinsic
    session was lost that way: the Main PC made its output dirs locally while the
    capture PCs wrote to the real NAS path and hit FileNotFoundError.

    Only fires when /etc/fstab declares the path, so machines that legitimately keep
    ``shared_data`` on local disk are unaffected.
    """
    if os.path.ismount(shared_dir):
        return
    try:
        with open("/etc/fstab") as f:
            declared = any(
                len(parts) > 1 and os.path.realpath(parts[1]) == os.path.realpath(shared_dir)
                for parts in (line.split() for line in f)
                if parts and not parts[0].startswith("#")
            )
    except OSError:
        return                      # no fstab to consult: don't block the user
    if declared:
        raise RuntimeError(
            f"{shared_dir} is an fstab mount point but is NOT mounted — {what} would "
            f"write to the local disk and be lost.\n"
            f"    sudo mount {shared_dir}   (then check: mount | grep shared_data)")
