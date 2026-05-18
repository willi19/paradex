"""Pull factory kinematic calibration from an xArm and patch paradex URDFs.

Usage:
    python src/calibration/xarm_kinematic_calibration.py --robot_ip 192.168.1.xxx
"""
import argparse
import datetime
import os

from paradex.robot.xarm_kinematic_calib import (
    apply_kinematics_to_urdf,
    read_xarm_kinematic_params,
    save_kinematic_yaml,
)
from paradex.utils.system import network_info

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_URDFS = [
    os.path.join(REPO_ROOT, "rsc/robot/xarm.urdf"),
    os.path.join(REPO_ROOT, "rsc/robot/xarm_allegro.urdf"),
    os.path.join(REPO_ROOT, "rsc/robot/xarm_inspire.urdf"),
]
YAML_DIR = os.path.expanduser("~/shared_data/xarm_kinematics")


def _print_diff(urdf_path: str, diff: dict) -> None:
    print(f"\n[{os.path.basename(urdf_path)}]")
    if not diff:
        print("  (no joints patched — names did not match)")
        return
    for joint_name, d in diff.items():
        dx, dy, dz = d["d_xyz_mm"]
        dr, dp, dyaw = d["d_rpy_deg"]
        print(
            f"  {joint_name}: Δxyz=({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) mm  "
            f"Δrpy=({dr:+.4f}, {dp:+.4f}, {dyaw:+.4f}) deg"
        )
        if max(abs(dx), abs(dy), abs(dz)) > 20.0 or max(abs(dr), abs(dp), abs(dyaw)) > 5.0:
            print(f"    WARNING: large delta on {joint_name} — sanity-check before use.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--robot_ip",
        default=network_info["xarm"]["param"]["ip"],
        help="xArm controller IP (default: from system/current/network.json)",
    )
    p.add_argument(
        "--suffix",
        default=datetime.date.today().isoformat(),
        help="Suffix for YAML filename (default: today's date)",
    )
    p.add_argument(
        "--no_apply",
        action="store_true",
        help="Only save YAML; do not patch URDFs",
    )
    p.add_argument(
        "--urdf_paths",
        nargs="*",
        default=DEFAULT_URDFS,
        help="URDFs to patch in-place (default: xarm.urdf + allegro/inspire combos)",
    )
    args = p.parse_args()

    print(f"Reading kinematic calibration from {args.robot_ip} ...")
    params = read_xarm_kinematic_params(args.robot_ip)
    print(f"  robot: {params['robot_name']} (DOF={params['robot_dof']})")

    yaml_path = os.path.join(
        YAML_DIR, f"{params['robot_name']}_kinematics_{args.suffix}.yaml"
    )
    save_kinematic_yaml(params, yaml_path)
    print(f"Saved calibration to {yaml_path}")

    if args.no_apply:
        print("--no_apply set; skipping URDF patch.")
        return

    for urdf_path in args.urdf_paths:
        if not os.path.exists(urdf_path):
            print(f"\n[skip] {urdf_path} not found")
            continue
        diff = apply_kinematics_to_urdf(urdf_path, params["kinematics"])
        _print_diff(urdf_path, diff)

    print(
        "\nDone. Original URDFs backed up as `*.urdf.original`. "
        "To rollback: `cp xarm.urdf.original xarm.urdf` etc."
    )


if __name__ == "__main__":
    main()
