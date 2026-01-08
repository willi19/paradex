import time
import numpy as np
import mujoco
import mujoco.viewer

from paradex.io.teleop.xsens.receiver import XSensReceiver
from paradex.utils.system import network_info

from .retargetor import OpenArmRetargetor


def body_pose(data: mujoco.MjData, body_id: int) -> np.ndarray:
    pos = data.xpos[body_id]
    rot = data.xmat[body_id].reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def build_actuator_map(model: mujoco.MjModel):
    mapping = []
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id][0]
        qpos_idx = model.jnt_qposadr[joint_id]
        qvel_idx = model.jnt_dofadr[joint_id]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
        mapping.append(
            {
                "act_id": act_id,
                "joint_id": joint_id,
                "qpos_idx": qpos_idx,
                "qvel_idx": qvel_idx,
                "name": name,
            }
        )
    return mapping


def apply_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_map,
    target_qpos: np.ndarray,
    kp_arm: float = 120.0,
    kd_arm: float = 4.0,
):
    if target_qpos is None:
        data.ctrl[:] = 0
        return

    for entry in actuator_map:
        act_id = entry["act_id"]
        qpos_idx = entry["qpos_idx"]
        qvel_idx = entry["qvel_idx"]
        name = entry["name"]

        if "finger" in name:
            # Position actuator: ctrl is desired position.
            data.ctrl[act_id] = target_qpos[qpos_idx]
        else:
            err = target_qpos[qpos_idx] - data.qpos[qpos_idx]
            vel = data.qvel[qvel_idx]
            data.ctrl[act_id] = kp_arm * err - kd_arm * vel


def main():
    model = mujoco.MjModel.from_xml_path("v1/openarm_bimanual.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    actuator_map = build_actuator_map(model)

    # XSens receiver
    xsens_cfg = network_info.get("xsens", {}).get("param", {})
    receiver = XSensReceiver(**xsens_cfg)

    # Retargetor
    retargetor = OpenArmRetargetor(model)

    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_hand_tcp")
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_hand_tcp")
    home_left = body_pose(data, left_body)
    home_right = body_pose(data, right_body)
    retargetor.start({"Left": home_left, "Right": home_right})

    target_qpos = data.qpos.copy()
    smoothing = 0.2  # simple low-pass on qpos target

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                xsens_data = receiver.get_data()
                if xsens_data["Left"] is not None and xsens_data["Right"] is not None:
                    qpos_cmd = retargetor.get_action(xsens_data, data.qpos.copy())
                    if qpos_cmd is not None:
                        target_qpos = smoothing * target_qpos + (1.0 - smoothing) * qpos_cmd

                apply_control(model, data, actuator_map, target_qpos)
                mujoco.mj_step(model, data)
                viewer.sync()
    finally:
        receiver.end()


if __name__ == "__main__":
    main()
