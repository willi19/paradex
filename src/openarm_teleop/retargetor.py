import numpy as np
import mujoco

from paradex.retargetor.bimanual import Retargetor as BaseRetargetor
from paradex.transforms.coordinate import DEVICE2WRIST


class OpenArmRetargetor(BaseRetargetor):
    """
    OpenArm-specific retargetor that converts XSens wrist poses into
    joint-space targets for the MuJoCo OpenArm bimanual model.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        right_body: str = "openarm_right_hand_tcp",
        left_body: str = "openarm_left_hand_tcp",
        ik_iters: int = 8,
        ik_damping: float = 1e-3,
    ) -> None:
        super().__init__(arm_name="openarm", hand_name="gripper")

        # OpenArm currently uses the xArm wrist alignment as a proxy.
        self.device2wrist = DEVICE2WRIST["xarm"].copy()
        self.device2global = DEVICE2GLOBAL["xarm"].copy()

        self.model = model
        self.ik_data = mujoco.MjData(self.model)
        self.ik_iters = ik_iters
        self.ik_damping = ik_damping

        self.side_info = {
            "Right": {
                "body": right_body,
                "joints": [
                    "openarm_right_joint1",
                    "openarm_right_joint2",
                    "openarm_right_joint3",
                    "openarm_right_joint4",
                    "openarm_right_joint5",
                    "openarm_right_joint6",
                    "openarm_right_joint7",
                ],
                "fingers": [
                    "openarm_right_finger_joint1",
                    "openarm_right_finger_joint2",
                ],
            },
            "Left": {
                "body": left_body,
                "joints": [
                    "openarm_left_joint1",
                    "openarm_left_joint2",
                    "openarm_left_joint3",
                    "openarm_left_joint4",
                    "openarm_left_joint5",
                    "openarm_left_joint6",
                    "openarm_left_joint7",
                ],
                "fingers": [
                    "openarm_left_finger_joint1",
                    "openarm_left_finger_joint2",
                ],
            },
        }

        # Pre-compute useful indices for fast IK updates.
        for side, info in self.side_info.items():
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, info["body"]
            )
            joint_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in info["joints"]
            ]
            finger_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in info["fingers"]
            ]

            info["body_id"] = body_id
            info["joint_ids"] = joint_ids
            info["finger_ids"] = finger_ids
            info["dof_indices"] = [self.model.jnt_dofadr[jid] for jid in joint_ids]
            info["qpos_indices"] = [self.model.jnt_qposadr[jid] for jid in joint_ids]
            info["finger_qpos_indices"] = [
                self.model.jnt_qposadr[jid] for jid in finger_ids
            ]
            info["qpos_limits"] = [
                self.model.jnt_range[jid] if self.model.jnt_limited[jid] else None
                for jid in joint_ids
            ]

    def _rotation_error(self, target_R: np.ndarray, current_R: np.ndarray) -> np.ndarray:
        # 0.5 * (R_current^T * R_target - R_target^T * R_current)_vee
        return 0.5 * (
            np.cross(current_R[:, 0], target_R[:, 0])
            + np.cross(current_R[:, 1], target_R[:, 1])
            + np.cross(current_R[:, 2], target_R[:, 2])
        )

    def _damped_ls(self, J: np.ndarray, err: np.ndarray) -> np.ndarray:
        JJt = J @ J.T
        damping = (self.ik_damping ** 2) * np.eye(JJt.shape[0])
        return J.T @ np.linalg.solve(JJt + damping, err)

    def _solve_ik(
        self,
        side: str,
        target_pose: np.ndarray,
        qpos: np.ndarray,
    ) -> np.ndarray:
        info = self.side_info[side]
        q = qpos.copy()

        for _ in range(self.ik_iters):
            self.ik_data.qpos[:] = q
            mujoco.mj_forward(self.model, self.ik_data)

            body_id = info["body_id"]
            pos = self.ik_data.xpos[body_id]
            rot = self.ik_data.xmat[body_id].reshape(3, 3)

            pos_err = target_pose[:3, 3] - pos
            rot_err = self._rotation_error(target_pose[:3, :3], rot)
            err = np.concatenate([pos_err, rot_err])

            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-4:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.ik_data, jacp, jacr, body_id)

            cols = info["dof_indices"]
            J = np.vstack([jacp[:, cols], jacr[:, cols]])
            dq = self._damped_ls(J, err)

            q_indices = info["qpos_indices"]
            q[q_indices] += dq

            # Enforce joint limits where available.
            for idx, lim in zip(q_indices, info["qpos_limits"]):
                if lim is None:
                    continue
                q[idx] = np.clip(q[idx], lim[0], lim[1])

        return q

    def _finger_target(self, hand_pose: dict) -> float:
        if hand_pose is None:
            return None

        thumb = hand_pose.get("thumb_distal")
        index = hand_pose.get("index_distal")
        if thumb is None or index is None:
            return None

        dist = np.linalg.norm(thumb[:3, 3] - index[:3, 3])
        # Map distance to gripper width. Heuristic: 0.03m -> closed, 0.09m -> open.
        width = 0.04 * (dist - 0.03) / 0.06
        return float(np.clip(width, 0.0, 0.04))

    def get_action(self, data: dict, current_qpos: np.ndarray) -> np.ndarray:
        """
        Returns a full qpos target vector aligned with the model qpos layout.
        """
        if data is None or data.get("Left") is None or data.get("Right") is None:
            return None

        arm_targets, _ = super().get_action(data)
        if arm_targets is None:
            return None

        target_qpos = current_qpos.copy()

        for side in ["Left", "Right"]:
            pose = arm_targets[side]
            if pose is None:
                continue

            target_qpos = self._solve_ik(side, pose, target_qpos)

            # Finger command: same width for both sliders on each side.
            width = self._finger_target(data[side])
            if width is None:
                continue
            for idx in self.side_info[side]["finger_qpos_indices"]:
                target_qpos[idx] = width

        return target_qpos
