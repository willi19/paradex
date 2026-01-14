from typing import List

import numpy as np
import numpy.typing as npt
import pinocchio as pin

# Used :
# https://github.com/dexsuite/dex-retargeting/blob/main/dex_retargeting/robot_wrapper.py

# Makes it easier to use robot kinematics and dynamics without worrying about Pinocchio's details.

# Follow https://stack-of-tasks.github.io/pinocchio/download.html for pinocchio installation


class RobotWrapper:
    def __init__(self, urdf_path: str):
        # Create robot model and data
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()

        self.q0 = pin.neutral(self.model)
        if self.model.nv != self.model.nq:
            raise NotImplementedError(f"Can not handle robot with special joint.")

        # Build active (non-mimic) joint list and an expansion map for mimic joints.
        self.active_joint_ids = []
        self.active_q_spec = []  # list of (idx_q, nq)
        self.total_active_nq = 0

        # pinocchio provides mimicParent / mimicMultiplier / mimicOffset per joint (length = njoints)
        mimic_parent = getattr(self.model, "mimicParent", [])
        mimic_mult = getattr(self.model, "mimicMultiplier", [])
        mimic_offset = getattr(self.model, "mimicOffset", [])
        nj = len(self.model.joints)

        # Manual mimic map for Inspire URDF (Pinocchio may not populate mimic_* fields)
        manual_mimic = {}
        if "inspire" in urdf_path.lower():
            manual_mimic = {
                "left_thumb_3_joint": ("left_thumb_2_joint", 0.60, 0.0),
                "left_thumb_4_joint": ("left_thumb_2_joint", 0.8, 0.0),
                "left_index_2_joint": ("left_index_1_joint", 1.05, 0.0),
                "left_middle_2_joint": ("left_middle_1_joint", 1.05, 0.0),
                "left_ring_2_joint": ("left_ring_1_joint", 1.05, 0.0),
                "left_little_2_joint": ("left_little_1_joint", 1.18, 0.0),
            }

        # Decide mimic detection source
        use_manual_mimic = bool(manual_mimic) and (len(mimic_parent) != nj or all(mp == 0 for mp in mimic_parent))

        if "inspire" in urdf_path.lower() and use_manual_mimic:
            # Force active joints list for Inspire: 6 xArm + 6 Inspire proximal
            # active_names = [
            #     "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
            #     "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
            #     "index_proximal_joint", "middle_proximal_joint",
            #     "ring_proximal_joint", "pinky_proximal_joint",
            # ]
            active_names = [
                "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
                "left_thumb_1_joint", "left_thumb_2_joint",
                "left_index_1_joint", "left_middle_1_joint", 
                "left_ring_1_joint", "left_little_1_joint",
            ]
            for name in active_names:
                jid = self.model.getJointId(name)
                j = self.model.joints[jid]
                self.active_joint_ids.append(jid)
                self.active_q_spec.append((j.idx_q, j.nq))
                self.total_active_nq += j.nq
        else:
            for jid in range(nj):
                j = self.model.joints[jid]
                if j.nq == 0:
                    continue
                is_mimic = False
                if use_manual_mimic:
                    jname = self.model.names[jid]
                    is_mimic = jname in manual_mimic
                elif len(mimic_parent) == nj:
                    # mimicParent holds the parent joint id; 0 typically means no mimic (universe)
                    # Treat nonzero/positive parent (not self) as mimic.
                    is_mimic = mimic_parent[jid] != 0 and mimic_parent[jid] != jid
                if not is_mimic:
                    self.active_joint_ids.append(jid)
                    self.active_q_spec.append((j.idx_q, j.nq))
                    self.total_active_nq += j.nq

        def expand_qpos(qpos_short: np.ndarray) -> np.ndarray:
            """
            Expand a reduced qpos (active joints only) into full qpos by populating mimic joints.
            """
            full = np.zeros(self.model.nq, dtype=float)
            # fill active joints
            cursor = 0
            for jid, (idx_q, nq) in zip(self.active_joint_ids, self.active_q_spec):
                full[idx_q:idx_q + nq] = qpos_short[cursor:cursor + nq]
                cursor += nq
            # fill mimic joints
            if use_manual_mimic:
                for child_name, (parent_name, mult, offset) in manual_mimic.items():
                    try:
                        child_id = self.model.getJointId(child_name)
                        parent_id = self.model.getJointId(parent_name)
                    except:
                        continue
                    if child_id >= len(self.model.joints) or parent_id >= len(self.model.joints):
                        continue
                    child_j = self.model.joints[child_id]
                    parent_j = self.model.joints[parent_id]
                    full[child_j.idx_q:child_j.idx_q + child_j.nq] = mult * full[parent_j.idx_q:parent_j.idx_q + parent_j.nq] + offset
            elif len(mimic_parent) == nj and len(mimic_mult) == nj and len(mimic_offset) == nj:
                for jid in range(nj):
                    j = self.model.joints[jid]
                    if j.nq == 0:
                        continue
                    parent = mimic_parent[jid] if len(mimic_parent) == nj else 0
                    if parent != 0 and parent != jid:
                        parent_idx = self.model.joints[parent].idx_q
                        nq = j.nq
                        idx_q = j.idx_q
                        mult = mimic_mult[jid] if len(mimic_mult) == nj else 1.0
                        offset = mimic_offset[jid] if len(mimic_offset) == nj else 0.0
                        full[idx_q:idx_q + nq] = mult * full[parent_idx:parent_idx + nq] + offset
            return full

        self._expand_qpos = expand_qpos

    # -------------------------------------------------------------------------- #
    # Robot property
    # -------------------------------------------------------------------------- #
    @property
    def joint_names(self) -> List[str]:
        return list(self.model.names)

    @property
    def dof_joint_names(self) -> List[str]:
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def link_names(self) -> List[str]:
        link_names = []
        for i, frame in enumerate(self.model.frames):
            link_names.append(frame.name)
        return link_names

    @property
    def joint_limits(self):
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    # -------------------------------------------------------------------------- #
    # Query function
    # -------------------------------------------------------------------------- #
    def get_joint_index(self, name: str):
        return self.dof_joint_names.index(name)

    def get_joint_indices(self, names: List[str]):
        print(self.dof_joint_names)
        return [self.get_joint_index(name) for name in names]

    def get_link_index(self, name: str):
        return self.model.getFrameId(name, pin.FrameType.BODY)

    def get_link_indices(self, names: List[str]):
        return [self.get_link_index(name) for name in names]

    def get_frame_list(self):
        return self.model.frames

    # -------------------------------------------------------------------------- #
    # Kinematics function
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics(self, qpos: npt.NDArray):
        if qpos.shape[0] == self.model.nq:
            full_q = qpos
        elif qpos.shape[0] == getattr(self, "total_active_nq", qpos.shape[0]):
            full_q = self._expand_qpos(qpos)
        else:
            raise ValueError(f"qpos length mismatch: got {qpos.shape[0]}, expected {self.model.nq} or {self.total_active_nq} (active DOF)")
        pin.forwardKinematics(self.model, self.data, full_q)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.homogeneous

    def get_link_pose_pino(self, link_id: int) -> npt.NDArray:
        pose = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.inverse().homogeneous

    def get_joint_pose(self, joint_id: int) -> npt.NDArray:
        """Returns the SE3 pose of the joint as a 4x4 homogeneous matrix."""
        pose: pin.SE3 = self.data.oMi[joint_id]
        return pose.homogeneous

    def get_frame_pose(self, frame_id: int) -> npt.NDArray:
        pose: pin.SE3 = self.data.oMf[frame_id]
        return pose.homogeneous

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> npt.NDArray:
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
        return J

    def integrate(self, q, v, dt):
        return pin.integrate(self.model, q, v * dt)

    def get_all_link_poses(self):
        """
        Returns a dictionary with link names as keys and their homogeneous transformation matrices as values.
        """
        link_poses = {}
        for frame in self.model.frames:
            link_id = frame.id
            pose: pin.SE3 = self.data.oMf[link_id]
            link_poses[frame.name] = pose.homogeneous
        return link_poses
