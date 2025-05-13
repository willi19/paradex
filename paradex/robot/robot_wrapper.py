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
        pin.forwardKinematics(self.model, self.data, qpos)

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
