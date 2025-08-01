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
        return [
            frame.name
            for frame in self.model.frames
            if frame.type == pin.FrameType.BODY
        ]

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

    def solve_ik(
        self,
        target_pose: npt.NDArray,
        end_effector_name: str,
        q_init: npt.NDArray = None,
        max_iter: int = 1000,
        tol: float = 1e-8,
        alpha: float = 5e-2,
        try_num : int = 30
    ) -> npt.NDArray:
        
        target_pose = target_pose.copy()
        q_min = self.model.lowerPositionLimit
        q_max = self.model.upperPositionLimit
        
        for try_cnt in range(try_num):
            if try_cnt != 0 or q_init is None:
                q = np.random.uniform(low=q_min, high=q_max)
            else:
                q = q_init.copy()
                
            link_id = self.get_link_index(end_effector_name)

            target_se3 = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

            for i in range(max_iter):
                self.compute_forward_kinematics(q)
                current_se3 = pin.updateFramePlacement(self.model, self.data, link_id)
                # 6D error: log(target⁻¹ * current)
                error = pin.log(target_se3.inverse() * current_se3).vector
                if np.linalg.norm(error) < tol:
                    self.compute_forward_kinematics(q)
                    real_pos = self.get_link_pose(link_id)
                    diff_rot = real_pos[:3,:3] @ np.linalg.inv(target_pose[:3,:3]) - np.eye(3)
                    diff_pos = real_pos[:3,3] - target_pose[:3, 3]
                    if np.linalg.norm(diff_rot) < 0.01 and np.linalg.norm(diff_pos) < 0.01:
                        return q, True
                    else:
                        break

                J = pin.computeFrameJacobian(self.model, self.data, q, link_id)
                lambda_ = 1e-6
                JTJ = J.T @ J + lambda_ * np.eye(J.shape[1])
                delta_q = -np.linalg.solve(JTJ, J.T @ error)

                q = self.integrate(q, delta_q, alpha)
                q = np.clip(q, q_min, q_max)
            
        return q, False

    def get_end_links(self) -> List[str]:
        """
        Returns names of links (frames of type BODY) that are not referenced
        as parent by any joint. These are the "end links".
        """
        # 모든 link frame들만 추출 (BODY 타입)
        link_name_to_id = {
            frame.name: frame.parent
            for frame in self.model.frames
            if frame.type == pin.FrameType.BODY
        }

        all_link_names = set(link_name_to_id.keys())
        parent_joint_ids = set(self.model.parents[1:])  # joint 0 (universe)는 제외

        # parent로 등장한 joint들이 연결된 링크 이름들 추출
        parent_link_names = {
            name for name, jid in link_name_to_id.items() if jid in parent_joint_ids
        }

        # end links: parent로 단 한 번도 등장하지 않은 링크
        end_link_names = list(all_link_names - parent_link_names)
        return end_link_names

    
    def get_root_link(self) -> str:
        """Returns the root link name (never appears as a child link)."""
        all_links = set(self.link_names)
        print(all_links)
        child_links = {
            frame.name
            for frame in self.model.frames
            if frame.parent != 0 and frame.type == pin.FrameType.BODY
        }
        root_links = list(all_links - child_links)
        if len(root_links) != 1:
            raise RuntimeError(f"Ambiguous or missing root link(s): {root_links}")
        return root_links[0]