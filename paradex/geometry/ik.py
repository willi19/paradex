import numpy as np
import pinocchio as pin
import time

def inverse_kinematics(
    robot,
    target_pose: np.ndarray,
    end_effector_name: str,
    q_init: np.ndarray = None,
    max_iter: int = 100,
    tol: float = 1e-7,
    alpha: float = 1e-2,
) -> np.ndarray:
    q = q_init.copy()
    link_id = robot.get_link_index(end_effector_name)

    target_se3 = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

    for i in range(max_iter):
        robot.compute_forward_kinematics(q)
        current_se3 = pin.updateFramePlacement(robot.model, robot.data, link_id)

        # 6D error: log(target⁻¹ * current)
        error = pin.log(target_se3.inverse() * current_se3).vector
        
        if np.linalg.norm(error) < tol:
            return q

        J = pin.computeFrameJacobian(robot.model, robot.data, q, link_id)
        lambda_ = 1e-6
        JTJ = J.T @ J + lambda_ * np.eye(J.shape[1])
        delta_q = -np.linalg.solve(JTJ, J.T @ error)

        q = robot.integrate(q, delta_q, alpha)

    print("IK did not converge.")
    return q
