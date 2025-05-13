import numpy as np
from copy import deepcopy as copy
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from ..constants import *
from ..utils.timer import FrequencyTimer
from ..robot.allegro_retargeters import (
    AllegroKDLControl,
    AllegroJointControl,
)
from ..utils.vectorops import *
from ..utils.files import *


from scipy.spatial.transform import Rotation, Slerp
from ..calibrators.allegro import OculusThumbBoundCalibrator
from ..constants import (
    VR_FREQ,
    ARM_LOW_RESOLUTION,
    ARM_HIGH_RESOLUTION,
    ARM_TELEOP_STOP,
    ARM_TELEOP_CONT,
    OCULUS_JOINTS,
)

# np.set_printoptions(precision=2, suppress=True)


# Rotation should be filtered when it's being sent
class CompStateFilter:
    def __init__(self, state, comp_ratio=0.6):
        self.pos_state = state[:3]
        self.ori_state = state[3:7]
        self.comp_ratio = comp_ratio
        self.step = 0

    def __call__(self, next_state):
        self.pos_state = self.pos_state[:3] * self.comp_ratio + next_state[:3] * (
            1 - self.comp_ratio
        )
        ori_interp = Slerp(
            [0, 1],
            Rotation.from_quat(np.stack([self.ori_state, next_state[3:7]], axis=0)),
        )
        self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_quat()
        return np.concatenate([self.pos_state, self.ori_state])


class AllegroRetargeter:
    def __init__(
        self,
        finger_configs,
        init_robot_hand_joint_angle_path,
        init_robot_eef_frame_path,
        moving_average_limit,
        allow_rotation=False,
        use_filter=False,
    ):
        self.allow_rotation = allow_rotation
        self.resolution_scale = 1
        self.arm_teleop_state = ARM_TELEOP_STOP

        self.fingertip_solver = AllegroKDLControl()
        self.finger_joint_solver = AllegroJointControl()

        # Adding Allegro Hand Specific things

        self.finger_configs = finger_configs
        self.moving_average_queues = {
            "thumb": [],
            "index": [],
            "middle": [],
            "ring": [],
        }

        # Calibrating to get the thumb bounds
        self._calibrate_bounds()

        # Getting the bounds for the allegro hand
        allegro_bounds_path = get_path_in_package(
            "teleop/components/configs/allegro.yaml"
        )
        self.allegro_bounds = get_yaml_data(allegro_bounds_path)

        self._timer = FrequencyTimer(VR_FREQ)

        # torch.set_num_threads(1)

        self.real = False
        self._robot = "Allegro_Moving_Sim"

        self.is_first_frame = True

        self.use_filter = use_filter
        if use_filter:
            robot_init_cart = self._homo2cart(self.robot_init_H)
            self.comp_filter = CompStateFilter(robot_init_cart, comp_ratio=0.8)

        if allow_rotation:
            self.initial_quat = np.array(
                [-0.27686286, -0.66575766, -0.63895273, 0.26805457]
            )
            self.rotation_axis = np.array([0, 0, 1])

        # Getting the bounds to perform linear transformation
        bounds_file = get_path_in_package("teleop/components/configs/franka.yaml")
        bounds_data = get_yaml_data(bounds_file)

        # Bounds for performing linear transformation
        self.corresponding_robot_axes = bounds_data["corresponding_robot_axes"]
        self.franka_bounds = bounds_data["robot_bounds"]
        self.wrist_bounds = bounds_data["wrist_bounds"]

        # Matrices to reorient the end-effector rotation frames
        self.frame_realignment_matrix = np.array(
            bounds_data["frame_realignment_matrix"]
        ).reshape(3, 3)
        self.rotation_realignment_matrix = np.array(
            bounds_data["rotation_alignment_matrix"]
        ).reshape(3, 3)

        # Frequency timer
        self._timer = FrequencyTimer(VR_FREQ)

        self.direction_counter = 0
        self.current_direction = 0

        # Moving average queues
        self.moving_Average_queue = []
        self.moving_average_limit = moving_average_limit

        self.hand_frames = []

        self.current_robot_hand_joint_angle = np.load(init_robot_hand_joint_angle_path)
        self.current_robot_eef_frame = np.load(init_robot_eef_frame_path)
        # print(init_robot_eef_frame_path)        
        # self.current_robot_eef_frame = self.cart2homo(self.current_robot_eef_frame)
        self.step = 0

    @property
    def timer(self):
        return self._timer

    @property
    def robot(self):
        return self._robot

    def _calibrate_bounds(self):

        calibrator = OculusThumbBoundCalibrator()  # self._host, self._port)
        self.hand_thumb_bounds = (
            calibrator.get_bounds()
        )  # Provides [thumb-index bounds, index-middle bounds, middle-ring-bounds]
        print(f"THUMB BOUNDS IN THE OPERATOR: {self.hand_thumb_bounds}")

    # Hand Teleoperation Specific Functions

    def _get_finger_coords(self, hand_keypoints):
        return dict(
            index=np.vstack(
                [
                    hand_keypoints[0],
                    hand_keypoints[OCULUS_JOINTS["index"]],
                ]
            ),
            middle=np.vstack(
                [
                    hand_keypoints[0],
                    hand_keypoints[OCULUS_JOINTS["middle"]],
                ]
            ),
            ring=np.vstack(
                [
                    hand_keypoints[0],
                    hand_keypoints[OCULUS_JOINTS["ring"]],
                ]
            ),
            thumb=np.vstack(
                [
                    hand_keypoints[0],
                    hand_keypoints[OCULUS_JOINTS["thumb"]],
                ]
            ),
        )

    def _get_3d_thumb_angles(self, thumb_keypoints, curr_angles):

        # We will be using polygon implementations of shapely library to test this
        planar_point = Point(thumb_keypoints)
        planar_thumb_bounds = Polygon(self.hand_thumb_bounds[:4])

        # Get the closest point from the thumb to the point
        # this will return the point if it's inside the bounds
        closest_point = nearest_points(planar_thumb_bounds, planar_point)[0]
        closest_point_coords = [closest_point.x, closest_point.y, thumb_keypoints[2]]
        return self.fingertip_solver.thumb_motion_3D(
            hand_coordinates=closest_point_coords,
            xy_hand_bounds=self.hand_thumb_bounds[:4],
            yz_robot_bounds=self.allegro_bounds["thumb_bounds"][0][
                "projective_bounds"
            ],  # NOTE: We assume there is only one bound now
            z_hand_bound=self.hand_thumb_bounds[4],
            x_robot_bound=self.allegro_bounds["thumb_bounds"][0]["x_bounds"],
            moving_avg_arr=self.moving_average_queues["thumb"],
            curr_angles=curr_angles,
        )

    def _turn_frame_to_homo_mat(self, frame):
        t = frame[0]
        R = frame[1:]
        homo_mat = np.zeros((4, 4))
        homo_mat[:3, :3] = np.transpose(R)
        homo_mat[:3, 3] = t
        homo_mat[3, 3] = 1

        return homo_mat

    def _homo2cart(self, homo_mat):
        # Here we will use the resolution scale to set the translation resolution
        t = homo_mat[:3, 3]
        R = Rotation.from_matrix(homo_mat[:3, :3]).as_quat()

        cart = np.concatenate([t, R], axis=0)

        return cart

    def cart2homo(self, cart):
        import pdb; pdb.set_trace()
        homo = np.zeros((4, 4))
        t = cart[0:3]
        R = Rotation.from_quat(cart[3:]).as_matrix()

        homo[0:3, 3] = t
        homo[:3, :3] = R
        homo[3, :] = np.array([0, 0, 0, 1])
        return homo

    def _get_scaled_cart_pose(self, moving_robot_homo_mat):
        # Get the cart pose without the scaling
        unscaled_cart_pose = self._homo2cart(moving_robot_homo_mat)

        # Get the current cart pose
        current_cart_pose = self.current_robot_eef_frame

        # Get the difference in translation between these two cart poses
        diff_in_translation = unscaled_cart_pose[:3] - current_cart_pose[:3]
        scaled_diff_in_translation = diff_in_translation * self.resolution_scale
        # print('SCALED_DIFF_IN_TRANSLATION: {}'.format(scaled_diff_in_translation))

        scaled_cart_pose = np.zeros(7)
        scaled_cart_pose[3:] = unscaled_cart_pose[3:]  # Get the rotation directly
        scaled_cart_pose[:3] = (
            current_cart_pose[:3] + scaled_diff_in_translation
        )  # Get the scaled translation only

        return scaled_cart_pose

    def _reset_teleop(self, hand_frame, current_eff_frame):
        # Just updates the beginning position of the arm
        print("****** RESETTING TELEOP ****** ")
        # self.robot_frame = (
        #     current_eff_frame  # self.end_eff_position_subscriber.recv_keypoints()
        # )
        self.robot_init_H = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )  # copy(current_eff_frame)

        first_hand_frame = hand_frame  # 4x3
        self.hand_init_H = self._turn_frame_to_homo_mat(first_hand_frame)
        self.hand_init_t = copy(self.hand_init_H[:3, 3])

        self.is_first_frame = False

        return first_hand_frame

    def retarget(
        self,
        transformed_hand_keypoint,
        transformed_hand_frame,
        new_arm_teleop_state=ARM_TELEOP_CONT,
        arm_teleoperation_scale_mode=ARM_HIGH_RESOLUTION,
        log=False,
    ):
        ret = {}
        # ------------------ HAND Joint RETARGETING CODE --------------
        finger_keypoints = self._get_finger_coords(transformed_hand_keypoint)
        desired_joint_angles = np.copy(self.current_robot_hand_joint_angle)
        for finger_name in ["index", "middle", "ring"]:
            desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                finger_type=finger_name,
                finger_joint_coords=finger_keypoints[finger_name],
                curr_angles=desired_joint_angles,
                moving_avg_arr=self.moving_average_queues[finger_name],
            )

        desired_joint_angles = self._get_3d_thumb_angles(
            finger_keypoints["thumb"][-1], desired_joint_angles
        )  # Passing just the tip coordinates

        ret["desired_angles"] = desired_joint_angles
        self.current_robot_hand_joint_angle = desired_joint_angles

        # Moving End Effector Teleoperation Code
        # See if there is a reset in the teleop

        # ---------------------- SET TELEOPERATION_STATE ----------------------
        if self.is_first_frame or (new_arm_teleop_state == ARM_TELEOP_STOP):
            self._reset_teleop(
                transformed_hand_frame,
                self.current_robot_eef_frame,
            )
            # Should get the moving hand frame only once

        self.arm_teleop_state = new_arm_teleop_state

        if arm_teleoperation_scale_mode == ARM_HIGH_RESOLUTION:
            self.resolution_scale = 1.0
        elif arm_teleoperation_scale_mode == ARM_LOW_RESOLUTION:
            self.resolution_scale = 0.6

        # ---------------------- END EFFECTOR RETARGETING CODE ----------------------
        self.hand_moving_H = self._turn_frame_to_homo_mat(transformed_hand_frame)
        # Transformation code
        H_HI_HH = copy(
            self.hand_init_H
        )  # Homo matrix that takes P_HI to P_HH - Point in Inital Hand Frame to Point in Home Hand Frame
        H_HT_HH = copy(self.hand_moving_H)  # Homo matrix that takes P_HT to P_HH
        H_RI_RH = copy(self.robot_init_H)  # Homo matrix that takes P_RI to P_RH

        # H_R_V = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # H_T_V = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # H_HT_HI = (
        #     np.linalg.pinv(H_HI_HH) @ H_HT_HH
        # )  # Homo matrix that takes P_HT to P_HI
        # print(H_HI_HH)
        # H_HT_HI_r = H_HT_HI[:3, :3]  # (np.linalg.pinv(H_R_V) @ H_HT_HI @ H_R_V)[:3, :3]
        # H_HT_HI_t = H_HT_HI[:3, 3]  # (np.linalg.pinv(H_T_V) @ H_HT_HI @ H_T_V)[:3, 3]
        # relative_affine = np.block([[H_HT_HI_r, H_HT_HI_t.reshape(3, 1)], [0, 0, 0, 1]])
        # H_RT_RH = H_RI_RH @ H_HT_HI  # relative_affine

        # H_HT_HI = H_HT_HH @ np.linalg.inv(H_HI_HH)
        # H_HT_HI_r = H_HT_HI[:3, :3]
        # H_HT_HI_t = H_HT_HI[:3, 3]
        # relative_affine = np.block([[H_HT_HI_r, H_HT_HI_t.reshape(3, 1)], [0, 0, 0, 1]])
        # H_RT_RH = H_HT_HI @ H_RI_RH  # relative_affine
        # self.robot_moving_H = copy(H_RT_RH)
        UNITY2ISAAC = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])

        delta_R = (
            UNITY2ISAAC
            @ H_HT_HH[:3, :3]
            @ np.linalg.inv(H_HI_HH[:3, :3])
            @ np.linalg.inv(UNITY2ISAAC)
        )
        # delta_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ delta_R
        # delta_R = UNITY2ISAAC @ delta_R @ np.linalg.inv(UNITY2ISAAC)
        self.step += 1
        bias = Rotation.from_euler("xyz", [0, self.step, 0], degrees=True).as_matrix()
        # delta_R = bias @ delta_R
        # delta_R = np.linalg.inv(delta_R)
        delta_t = H_HT_HH[:3, 3] - H_HI_HH[:3, 3]
        UNITY2ISAAC = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
        delta_t = UNITY2ISAAC @ delta_t
        H_RT_RH = np.eye(4)
        H_RT_RH[:3, :3] = delta_R  # @ H_RI_RH[:3, :3]
        H_RT_RH[:3, 3] = delta_t  # + H_RI_RH[:3, 3]
        self.robot_moving_H = copy(H_RT_RH)
        # Use the resolution scale to get the final cart pose
        final_pose = self._homo2cart(
            self.robot_moving_H
        )  # self._get_scaled_cart_pose(self.robot_moving_H)

        # if self.use_filter:
        #     final_pose = self.comp_filter(final_pose)

        ret["endeff_coords"] = self.robot_moving_H
        ret["init_hand_frame"] = self.hand_init_H
        ret["transformed_hand_frame"] = self.hand_moving_H
        ret["robot_init_H"] = self.robot_init_H

        self.current_robot_eef_frame = copy(H_RT_RH)

        # print(self.robot_moving_H, "robot_moving_H")
        # print(H_RI_RH, "H_RI_RH")
        # # print(relative_affine, "relative_affine")
        # # print(H_HT_HI, "H_HT_HI")
        # print(H_HT_HH, "H_HT_HH")

        # ------------------------------------------------------------------------------
        if log:
            print(
                "** ROBOT MOVING H **\n{}\n** ROBOT INIT H **\n{}\n".format(
                    self.robot_moving_H, self.robot_init_H
                )
            )
            print(
                "** HAND MOVING H: **\n{}\n** HAND INIT H: **\n{} - HAND INIT T: {}".format(
                    self.hand_moving_H, self.hand_init_H, self.hand_init_t
                )
            )
            print("***** TRANSFORM MULT: ******\n{}".format(H_HT_HI))

            print("\n------------------------------------\n\n\n")
        return ret
