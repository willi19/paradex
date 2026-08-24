import os
import cv2
import argparse
import numpy as np
import tqdm
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy

from paradex.utils.file_io import find_latest_directory

from paradex.calibration.utils import (
    handeye_calib_bimanual_path,
    handeye_calib_path,
    load_camparam,
)
from paradex.calibration.Tsai_Lenz import solve_ax_xb, solve_axb_cpu    
from paradex.robot.utils import get_robot_urdf_path
from paradex.robot.robot_wrapper_deprecated import RobotWrapper
from paradex.image.image_dict import ImageDict
from paradex.image.aruco import merge_charuco_detection, find_common_indices, detect_charuco
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.visualization.robot import RobotModule

EXCLUDED_SERIALS = {
    # "25452066",
    # "25452062",
    # # capture12 (excluded due to PC issue)
    # "23022639",
    # "22684737",
    # "23173282",
    # "22684210",
}
DEFAULT_CHARUCO_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_DEBUG_WORKERS = max(1, os.cpu_count() or 1)


def _filter_images_for_calibration(images, intrinsics=None, extrinsics=None):
    filtered = {}
    for serial_num, image in images.items():
        if serial_num in EXCLUDED_SERIALS:
            continue
        if intrinsics is not None and serial_num not in intrinsics:
            continue
        if extrinsics is not None and serial_num not in extrinsics:
            continue
        filtered[serial_num] = image
    return filtered


def _detect_charuco_images(img_dict, executor):
    serials = list(img_dict.images)
    detections = executor.map(
        detect_charuco,
        (img_dict.images[serial] for serial in serials),
    )
    return dict(zip(serials, detections))


def undistort_and_detect_charuco(
    root_dir,
    charuco_workers=DEFAULT_CHARUCO_WORKERS,
):
    if charuco_workers < 1:
        raise ValueError("charuco_workers must be at least 1")

    img_dict = None
    index_list = sorted(os.listdir(root_dir))
    intrinsic, extrinsic = load_camparam(os.path.join(root_dir, "0"))

    with ThreadPoolExecutor(max_workers=charuco_workers) as executor:
        for index in tqdm.tqdm(index_list, desc="Undistort and detect charuco"):
            print(f"Processing index {index}...")
            if os.path.exists(os.path.join(root_dir, index, "charuco_3d_ids.npy")) and \
               os.path.exists(os.path.join(root_dir, index, "charuco_3d_corners.npy")):
                continue

            if os.path.exists(os.path.join(root_dir, index, "undistort", "images")) and \
                len(os.listdir(os.path.join(root_dir, index, "undistort", "images"))) == \
                len(os.listdir(os.path.join(root_dir, index, "images"))):
                continue

            os.makedirs(os.path.join(root_dir, index, "undistort", "images"), exist_ok=True)
            if img_dict is None:
                img_dict = ImageDict.from_path(os.path.join(root_dir, index))
                img_dict.images = _filter_images_for_calibration(
                    img_dict.images, intrinsics=intrinsic, extrinsics=extrinsic
                )
                img_dict.set_camparam(intrinsic, extrinsic)
            else:
                img_dict.update_path(os.path.join(root_dir, index))
                img_dict.images = _filter_images_for_calibration(
                    img_dict.images, intrinsics=intrinsic, extrinsics=extrinsic
                )
            if len(img_dict.images) == 0:
                print(f"No valid cameras after exclusion for index {index}, skipping.")
                continue

            print(f"Undistorting and detecting charuco for index {index}...")
            undistort_img_dict = img_dict.undistort(save_path=os.path.join(root_dir, index, "undistort"))

            charuco_2d = _detect_charuco_images(undistort_img_dict, executor)
            charuco_3d = undistort_img_dict.triangulate_charuco(
                detections=charuco_2d
            )
            charuco_3d = merge_charuco_detection(charuco_3d)

            detection = {}
            for serial in charuco_2d:
                detection[serial] = merge_charuco_detection(charuco_2d[serial])['checkerCorner']

            detectionDict = undistort_img_dict.draw_keypoint(detection, color=(0,255,0))
            print(charuco_3d['checkerCorner'].shape)
            if len(charuco_3d['checkerCorner'])==0:
                print(f"No charuco corners detected for index {index}, skipping saving detections.")
                continue
            projected_dict = undistort_img_dict.project_pointcloud(charuco_3d['checkerCorner'])
            detectionDict = detectionDict.draw_keypoint(projected_dict, color=(255,0,0))
            detectionDict.save(os.path.join(root_dir, index, "detection"))

            np.save(os.path.join(root_dir, index, "charuco_3d_ids.npy"), charuco_3d['checkerIDs'])
            np.save(os.path.join(root_dir, index, "charuco_3d_corners.npy"), charuco_3d['checkerCorner'])

def compute_fk(root_dir, arm):
    index_list = sorted(
        (
            index
            for index in os.listdir(root_dir)
            if index.isdigit() and os.path.isdir(os.path.join(root_dir, index))
        ),
        key=int,
    )

    robot_wrapper = RobotWrapper(get_robot_urdf_path(arm_name=arm))

    for index in index_list:
        qpos = np.load(os.path.join(root_dir, index, "qpos.npy"))
        eef = robot_wrapper.compute_forward_kinematics(qpos, link_list=["link6"])['link6']
        np.save(os.path.join(root_dir, index, "eef_fk.npy"), eef)


def get_valid_indices(root_dir):
    """Get indices that have valid charuco detection files with actual points."""
    index_list = sorted(os.listdir(root_dir))
    valid_indices = []
    for index in index_list:
        index_path = os.path.join(root_dir, index)
        if not os.path.isdir(index_path):
            continue
        ids_path = os.path.join(index_path, "charuco_3d_ids.npy")
        corners_path = os.path.join(index_path, "charuco_3d_corners.npy")
        if os.path.exists(ids_path) and os.path.exists(corners_path):
            # Check if files have actual points (not empty)
            ids = np.load(ids_path)
            if len(ids) == 0:
                print(f"Skipping index {index}: empty charuco detection")
                continue
            valid_indices.append(index)
        else:
            print(f"Skipping index {index}: missing charuco detection files")
    return valid_indices


def validate_capture_directory(root_dir):
    cam_param_dir = os.path.join(root_dir, "0", "cam_param")
    required_cam_params = ("intrinsics.json", "extrinsics.json")
    missing_cam_params = [
        name
        for name in required_cam_params
        if not os.path.isfile(os.path.join(cam_param_dir, name))
    ]
    if missing_cam_params:
        raise ValueError(
            f"Incomplete calibration capture in {root_dir}: missing camera "
            f"parameters {missing_cam_params}"
        )

    index_list = sorted(
        name
        for name in os.listdir(root_dir)
        if name.isdigit() and os.path.isdir(os.path.join(root_dir, name))
    )
    if len(index_list) < 2:
        raise ValueError(
            f"Incomplete calibration capture in {root_dir}: "
            f"need at least 2 captured steps, got {len(index_list)}"
        )

    for index in index_list:
        index_path = os.path.join(root_dir, index)
        missing = [
            name
            for name in ("qpos.npy", "eef.npy", "images")
            if not os.path.exists(os.path.join(index_path, name))
        ]
        if missing:
            raise ValueError(
                f"Incomplete calibration capture at {index_path}: missing {missing}"
            )


def compute_motion(root_dir):
    motion_wrt_cam = []
    motion_wrt_robot = []

    index_list = get_valid_indices(root_dir)

    if len(index_list) < 2:
        raise ValueError(f"Need at least 2 valid indices for motion computation, got {len(index_list)}")

    eef_list = [np.load(os.path.join(root_dir, index, "eef_fk.npy")) for index in index_list]

    charuco_id_list = [np.load(os.path.join(root_dir, index, "charuco_3d_ids.npy")) for index in index_list]
    charuco_cor_list = [np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy")) for index in index_list]
    
    for i in range(1, len(index_list)):
        eef = eef_list[i]
        eef_prev = eef_list[i-1]

        motion_wrt_robot.append(eef_prev @ np.linalg.inv(eef)) #M1 R1 R2 M2
        
        ids = charuco_id_list[i]
        ids_prev = charuco_id_list[i-1]
        
        common_idx, common_idx_prev = find_common_indices(ids, ids_prev)

        cam_cor = charuco_cor_list[i][common_idx]
        cam_cor_prev = charuco_cor_list[i-1][common_idx_prev]
        motion_wrt_cam.append(SOLVE_XA_B(cam_cor, cam_cor_prev)) #M1 C1  C2 M2
        # print(np.linalg.norm((eef_prev - eef)[:3, 3]), np.linalg.norm(np.mean(cam_cor - cam_cor_prev, axis=0)))
        # import pdb; pdb.set_trace()
        err = cam_cor_prev - (motion_wrt_cam[-1][:3, :3] @ cam_cor.T).T - motion_wrt_cam[-1][:3, 3]
        print(f"Motion {i-1}->{i} cam points fitting error: {np.mean(np.linalg.norm(err, axis=1))*1000:.2f} mm")
    
    return motion_wrt_cam, motion_wrt_robot

def debug(root_dir, arm, robot_wrt_cam_world):
    index_list = get_valid_indices(root_dir)
    marker_pos = {}
    
    rm = RobotModule(get_robot_urdf_path(arm_name=arm))
    intrinsic, extrinsic = load_camparam(os.path.join(root_dir, "0"))
    
    cam_world_wrt_robot = np.linalg.inv(robot_wrt_cam_world)
    for index in index_list:
        eef = np.load(os.path.join(root_dir, index, "eef_fk.npy"))
        eef_from_robot = np.load(os.path.join(root_dir, index, "eef.npy"))
        # FK error
        eef_error = np.linalg.inv(eef) @ eef_from_robot
        trans_error = np.linalg.norm(eef_error[:3, 3]) * 1000
        
        # Rotation error (degrees)
        R_error = eef_error[:3, :3]
        angle_error = np.arccos((np.trace(R_error) - 1) / 2) * 180 / np.pi
        
        print(f"fk error {index}: trans={trans_error:.2f}mm, rot={angle_error:.2f}deg")
        
        charuco_3d_cor = np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy"))
        charuco_id_cor = np.load(os.path.join(root_dir, index, "charuco_3d_ids.npy"))
        
        marker_h = np.column_stack(
            (charuco_3d_cor, np.ones(len(charuco_3d_cor)))
        )
        marker_in_eef = (
            np.linalg.inv(eef) @ cam_world_wrt_robot @ marker_h.T
        ).T[:, :3]
        for mid, cor in zip(charuco_id_cor, marker_in_eef):
            if mid not in marker_pos:
                marker_pos[mid] = []
            marker_pos[mid].append(cor)

    marker_wrt_eef = []
    for mid in marker_pos:
        pos = np.array(marker_pos[mid])
        mean_pos = np.mean(pos, axis=0)
        std_pos = np.std(pos, axis=0)
        print(f"Marker {mid}: std={std_pos}, mean={mean_pos}")

        marker_wrt_eef.append(mean_pos)
    marker_wrt_eef = np.array(marker_wrt_eef)
    
    render_robot_debug_parallel(
        root_dir,
        index_list,
        rm,
        intrinsic,
        extrinsic,
        robot_wrt_cam_world,
        marker_wrt_eef,
    )


def project_debug_points(points, projection_matrix):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (projection_matrix @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


def save_robot_debug_image(
    image_path,
    output_path,
    robot_mask,
    fitted_markers,
    observed_markers,
):
    from paradex.image.overlay import overlay_mask

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read undistorted image: {image_path}")
    image = overlay_mask(image, robot_mask, color=(0, 255, 0), alpha=0.5)
    for point in fitted_markers:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 3, (255, 0, 0), -1)
    for point in observed_markers:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 3, (0, 0, 255), -1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not cv2.imwrite(
        str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]
    ):
        raise IOError(f"Failed to save robot debug image: {output_path}")


def _finish_debug_futures(pending, progress, wait_for_one):
    if not pending:
        return pending
    if wait_for_one:
        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
    else:
        completed, pending = pending, set()
    for future in completed:
        future.result()
        progress.update(1)
    return pending


def render_robot_debug_parallel(
    root_dir,
    index_list,
    robot_module,
    intrinsic,
    extrinsic,
    robot_wrt_cam_world,
    marker_wrt_eef,
    debug_workers=DEFAULT_DEBUG_WORKERS,
):
    from paradex.image.projection import BatchRenderer

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    serials_by_index = {}
    for index in index_list:
        raw_image_dir = os.path.join(root_dir, index, "images")
        undistort_image_dir = os.path.join(
            root_dir, index, "undistort", "images"
        )
        raw_serials = {
            os.path.splitext(filename)[0]
            for filename in os.listdir(raw_image_dir)
            if filename.lower().endswith(image_extensions)
        }
        undistorted_serials = {
            os.path.splitext(filename)[0]
            for filename in os.listdir(undistort_image_dir)
            if filename.lower().endswith(image_extensions)
        }
        serials_by_index[index] = raw_serials & undistorted_serials

    available_serials = set().union(*serials_by_index.values())
    serials = sorted(
        ((set(intrinsic) & set(extrinsic)) - set(EXCLUDED_SERIALS))
        & available_serials
    )
    if not serials:
        raise ValueError("No camera images are available for robot debug rendering")
    filtered_intrinsic = {serial: intrinsic[serial] for serial in serials}
    filtered_extrinsic = {serial: extrinsic[serial] for serial in serials}
    renderer = BatchRenderer(filtered_intrinsic, filtered_extrinsic)
    projection_matrices = {
        serial: filtered_intrinsic[serial]["intrinsics_undistort"]
        @ filtered_extrinsic[serial]
        for serial in serials
    }
    marker_wrt_eef_h = np.column_stack(
        (marker_wrt_eef, np.ones(len(marker_wrt_eef)))
    )
    worker_count = min(debug_workers, max(1, len(index_list) * len(serials)))
    max_pending = max(1, worker_count * 2)
    previous_opencv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending = set()
            total = sum(
                len(set(serials) & serials_by_index[index])
                for index in index_list
            )
            save_progress = tqdm.tqdm(total=total, desc="Saving robot debug images")
            try:
                for index in tqdm.tqdm(index_list, desc="Rendering robot debug"):
                    qpos = np.load(os.path.join(root_dir, index, "qpos.npy"))
                    eef = np.load(os.path.join(root_dir, index, "eef_fk.npy"))
                    robot_module.update_cfg(qpos)
                    robot_mesh = robot_module.get_robot_mesh()
                    robot_mesh.apply_transform(robot_wrt_cam_world)
                    masks = renderer.render_mask(robot_mesh)

                    marker_wrt_cam = (
                        robot_wrt_cam_world @ eef @ marker_wrt_eef_h.T
                    ).T[:, :3]
                    marker_3d = np.load(
                        os.path.join(root_dir, index, "charuco_3d_corners.npy")
                    )
                    for serial in sorted(set(serials) & serials_by_index[index]):
                        image_path = os.path.join(
                            root_dir, index, "undistort", "images", f"{serial}.png"
                        )
                        output_path = os.path.join(
                            root_dir, index, "debug", "images", f"{serial}.png"
                        )
                        pending.add(
                            executor.submit(
                                save_robot_debug_image,
                                image_path,
                                output_path,
                                masks[serial],
                                project_debug_points(
                                    marker_wrt_cam, projection_matrices[serial]
                                ),
                                project_debug_points(
                                    marker_3d, projection_matrices[serial]
                                ),
                            )
                        )
                        if len(pending) >= max_pending:
                            pending = _finish_debug_futures(
                                pending, save_progress, wait_for_one=True
                            )
                _finish_debug_futures(pending, save_progress, wait_for_one=False)
            finally:
                save_progress.close()
    finally:
        cv2.setNumThreads(previous_opencv_threads)
                
def calculate_sequence(
    root_path,
    arm,
    save_path,
    charuco_workers=DEFAULT_CHARUCO_WORKERS,
    precomputed_charuco=False,
):
    validate_capture_directory(root_path)
    if not precomputed_charuco:
        undistort_and_detect_charuco(root_path, charuco_workers)
    compute_fk(root_path, arm)
    motion_wrt_cam, motion_wrt_robot = compute_motion(root_path)
    robot_wrt_cam_world = solve_ax_xb(
        motion_wrt_cam,
        motion_wrt_robot,
        verbose=True,
    )

    valid_index_list = get_valid_indices(root_path)
    for i in range(len(valid_index_list) - 1):
        diff = (
            motion_wrt_cam[i] @ robot_wrt_cam_world
            - robot_wrt_cam_world @ motion_wrt_robot[i]
        )
        trans_error = np.linalg.norm(diff[:3, 3]) * 1000
        angle_error = 0
        print(f"Motion {i}: trans={trans_error:.2f}mm, rot={angle_error:.2f}deg")

    if save_path is None:
        save_path = os.path.join(root_path, valid_index_list[0], "C2R.npy")
    np.save(save_path, robot_wrt_cam_world)
    print(f"Saved C2R to {save_path}")
    debug(root_path, arm, robot_wrt_cam_world)


def get_bimanual_arm_name(arm, side):
    if arm == "xarm" and side == "Left":
        return "xarm_left"
    return arm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")
    parser.add_argument("--arm", type=str, default="xarm", help="Name of the robot arm.")
    parser.add_argument(
        "--bimanual",
        action="store_true",
        help="Calculate right and left C2R transforms from a bimanual session.",
    )
    parser.add_argument(
        "--charuco-workers",
        type=int,
        default=DEFAULT_CHARUCO_WORKERS,
        help="Number of camera images to process in parallel for ChArUco detection.",
    )
    args = parser.parse_args()

    base_dir = handeye_calib_bimanual_path if args.bimanual else handeye_calib_path
    if args.name is None:
        args.name = find_latest_directory(base_dir)

    root_path = os.path.join(base_dir, args.name)
    if args.bimanual:
        side_paths = {}
        for side, suffix in (("Right", "R"), ("Left", "L")):
            side_path = os.path.join(root_path, side)
            if not os.path.isdir(side_path):
                raise FileNotFoundError(
                    f"Missing {side} capture directory: {side_path}"
                )
            validate_capture_directory(side_path)
            side_paths[side] = (side_path, suffix)

        for side, (side_path, suffix) in side_paths.items():
            print(f"Calculating {side} xArm calibration...")
            calculate_sequence(
                side_path,
                get_bimanual_arm_name(args.arm, side),
                os.path.join(root_path, f"C2R_{suffix}.npy"),
                args.charuco_workers,
            )
    else:
        calculate_sequence(root_path, args.arm, None, args.charuco_workers)


if __name__ == "__main__":
    main()
