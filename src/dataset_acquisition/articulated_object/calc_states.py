import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parents[3]))


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cv2_imread(path: str, flags: int) -> Optional["np.ndarray"]:
    import cv2
    import numpy as np

    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _cv2_imwrite(path: str, image: "np.ndarray") -> bool:
    import cv2

    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(path)
    return True


def _list_images(images_dir: str) -> List[str]:
    if not os.path.isdir(images_dir):
        return []

    serials = []
    for name in os.listdir(images_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTENSIONS:
            serials.append(stem)
    return sorted(serials)


def _expand_path_candidates(raw_path: Optional[str], session_path: str) -> List[str]:
    if not raw_path:
        return []

    from paradex.utils.path import shared_dir

    expanded = os.path.expanduser(str(raw_path))
    if os.path.isabs(expanded):
        return [os.path.abspath(expanded)]

    candidates = [
        os.path.abspath(os.path.join(session_path, expanded)),
        os.path.abspath(os.path.join(shared_dir, expanded)),
        os.path.abspath(expanded),
    ]
    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _resolve_mesh_root_dir(args: argparse.Namespace) -> str:
    from paradex.utils.path import shared_dir

    mesh_root_dir = args.mesh_root_dir or os.path.join(shared_dir, "mesh_blender")
    return os.path.abspath(os.path.expanduser(str(mesh_root_dir)))


def _resolve_mesh_path(args: argparse.Namespace, session: dict, session_path: str) -> Tuple[Optional[str], List[str]]:
    object_name = args.object_name or session.get("object_name")
    mesh_root_dir = _resolve_mesh_root_dir(args)

    candidates: List[str] = []
    candidates.extend(_expand_path_candidates(args.mesh_path, session_path))

    if object_name:
        candidates.append(os.path.join(mesh_root_dir, object_name, f"{object_name}.obj"))

    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)

    for candidate in deduped:
        if os.path.exists(candidate):
            return candidate, deduped
    return (deduped[0] if deduped else None), deduped


def _resolve_session_path(args: argparse.Namespace) -> str:
    from paradex.utils.file_io import find_latest_directory
    from paradex.utils.path import shared_dir

    if args.session_path:
        raw_path = os.path.expanduser(args.session_path)
        if os.path.isabs(raw_path):
            return os.path.abspath(raw_path)
        return os.path.abspath(os.path.join(shared_dir, raw_path))

    if not args.object_name:
        raise ValueError("Either --session-path or --object-name is required.")

    object_root = os.path.join(shared_dir, args.capture_root, args.object_name)
    session_name = args.session_name
    if session_name is None:
        session_name = find_latest_directory(object_root)
        if session_name is None:
            raise FileNotFoundError(f"No sessions found under {object_root}")

    return os.path.join(object_root, session_name)


def _load_session(session_path: str) -> Tuple[dict, List[dict]]:
    session_json = os.path.join(session_path, "session.json")
    if os.path.exists(session_json):
        session = _read_json(session_json)
    else:
        session = {
            "session_path": session_path,
            "object_name": os.path.basename(os.path.dirname(session_path)),
            "session_name": os.path.basename(session_path),
            "states": [],
        }

    states = []
    for entry in session.get("states", []):
        state_id = entry.get("state_id")
        if not state_id:
            continue
        state_path = entry.get("path") or os.path.join(session_path, state_id)
        if not os.path.isdir(state_path):
            state_path = os.path.join(session_path, state_id)
        states.append(
            {
                "state_id": state_id,
                "state_label": entry.get("state_label", state_id),
                "state_kind": entry.get("state_kind", "unspecified"),
                "path": state_path,
            }
        )

    if not states:
        for name in sorted(os.listdir(session_path)):
            state_path = os.path.join(session_path, name)
            if not os.path.isdir(state_path):
                continue
            if not os.path.isdir(os.path.join(state_path, "raw", "images")):
                continue
            metadata_path = os.path.join(state_path, "metadata.json")
            metadata = _read_json(metadata_path) if os.path.exists(metadata_path) else {}
            states.append(
                {
                    "state_id": name,
                    "state_label": metadata.get("state_label", name),
                    "state_kind": metadata.get("state_kind", "unspecified"),
                    "path": state_path,
                }
            )

    return session, states


def _load_intrinsics(state_path: str) -> Dict[str, dict]:
    intrinsics_path = os.path.join(state_path, "cam_param", "intrinsics.json")
    with open(intrinsics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _undistort_raw_images(state_path: str, force: bool) -> List[str]:
    import cv2
    import numpy as np

    raw_images_dir = os.path.join(state_path, "raw", "images")
    out_dir = os.path.join(state_path, "images")
    os.makedirs(out_dir, exist_ok=True)

    intrinsics = _load_intrinsics(state_path)
    written = []
    for name in sorted(os.listdir(raw_images_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTENSIONS:
            continue

        out_path = os.path.join(out_dir, f"{stem}.png")
        if os.path.exists(out_path) and not force:
            written.append(stem)
            continue

        if stem not in intrinsics:
            print(f"[WARN] {state_path}: no intrinsics for camera {stem}, skipping undistort.")
            continue

        raw_path = os.path.join(raw_images_dir, name)
        image = _cv2_imread(raw_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] {state_path}: failed to read {raw_path}")
            continue

        intr = intrinsics[stem]
        original_k = np.asarray(intr["original_intrinsics"], dtype=np.float64).reshape(3, 3)
        new_k = np.asarray(intr["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(intr["dist_params"], dtype=np.float64).reshape(-1)
        undistorted = cv2.undistort(image, original_k, dist, None, new_k)
        _cv2_imwrite(out_path, undistorted)
        written.append(stem)

    return sorted(set(written))


def _inspect_mesh(mesh_path: Optional[str], candidate_paths: List[str]) -> Dict[str, object]:
    info: Dict[str, object] = {
        "mesh_path": mesh_path,
        "role": "input",
        "candidate_paths": candidate_paths,
        "exists": False,
        "warnings": [],
    }
    if not mesh_path:
        info["warnings"].append("No input mesh path was provided or inferred.")
        return info

    mesh_path = os.path.expanduser(str(mesh_path))
    info["mesh_path"] = mesh_path
    info["exists"] = os.path.exists(mesh_path)

    if not os.path.exists(mesh_path):
        info["warnings"].append("input mesh path does not exist.")
        return info

    if os.path.splitext(mesh_path)[1].lower() != ".obj":
        return info

    obj_dir = os.path.dirname(mesh_path)
    missing_sidecars = []
    with open(mesh_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("mtllib "):
                continue
            for mtl_name in stripped.split()[1:]:
                mtl_path = os.path.join(obj_dir, mtl_name)
                if not os.path.exists(mtl_path):
                    missing_sidecars.append(mtl_path)

    if missing_sidecars:
        info["warnings"].append(f"Missing OBJ material files: {missing_sidecars}")
    info["missing_sidecars"] = missing_sidecars

    try:
        import trimesh

        geom = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(geom, trimesh.Scene):
            geoms = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
            geom = trimesh.util.concatenate(geoms) if geoms else None
        if isinstance(geom, trimesh.Trimesh):
            info["mesh_stats"] = {
                "vertices": int(len(geom.vertices)),
                "faces": int(len(geom.faces)),
                "bounds": geom.bounds.tolist() if geom.bounds is not None else None,
                "extents": geom.extents.tolist() if geom.extents is not None else None,
            }
    except Exception as exc:
            info["warnings"].append(f"Could not inspect mesh geometry: {exc}")
    return info


def _resolve_part_mesh_paths(args: argparse.Namespace, session_path: str) -> Tuple[List[Optional[str]], List[str]]:
    paths: List[Optional[str]] = []
    warnings = []
    for idx, raw_path in enumerate(args.part_mesh_paths or []):
        resolved = _resolve_optional_input_path(raw_path, session_path)
        paths.append(resolved)
        if not resolved or not os.path.exists(resolved):
            warnings.append(f"part mesh {idx} path does not exist: {resolved}")
    return paths, warnings


def _export_composite_mesh_from_parts(
    part_mesh_paths: List[str],
    output_dir: str,
) -> dict:
    import trimesh

    if not part_mesh_paths:
        raise ValueError("no part mesh paths were provided.")

    composite_dir = os.path.join(output_dir, "multi_mesh")
    os.makedirs(composite_dir, exist_ok=True)
    geometries = []
    sources = []
    for mesh_path in part_mesh_paths:
        mesh = _load_mesh_geometry(mesh_path)
        geometries.append(mesh)
        sources.append(
            {
                "path": mesh_path,
                "vertex_count": int(len(mesh.vertices)),
                "face_count": int(len(mesh.faces)),
            }
        )

    composite = trimesh.util.concatenate(geometries)
    composite_path = os.path.join(composite_dir, "composite_input_mesh.obj")
    composite.export(composite_path)
    return {
        "path": composite_path,
        "source_part_meshes": sources,
        "vertex_count": int(len(composite.vertices)),
        "face_count": int(len(composite.faces)),
    }


def _prepare_mesh_info_from_provided_parts(
    args: argparse.Namespace,
    session_path: str,
    output_dir: str,
    mesh_info: dict,
) -> dict:
    part_mesh_paths, path_warnings = _resolve_part_mesh_paths(args, session_path)
    existing_part_mesh_paths = [path for path in part_mesh_paths if path and os.path.exists(path)]
    if not part_mesh_paths:
        return mesh_info

    mesh_info["provided_part_mesh_paths"] = part_mesh_paths
    mesh_info["provided_part_mesh_warnings"] = path_warnings
    mesh_info["warnings"].extend(path_warnings)
    if mesh_info.get("exists"):
        return mesh_info

    if not existing_part_mesh_paths:
        return mesh_info

    try:
        composite = _export_composite_mesh_from_parts(existing_part_mesh_paths, output_dir)
        composite_info = _inspect_mesh(composite["path"], [composite["path"]])
        composite_info["role"] = "composite_provided_part_meshes"
        composite_info["generated_from_part_meshes"] = True
        composite_info["source_part_mesh_paths"] = existing_part_mesh_paths
        composite_info["composite"] = composite
        composite_info["original_input_mesh"] = mesh_info
        composite_info["warnings"].extend(path_warnings)
        composite_info["warnings"].append(
            "No whole input mesh was available; generated a composite input mesh from provided part meshes."
        )
        return composite_info
    except Exception as exc:
        mesh_info["warnings"].append(f"failed to generate composite input mesh from provided parts: {exc}")
        return mesh_info


def _load_optional_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    return _read_json(path)


def _find_image_path(images_dir: str, serial: str) -> Optional[str]:
    for ext in sorted(IMAGE_EXTENSIONS):
        path = os.path.join(images_dir, f"{serial}{ext}")
        if os.path.exists(path):
            return path
    return None


def _state_camera_records(
    state_path: str,
    raw_cameras: List[str],
    undistorted_cameras: List[str],
) -> Dict[str, dict]:
    intrinsics = _load_optional_json(os.path.join(state_path, "cam_param", "intrinsics.json"))
    extrinsics = _load_optional_json(os.path.join(state_path, "cam_param", "extrinsics.json"))
    serials = sorted(set(raw_cameras) | set(undistorted_cameras) | set(intrinsics.keys()) | set(extrinsics.keys()))

    records = {}
    raw_images_dir = os.path.join(state_path, "raw", "images")
    undistorted_images_dir = os.path.join(state_path, "images")
    for serial in serials:
        raw_path = _find_image_path(raw_images_dir, serial)
        undistorted_path = _find_image_path(undistorted_images_dir, serial)
        intr = intrinsics.get(serial, {})
        records[serial] = {
            "raw_image": raw_path,
            "undistorted_image": undistorted_path,
            "has_intrinsics": serial in intrinsics,
            "has_extrinsics": serial in extrinsics,
            "width": intr.get("width"),
            "height": intr.get("height"),
            "intrinsics_undistort": intr.get("intrinsics_undistort"),
        }
    return records


def _process_state(
    state: dict,
    expected_cameras: List[str],
    *,
    undistort: bool,
    force: bool,
) -> dict:
    state_path = state["path"]
    raw_images_dir = os.path.join(state_path, "raw", "images")
    undistorted_images_dir = os.path.join(state_path, "images")
    cam_param_dir = os.path.join(state_path, "cam_param")
    intrinsics_path = os.path.join(cam_param_dir, "intrinsics.json")

    raw_cameras = _list_images(raw_images_dir)
    undistorted_cameras = _list_images(undistorted_images_dir)
    expected_set = set(expected_cameras)
    raw_missing = sorted(expected_set - set(raw_cameras)) if expected_set else []

    warnings = []
    blocking_warnings = []
    if not os.path.isdir(raw_images_dir):
        blocking_warnings.append("raw/images directory missing.")
    if not os.path.isdir(cam_param_dir):
        blocking_warnings.append("cam_param directory missing.")
    elif not os.path.exists(intrinsics_path):
        blocking_warnings.append("cam_param/intrinsics.json missing.")
    if raw_missing:
        warnings.append(f"Missing raw images for cameras: {raw_missing}")
    warnings.extend(blocking_warnings)

    undistort_ran = False
    target_cameras = expected_set if expected_set else set(raw_cameras)
    undistorted_missing_before = sorted(target_cameras - set(undistorted_cameras))
    if undistort and not blocking_warnings:
        if force or undistorted_missing_before:
            _undistort_raw_images(state_path, force=force)
            undistort_ran = True
            undistorted_cameras = _list_images(undistorted_images_dir)

    undistorted_missing = sorted(target_cameras - set(undistorted_cameras))
    if undistort and undistorted_missing:
        warnings.append(f"Missing undistorted images for cameras: {undistorted_missing}")

    return {
        "state_id": state["state_id"],
        "state_label": state.get("state_label", state["state_id"]),
        "state_kind": state.get("state_kind", "unspecified"),
        "path": state_path,
        "raw_images_dir": raw_images_dir,
        "undistorted_images_dir": undistorted_images_dir,
        "cam_param_dir": cam_param_dir,
        "raw_cameras": raw_cameras,
        "undistorted_cameras": undistorted_cameras,
        "missing_raw_cameras": raw_missing,
        "missing_undistorted_cameras": undistorted_missing,
        "undistort_ran": undistort_ran,
        "blocking_warnings": blocking_warnings,
        "warnings": warnings,
        "cameras": _state_camera_records(state_path, raw_cameras, undistorted_cameras),
    }


def _to_3x4(extrinsic_value: Any) -> Optional["np.ndarray"]:
    import numpy as np

    arr = np.asarray(extrinsic_value, dtype=np.float64)
    if arr.shape == (3, 4):
        return arr
    if arr.shape == (4, 4):
        return arr[:3, :]
    if arr.size == 12:
        return arr.reshape(3, 4)
    if arr.size == 16:
        return arr.reshape(4, 4)[:3, :]
    return None


def _camera_center_world(cam_from_world: "np.ndarray") -> "np.ndarray":
    return -cam_from_world[:3, :3].T @ cam_from_world[:3, 3]


def _rotation_matrix_to_colmap_qvec(rotation: "np.ndarray") -> List[float]:
    import numpy as np

    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    qvec = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(qvec)
    if norm > 0:
        qvec /= norm
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec.tolist()


def _load_state_camera_bundle(state_path: str, image_serials: List[str]) -> Tuple[Dict[str, dict], List[str]]:
    import numpy as np

    intrinsics = _load_optional_json(os.path.join(state_path, "cam_param", "intrinsics.json"))
    extrinsics = _load_optional_json(os.path.join(state_path, "cam_param", "extrinsics.json"))
    images_dir = os.path.join(state_path, "images")
    bundle: Dict[str, dict] = {}
    warnings = []

    for serial in sorted(image_serials):
        image_path = _find_image_path(images_dir, serial)
        if image_path is None:
            continue
        if serial not in intrinsics:
            warnings.append(f"{serial}: missing intrinsics.")
            continue
        if serial not in extrinsics:
            warnings.append(f"{serial}: missing extrinsics.")
            continue

        cam_from_world = _to_3x4(extrinsics[serial])
        if cam_from_world is None:
            warnings.append(f"{serial}: unsupported extrinsic shape.")
            continue

        intr = intrinsics[serial]
        K = np.asarray(intr["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
        P = K @ cam_from_world
        bundle[serial] = {
            "serial": serial,
            "image_path": image_path,
            "width": int(intr.get("width", 0)),
            "height": int(intr.get("height", 0)),
            "K": K,
            "cam_from_world": cam_from_world,
            "projection": P,
            "camera_center_world": _camera_center_world(cam_from_world),
        }

    return bundle, warnings


def _jsonable_camera_bundle(bundle: Dict[str, dict]) -> Dict[str, dict]:
    return {
        serial: {
            "image_path": cam["image_path"],
            "width": cam["width"],
            "height": cam["height"],
            "intrinsics_undistort": cam["K"].tolist(),
            "cam_from_world": cam["cam_from_world"].tolist(),
            "camera_center_world": cam["camera_center_world"].tolist(),
        }
        for serial, cam in bundle.items()
    }


def _create_feature_detector(feature_type: str, max_features: int) -> Tuple[Any, int, str]:
    import cv2

    requested = str(feature_type).lower()
    if requested in {"auto", "sift"} and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=int(max_features)), cv2.NORM_L2, "sift"
    if requested == "sift":
        print("[WARN] cv2.SIFT_create is unavailable; falling back to ORB.")
    return cv2.ORB_create(nfeatures=int(max_features)), cv2.NORM_HAMMING, "orb"


def _detect_features_for_bundle(bundle: Dict[str, dict], feature_type: str, max_features: int) -> Tuple[Dict[str, dict], str]:
    import cv2

    detector, norm_type, detector_name = _create_feature_detector(feature_type, max_features)
    features = {}
    for serial, cam in bundle.items():
        image = _cv2_imread(cam["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            features[serial] = {"keypoints": [], "descriptors": None, "image": None, "norm_type": norm_type}
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        features[serial] = {
            "keypoints": keypoints or [],
            "descriptors": descriptors,
            "image": image,
            "norm_type": norm_type,
        }
    return features, detector_name


def _project_points(projection: "np.ndarray", points_world: "np.ndarray") -> "np.ndarray":
    import numpy as np

    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    pixels_h = (projection @ points_h.T).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def _depths_in_camera(cam_from_world: "np.ndarray", points_world: "np.ndarray") -> "np.ndarray":
    import numpy as np

    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    return (cam_from_world @ points_h.T).T[:, 2]


def _triangulation_angles_deg(cam1: dict, cam2: dict, points_world: "np.ndarray") -> "np.ndarray":
    import numpy as np

    ray1 = points_world - cam1["camera_center_world"].reshape(1, 3)
    ray2 = points_world - cam2["camera_center_world"].reshape(1, 3)
    denom = np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1)
    valid = denom > 1.0e-12
    cosv = np.ones(points_world.shape[0], dtype=np.float64)
    cosv[valid] = np.sum(ray1[valid] * ray2[valid], axis=1) / denom[valid]
    cosv = np.clip(cosv, -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


def _sample_colors(image: "np.ndarray", pixels: "np.ndarray") -> "np.ndarray":
    import numpy as np

    h, w = image.shape[:2]
    xy = np.rint(pixels).astype(np.int32)
    xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
    bgr = image[xy[:, 1], xy[:, 0], :]
    return bgr[:, ::-1].astype(np.uint8)


def _match_and_triangulate_pair(
    serial1: str,
    serial2: str,
    bundle: Dict[str, dict],
    features: Dict[str, dict],
    args: argparse.Namespace,
) -> Tuple[Optional[dict], Optional[Tuple["np.ndarray", "np.ndarray"]]]:
    import cv2
    import numpy as np

    feat1 = features[serial1]
    feat2 = features[serial2]
    desc1 = feat1["descriptors"]
    desc2 = feat2["descriptors"]
    pair_report = {
        "serials": [serial1, serial2],
        "keypoints": [len(feat1["keypoints"]), len(feat2["keypoints"])],
        "raw_matches": 0,
        "ratio_matches": 0,
        "ransac_inliers": 0,
        "triangulated_points": 0,
        "warnings": [],
    }

    if desc1 is None or desc2 is None or len(feat1["keypoints"]) == 0 or len(feat2["keypoints"]) == 0:
        pair_report["warnings"].append("missing descriptors.")
        return pair_report, None

    matcher = cv2.BFMatcher(feat1["norm_type"], crossCheck=False)
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    pair_report["raw_matches"] = len(raw_matches)
    good = []
    for match_group in raw_matches:
        if len(match_group) < 2:
            continue
        m, n = match_group
        if m.distance < float(args.match_ratio) * n.distance:
            good.append(m)
    pair_report["ratio_matches"] = len(good)
    if len(good) < int(args.min_pair_matches):
        pair_report["warnings"].append("too few ratio-test matches.")
        return pair_report, None

    pts1 = np.asarray([feat1["keypoints"][m.queryIdx].pt for m in good], dtype=np.float64)
    pts2 = np.asarray([feat2["keypoints"][m.trainIdx].pt for m in good], dtype=np.float64)

    if len(good) >= 8:
        _, mask = cv2.findFundamentalMat(
            pts1,
            pts2,
            cv2.FM_RANSAC,
            float(args.ransac_reproj_thresh),
            0.999,
        )
        if mask is not None:
            inlier_mask = mask.reshape(-1).astype(bool)
            pts1 = pts1[inlier_mask]
            pts2 = pts2[inlier_mask]
    pair_report["ransac_inliers"] = int(pts1.shape[0])
    if pts1.shape[0] < int(args.min_pair_matches):
        pair_report["warnings"].append("too few geometric inliers.")
        return pair_report, None

    cam1 = bundle[serial1]
    cam2 = bundle[serial2]
    points_h = cv2.triangulatePoints(
        cam1["projection"],
        cam2["projection"],
        pts1.T.astype(np.float64),
        pts2.T.astype(np.float64),
    ).T
    valid_h = np.abs(points_h[:, 3]) > 1.0e-12
    points = points_h[valid_h, :3] / points_h[valid_h, 3:4]
    pts1 = pts1[valid_h]
    pts2 = pts2[valid_h]
    if points.shape[0] == 0:
        pair_report["warnings"].append("triangulation produced no finite points.")
        return pair_report, None

    proj1 = _project_points(cam1["projection"], points)
    proj2 = _project_points(cam2["projection"], points)
    err1 = np.linalg.norm(proj1 - pts1, axis=1)
    err2 = np.linalg.norm(proj2 - pts2, axis=1)
    depths1 = _depths_in_camera(cam1["cam_from_world"], points)
    depths2 = _depths_in_camera(cam2["cam_from_world"], points)
    angles = _triangulation_angles_deg(cam1, cam2, points)

    keep = (
        (depths1 > 0.0)
        & (depths2 > 0.0)
        & (err1 <= float(args.triangulation_reproj_thresh))
        & (err2 <= float(args.triangulation_reproj_thresh))
        & (angles >= float(args.min_triangulation_angle_deg))
    )
    if float(args.max_depth) > 0:
        keep &= (depths1 <= float(args.max_depth)) & (depths2 <= float(args.max_depth))

    points = points[keep]
    pts1 = pts1[keep]
    if points.shape[0] == 0:
        pair_report["warnings"].append("all triangulated points were filtered.")
        return pair_report, None

    colors = _sample_colors(feat1["image"], pts1)
    pair_report["triangulated_points"] = int(points.shape[0])
    pair_report["mean_reprojection_error_px"] = float(np.mean(np.maximum(err1[keep], err2[keep])))
    pair_report["mean_triangulation_angle_deg"] = float(np.mean(angles[keep]))
    return pair_report, (points, colors)


def _write_ply(path: str, points: "np.ndarray", colors: "np.ndarray") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{float(point[0]):.8f} {float(point[1]):.8f} {float(point[2]):.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _write_colored_mesh_ply(
    path: str,
    vertices: "np.ndarray",
    faces: "np.ndarray",
    vertex_colors: "np.ndarray",
) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex, color in zip(vertices, vertex_colors):
            f.write(
                f"{float(vertex[0]):.8f} {float(vertex[1]):.8f} {float(vertex[2]):.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
    return {"path": path, "vertex_count": int(vertices.shape[0]), "face_count": int(faces.shape[0])}


def _export_colmap_text_model(multiview_dir: str, bundle: Dict[str, dict]) -> str:
    colmap_dir = os.path.join(multiview_dir, "colmap_text")
    os.makedirs(colmap_dir, exist_ok=True)
    serials = sorted(bundle.keys())
    camera_ids = {serial: idx + 1 for idx, serial in enumerate(serials)}

    with open(os.path.join(colmap_dir, "cameras.txt"), "w", encoding="utf-8") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for serial in serials:
            cam = bundle[serial]
            K = cam["K"]
            f.write(
                f"{camera_ids[serial]} PINHOLE {cam['width']} {cam['height']} "
                f"{K[0, 0]:.12g} {K[1, 1]:.12g} {K[0, 2]:.12g} {K[1, 2]:.12g}\n"
            )

    with open(os.path.join(colmap_dir, "images.txt"), "w", encoding="utf-8") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id, serial in enumerate(serials, start=1):
            cam = bundle[serial]
            qvec = _rotation_matrix_to_colmap_qvec(cam["cam_from_world"][:3, :3])
            t = cam["cam_from_world"][:3, 3]
            image_name = os.path.basename(cam["image_path"])
            f.write(
                f"{image_id} {qvec[0]:.12g} {qvec[1]:.12g} {qvec[2]:.12g} {qvec[3]:.12g} "
                f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} {camera_ids[serial]} {image_name}\n\n"
            )

    with open(os.path.join(colmap_dir, "points3D.txt"), "w", encoding="utf-8") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

    return colmap_dir


def _process_multiview_state(report: dict, output_dir: str, args: argparse.Namespace) -> dict:
    import numpy as np

    state_id = report["state_id"]
    multiview_dir = os.path.join(output_dir, "multiview", state_id)
    os.makedirs(multiview_dir, exist_ok=True)

    summary = {
        "state_id": state_id,
        "status": "skipped",
        "output_dir": multiview_dir,
        "camera_count": 0,
        "detector": None,
        "point_count": 0,
        "pointcloud_path": None,
        "colmap_text_dir": None,
        "pair_reports": [],
        "warnings": [],
    }

    if report["blocking_warnings"] or report["missing_undistorted_cameras"]:
        summary["warnings"].append("state has blocking or missing undistorted-image warnings.")
        _write_json(os.path.join(multiview_dir, "multiview_manifest.json"), summary)
        return summary

    bundle, camera_warnings = _load_state_camera_bundle(report["path"], report["undistorted_cameras"])
    summary["warnings"].extend(camera_warnings)
    summary["camera_count"] = len(bundle)
    _write_json(os.path.join(multiview_dir, "cameras.json"), _jsonable_camera_bundle(bundle))

    if len(bundle) < 2:
        summary["warnings"].append("at least two calibrated camera images are required for triangulation.")
        _write_json(os.path.join(multiview_dir, "multiview_manifest.json"), summary)
        return summary

    if not args.no_colmap_text:
        summary["colmap_text_dir"] = _export_colmap_text_model(multiview_dir, bundle)

    features, detector_name = _detect_features_for_bundle(bundle, args.feature_type, args.max_features)
    summary["detector"] = detector_name

    all_points = []
    all_colors = []
    serials = sorted(bundle.keys())
    for i, serial1 in enumerate(serials):
        for serial2 in serials[i + 1 :]:
            pair_report, pair_points = _match_and_triangulate_pair(serial1, serial2, bundle, features, args)
            if pair_report is not None:
                summary["pair_reports"].append(pair_report)
            if pair_points is None:
                continue
            points, colors = pair_points
            all_points.append(points)
            all_colors.append(colors)

    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        if int(args.max_points_per_state) > 0 and points.shape[0] > int(args.max_points_per_state):
            points = points[: int(args.max_points_per_state)]
            colors = colors[: int(args.max_points_per_state)]
        pointcloud_path = os.path.join(multiview_dir, "sparse_points.ply")
        _write_ply(pointcloud_path, points, colors)
        summary["status"] = "ok"
        summary["point_count"] = int(points.shape[0])
        summary["pointcloud_path"] = pointcloud_path
    else:
        summary["status"] = "no_points"
        summary["warnings"].append("no sparse points survived matching and triangulation.")

    summary["pair_diagnostics"] = _write_multiview_pair_diagnostics(multiview_dir, serials, summary["pair_reports"])
    _write_json(os.path.join(multiview_dir, "pairs.json"), {"pairs": summary["pair_reports"]})
    _write_json(os.path.join(multiview_dir, "multiview_manifest.json"), summary)
    return summary


def _process_multiview_states(state_reports: List[dict], output_dir: str, args: argparse.Namespace) -> List[dict]:
    if args.check_only or args.skip_multiview:
        return []

    summaries = []
    for report in state_reports:
        summary = _process_multiview_state(report, output_dir, args)
        report["multiview"] = summary
        summaries.append(summary)
        print(
            f"[MV] {report['state_id']} {report['state_label']}: "
            f"status={summary['status']}, cameras={summary['camera_count']}, points={summary['point_count']}"
        )
    _write_multiview_session_diagnostics(output_dir, summaries)
    return summaries


def _read_ascii_ply_points(path: str) -> Tuple["np.ndarray", Optional["np.ndarray"]]:
    import numpy as np

    with open(path, "r", encoding="ascii", errors="ignore") as f:
        header = []
        vertex_count = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"invalid PLY header: {path}")
            stripped = line.strip()
            header.append(stripped)
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            if stripped == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"PLY vertex count missing: {path}")

        rows = []
        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])] + [float(x) for x in parts[3:6]])

    if not rows:
        return np.zeros((0, 3), dtype=np.float64), None

    arr = np.asarray(rows, dtype=np.float64)
    points = arr[:, :3]
    colors = None
    if arr.shape[1] >= 6:
        colors = np.clip(arr[:, 3:6], 0, 255).astype(np.uint8)
    return points, colors


def _linspace_indices(length: int, count: int) -> List[int]:
    length = int(length)
    count = int(count)
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    return [int(round(i * (length - 1) / float(count - 1))) for i in range(count)]


def _read_ascii_ply_preview(
    path: str,
    output_dir: str,
    title: str,
    max_vertices: int = 18000,
    max_faces: int = 9000,
) -> Optional[dict]:
    """Load a bounded ASCII PLY subset for the self-contained HTML debug viewer."""
    if not path or not os.path.exists(path):
        return None

    with open(path, "r", encoding="ascii", errors="ignore") as f:
        vertex_count = None
        face_count = 0
        vertex_properties = []
        section = None
        while True:
            line = f.readline()
            if not line:
                return None
            stripped = line.strip()
            if stripped.startswith("format ") and "ascii" not in stripped:
                return None
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
                section = "vertex"
                continue
            if stripped.startswith("element face "):
                face_count = int(stripped.split()[-1])
                section = "face"
                continue
            if stripped.startswith("element "):
                section = None
                continue
            if stripped.startswith("property ") and section == "vertex":
                parts = stripped.split()
                if len(parts) >= 3:
                    vertex_properties.append(parts[-1])
                continue
            if stripped == "end_header":
                break

        if vertex_count is None:
            return None

        def prop_index(name: str) -> Optional[int]:
            try:
                return vertex_properties.index(name)
            except ValueError:
                return None

        x_idx = prop_index("x")
        y_idx = prop_index("y")
        z_idx = prop_index("z")
        if x_idx is None or y_idx is None or z_idx is None:
            return None
        r_idx = prop_index("red")
        g_idx = prop_index("green")
        b_idx = prop_index("blue")

        vertices = []
        colors = []
        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) <= max(x_idx, y_idx, z_idx):
                continue
            vertices.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
            if r_idx is not None and g_idx is not None and b_idx is not None and len(parts) > max(r_idx, g_idx, b_idx):
                colors.append([int(float(parts[r_idx])), int(float(parts[g_idx])), int(float(parts[b_idx]))])
            else:
                colors.append([180, 180, 180])

        faces = []
        for _ in range(face_count):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if not parts:
                continue
            count = int(parts[0])
            if count >= 3 and len(parts) >= count + 1:
                indices = [int(parts[i + 1]) for i in range(count)]
                for j in range(1, count - 1):
                    faces.append([indices[0], indices[j], indices[j + 1]])

    if not vertices:
        return None

    if faces:
        if len(faces) > max_faces:
            step_indices = _linspace_indices(len(faces), max_faces)
            faces = [faces[i] for i in step_indices]
        used = sorted({idx for face in faces for idx in face if 0 <= idx < len(vertices)})
        if len(used) > max_vertices:
            while faces and len(used) > max_vertices:
                target = max(1, int(len(faces) * 0.65))
                step_indices = _linspace_indices(len(faces), target)
                faces = [faces[i] for i in step_indices]
                used = sorted({idx for face in faces for idx in face if 0 <= idx < len(vertices)})
        if len(used) <= max_vertices:
            remap = {old: new for new, old in enumerate(used)}
            vertices_out = [vertices[i] for i in used]
            colors_out = [colors[i] for i in used]
            faces_out = [[remap[face[0]], remap[face[1]], remap[face[2]]] for face in faces]
        else:
            keep = _linspace_indices(len(vertices), max_vertices)
            vertices_out = [vertices[i] for i in keep]
            colors_out = [colors[i] for i in keep]
            faces_out = []
    else:
        if len(vertices) > max_vertices:
            keep = _linspace_indices(len(vertices), max_vertices)
            vertices_out = [vertices[i] for i in keep]
            colors_out = [colors[i] for i in keep]
        else:
            vertices_out = vertices
            colors_out = colors
        faces_out = []

    rel = _html_relpath(path, output_dir)
    return {
        "title": title,
        "path": rel or path,
        "vertexCount": int(vertex_count),
        "faceCount": int(face_count),
        "shownVertexCount": int(len(vertices_out)),
        "shownFaceCount": int(len(faces_out)),
        "vertices": vertices_out,
        "colors": colors_out,
        "faces": faces_out,
    }


def _load_mesh_sample_points(mesh_path: str, sample_count: int) -> "np.ndarray":
    import numpy as np
    import trimesh

    geom = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(geom, trimesh.Scene):
        geoms = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"no mesh geometry in scene: {mesh_path}")
        geom = trimesh.util.concatenate(geoms)
    if not isinstance(geom, trimesh.Trimesh):
        raise ValueError(f"unsupported mesh type: {type(geom)}")

    sample_count = int(sample_count)
    if sample_count <= 0:
        return np.asarray(geom.vertices, dtype=np.float64)
    try:
        return np.asarray(geom.sample(sample_count), dtype=np.float64)
    except Exception:
        vertices = np.asarray(geom.vertices, dtype=np.float64)
        if vertices.shape[0] <= sample_count:
            return vertices
        idx = np.linspace(0, vertices.shape[0] - 1, sample_count).astype(np.int64)
        return vertices[idx]


def _subsample_points(points: "np.ndarray", max_points: int) -> "np.ndarray":
    import numpy as np

    max_points = int(max_points)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
    return points[idx]


def _subsample_points_and_colors(
    points: "np.ndarray",
    colors: Optional["np.ndarray"],
    max_points: int,
) -> Tuple["np.ndarray", Optional["np.ndarray"]]:
    import numpy as np

    max_points = int(max_points)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
    sampled_colors = colors[idx] if colors is not None and colors.shape[0] == points.shape[0] else None
    return points[idx], sampled_colors


def _pca_axes(points: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    import numpy as np

    centroid = np.mean(points, axis=0)
    centered = points - centroid.reshape(1, 3)
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    extents = np.ptp(centered @ axes, axis=0)
    return centroid, axes, extents


def _estimate_pca_registration(
    mesh_points: "np.ndarray",
    target_points: "np.ndarray",
    scale_mode: str,
) -> Tuple["np.ndarray", dict]:
    import numpy as np

    mesh_centroid, mesh_axes, mesh_extents = _pca_axes(mesh_points)
    target_centroid, target_axes, target_extents = _pca_axes(target_points)
    R = target_axes @ mesh_axes.T
    if np.linalg.det(R) < 0:
        target_axes[:, -1] *= -1.0
        R = target_axes @ mesh_axes.T

    scale = 1.0
    if scale_mode == "uniform":
        valid = mesh_extents > 1.0e-9
        if np.any(valid):
            ratios = target_extents[valid] / mesh_extents[valid]
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
            if ratios.size > 0:
                scale = float(np.median(ratios))
    elif scale_mode != "none":
        raise ValueError(f"unsupported registration scale mode: {scale_mode}")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = scale * R
    T[:3, 3] = target_centroid - scale * R @ mesh_centroid
    diagnostics = {
        "mesh_centroid": mesh_centroid.tolist(),
        "target_centroid": target_centroid.tolist(),
        "mesh_extents_pca": mesh_extents.tolist(),
        "target_extents_pca": target_extents.tolist(),
        "scale": scale,
    }
    return T, diagnostics


def _apply_transform(points: "np.ndarray", transform: "np.ndarray") -> "np.ndarray":
    import numpy as np

    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    out = (transform @ points_h.T).T
    return out[:, :3]


def _nearest_distances(query: "np.ndarray", target: "np.ndarray", chunk_size: int = 2048) -> "np.ndarray":
    import numpy as np

    distances = []
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start : start + chunk_size]
        diff = chunk[:, None, :] - target[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        distances.append(np.sqrt(np.min(d2, axis=1)))
    return np.concatenate(distances, axis=0) if distances else np.zeros((0,), dtype=np.float64)


def _nearest_neighbors(
    query: "np.ndarray",
    target: "np.ndarray",
    chunk_size: int = 1024,
) -> Tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    all_indices = []
    all_distances = []
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start : start + chunk_size]
        diff = chunk[:, None, :] - target[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        indices = np.argmin(d2, axis=1)
        distances = np.sqrt(d2[np.arange(chunk.shape[0]), indices])
        all_indices.append(indices.astype(np.int64))
        all_distances.append(distances)
    if not all_indices:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    return np.concatenate(all_indices, axis=0), np.concatenate(all_distances, axis=0)


def _registration_quality(
    transformed_mesh_points: "np.ndarray",
    target_points: "np.ndarray",
    max_eval_points: int,
) -> dict:
    import numpy as np

    query = _subsample_points(transformed_mesh_points, max_eval_points)
    target = _subsample_points(target_points, max_eval_points)
    if query.shape[0] == 0 or target.shape[0] == 0:
        return {"evaluated_points": 0}
    distances = _nearest_distances(query, target)
    return {
        "evaluated_points": int(distances.shape[0]),
        "mean_nn_distance": float(np.mean(distances)),
        "median_nn_distance": float(np.median(distances)),
        "p90_nn_distance": float(np.percentile(distances, 90.0)),
    }


def _estimate_rigid_transform(source_points: "np.ndarray", target_points: "np.ndarray") -> "np.ndarray":
    import numpy as np

    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    source_centered = source_points - source_centroid.reshape(1, 3)
    target_centered = target_points - target_centroid.reshape(1, 3)
    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = target_centroid - R @ source_centroid
    return T


def _rotation_angle_deg(rotation: "np.ndarray") -> float:
    import numpy as np

    trace = float(np.trace(rotation))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _refine_registration_trimmed_icp(
    mesh_points: "np.ndarray",
    target_points: "np.ndarray",
    initial_transform: "np.ndarray",
    args: argparse.Namespace,
) -> Tuple["np.ndarray", dict]:
    import numpy as np

    source = _subsample_points(mesh_points, int(args.icp_max_source_points))
    target = _subsample_points(target_points, int(args.icp_max_target_points))
    min_correspondences = int(args.icp_min_correspondences)
    if source.shape[0] < min_correspondences:
        raise ValueError(f"mesh sample has fewer than {min_correspondences} ICP source points.")
    if target.shape[0] < min_correspondences:
        raise ValueError(f"target pointcloud has fewer than {min_correspondences} ICP target points.")

    trim_fraction = float(args.icp_trim_fraction)
    if not 0.0 < trim_fraction <= 1.0:
        raise ValueError("--icp-trim-fraction must be in (0, 1].")

    T = np.asarray(initial_transform, dtype=np.float64).reshape(4, 4).copy()
    max_distance = float(args.icp_max_correspondence_distance)
    tolerance = float(args.icp_tolerance)
    history = []
    previous_error = None
    converged = False

    for iteration in range(int(args.icp_iterations)):
        transformed = _apply_transform(source, T)
        nn_indices, distances = _nearest_neighbors(transformed, target)
        candidate_mask = np.isfinite(distances)
        if max_distance > 0.0:
            candidate_mask &= distances <= max_distance

        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.shape[0] < min_correspondences:
            raise ValueError(
                f"ICP iteration {iteration}: only {candidate_indices.shape[0]} valid correspondences."
            )

        ordered = candidate_indices[np.argsort(distances[candidate_indices])]
        keep_count = int(np.ceil(ordered.shape[0] * trim_fraction))
        keep_count = min(ordered.shape[0], max(min_correspondences, keep_count))
        kept = ordered[:keep_count]

        source_corr = transformed[kept]
        target_corr = target[nn_indices[kept]]
        delta = _estimate_rigid_transform(source_corr, target_corr)
        T = delta @ T

        kept_distances = distances[kept]
        mean_error = float(np.mean(kept_distances))
        history.append(
            {
                "iteration": iteration,
                "correspondences": int(kept.shape[0]),
                "mean_distance": mean_error,
                "median_distance": float(np.median(kept_distances)),
                "p90_distance": float(np.percentile(kept_distances, 90.0)),
                "delta_translation_norm": float(np.linalg.norm(delta[:3, 3])),
                "delta_rotation_deg": _rotation_angle_deg(delta[:3, :3]),
            }
        )

        if previous_error is not None and abs(previous_error - mean_error) <= tolerance:
            converged = True
            break
        previous_error = mean_error

    if not history:
        raise ValueError("ICP did not run any iterations.")

    diagnostics = {
        "method": "trimmed_icp",
        "status": "converged" if converged else "max_iterations",
        "iterations": len(history),
        "source_points": int(source.shape[0]),
        "target_points": int(target.shape[0]),
        "trim_fraction": trim_fraction,
        "max_correspondence_distance": max_distance if max_distance > 0.0 else None,
        "min_correspondences": min_correspondences,
        "history": history,
    }
    return T, diagnostics


def _pca_registration_candidates(
    mesh_points: "np.ndarray",
    target_points: "np.ndarray",
    scale_mode: str,
) -> Tuple[List["np.ndarray"], dict]:
    """PCA-init registration candidates covering principal-axis sign ambiguity.

    A flat, nearly symmetric object (e.g. a picture frame) has an ambiguous PCA
    orientation: the in-plane axes can swap and each axis sign can flip. A single
    PCA alignment therefore often locks onto a mirrored/rotated pose. This returns
    the (up to four) proper-rotation candidates so the caller can pick the one that
    best explains the observed object points after refinement.
    """
    import numpy as np

    mesh_centroid, mesh_axes, mesh_extents = _pca_axes(mesh_points)
    target_centroid, target_axes, target_extents = _pca_axes(target_points)

    scale = 1.0
    if scale_mode == "uniform":
        valid = mesh_extents > 1.0e-9
        if np.any(valid):
            ratios = target_extents[valid] / mesh_extents[valid]
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
            if ratios.size > 0:
                scale = float(np.median(ratios))
    elif scale_mode != "none":
        raise ValueError(f"unsupported registration scale mode: {scale_mode}")

    candidates: List["np.ndarray"] = []
    for signs in ((1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)):
        flipped_axes = target_axes * np.asarray(signs, dtype=np.float64).reshape(1, 3)
        R = flipped_axes @ mesh_axes.T
        if np.linalg.det(R) <= 0.0:
            continue
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = scale * R
        T[:3, 3] = target_centroid - scale * R @ mesh_centroid
        candidates.append(T)

    diagnostics = {
        "mesh_centroid": mesh_centroid.tolist(),
        "target_centroid": target_centroid.tolist(),
        "mesh_extents_pca": mesh_extents.tolist(),
        "target_extents_pca": target_extents.tolist(),
        "scale": scale,
        "candidate_count": int(len(candidates)),
    }
    return candidates, diagnostics


def _object_roi_register(
    mesh_points: "np.ndarray",
    target_points: "np.ndarray",
    mesh_diag: float,
    args: argparse.Namespace,
) -> dict:
    """Robust whole-object registration against a scene-contaminated point cloud.

    Multiview triangulation reconstructs the whole scene, so the target cloud mixes
    object points with background/outliers. PCA + ICP on the full cloud is dragged
    off the object. This routine, for each orientation candidate, iteratively (a)
    crops the target to points near the currently aligned mesh surface and (b) runs
    trimmed ICP on that object-only crop, shrinking the crop radius each iteration.
    The candidate that ends with the most tight object inliers is selected.
    """
    import numpy as np

    crop_distance = float(getattr(args, "registration_object_crop_distance", 0.15))
    iterations = max(1, int(getattr(args, "registration_crop_iterations", 3)))
    tight_fraction = max(1.0e-6, float(getattr(args, "registration_object_tight_fraction", 0.05)))
    min_correspondences = int(args.icp_min_correspondences)
    scale = max(float(mesh_diag), 1.0e-9)
    tight_distance = tight_fraction * scale

    candidates, coarse_diag = _pca_registration_candidates(
        mesh_points, target_points, args.registration_scale_mode
    )
    if not candidates:
        raise ValueError("no valid PCA registration orientation candidates.")

    eval_source = _subsample_points(mesh_points, int(args.icp_max_source_points))
    work_target = _subsample_points(target_points, max(int(args.icp_max_target_points) * 2, 20000))

    candidate_transforms: List["np.ndarray"] = []
    candidate_reports: List[dict] = []
    best_index = -1
    best_score = None

    for candidate_index, coarse_T in enumerate(candidates):
        T = np.asarray(coarse_T, dtype=np.float64).copy()
        history: List[dict] = []
        ok = True
        for iteration in range(iterations):
            frac = 1.0 - 0.5 * (iteration / max(iterations - 1, 1))
            crop_k = crop_distance * frac
            transformed = _apply_transform(eval_source, T)
            distances = _nearest_distances(work_target, transformed)
            cropped = work_target[distances <= crop_k * scale]
            if cropped.shape[0] < min_correspondences:
                ok = False
                history.append(
                    {"iteration": iteration, "status": "too_few_cropped_points", "cropped_target_points": int(cropped.shape[0])}
                )
                break
            try:
                T, _ = _refine_registration_trimmed_icp(mesh_points, cropped, T, args)
            except Exception as exc:
                ok = False
                history.append({"iteration": iteration, "status": "icp_failed", "reason": str(exc)})
                break
            history.append(
                {
                    "iteration": iteration,
                    "crop_distance_fraction": float(crop_k),
                    "cropped_target_points": int(cropped.shape[0]),
                }
            )

        transformed = _apply_transform(eval_source, T)
        surf = _nearest_distances(work_target, transformed)
        tight_mask = surf <= tight_distance
        tight_inliers = int(np.count_nonzero(tight_mask))
        median_surfdist = float(np.median(surf[tight_mask]) / scale) if tight_inliers > 0 else None
        candidate_transforms.append(T)
        candidate_reports.append(
            {
                "candidate_index": int(candidate_index),
                "ok": bool(ok),
                "tight_inliers": tight_inliers,
                "median_object_surfdist": median_surfdist,
                "history": history,
            }
        )
        score = (
            1 if ok else 0,
            tight_inliers,
            -(median_surfdist if median_surfdist is not None else 1.0e9),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_index = candidate_index

    if best_index < 0:
        raise ValueError("object-ROI registration failed for all orientation candidates.")

    best_T = candidate_transforms[best_index]
    transformed_full = _apply_transform(eval_source, best_T)
    distances_full = _nearest_distances(target_points, transformed_full)
    object_points = target_points[distances_full <= crop_distance * scale]
    tight_full = distances_full[distances_full <= tight_distance]
    object_quality = {
        "object_point_count": int(object_points.shape[0]),
        "tight_inlier_count": int(tight_full.shape[0]),
        "tight_inlier_distance_fraction": float(tight_fraction),
        "median_object_surfdist_fraction": float(np.median(tight_full) / scale) if tight_full.shape[0] else None,
        "p90_object_surfdist_fraction": float(np.percentile(tight_full, 90.0) / scale) if tight_full.shape[0] else None,
        "mesh_diag": float(scale),
    }
    return {
        "T_world_object": best_T,
        "coarse_candidates": coarse_diag,
        "selected_candidate_index": int(best_index),
        "candidate_reports": candidate_reports,
        "object_points": object_points,
        "object_quality": object_quality,
    }


def _write_points_ply(path: str, points: "np.ndarray", color: Tuple[int, int, int]) -> None:
    import numpy as np

    colors = np.tile(np.asarray(color, dtype=np.uint8).reshape(1, 3), (points.shape[0], 1))
    _write_ply(path, points, colors)


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = ";".join(str(item) for item in value)
    text = str(value).replace("\r", " ").replace("\n", " ")
    if any(char in text for char in [",", "\"", "\n"]):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text


def _write_csv_rows(path: str, columns: List[str], rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(_csv_cell(row.get(column)) for column in columns) + "\n")


def _write_colored_point_sets_ply(path: str, point_sets: List[Tuple["np.ndarray", Tuple[int, int, int]]]) -> dict:
    import numpy as np

    valid_sets = []
    total_points = 0
    for points, color in point_sets:
        points = np.asarray(points, dtype=np.float64)
        if points.shape[0] == 0:
            continue
        colors = np.tile(np.asarray(color, dtype=np.uint8).reshape(1, 3), (points.shape[0], 1))
        valid_sets.append((points, colors))
        total_points += int(points.shape[0])

    if total_points == 0:
        return {"path": None, "point_count": 0}

    all_points = np.concatenate([item[0] for item in valid_sets], axis=0)
    all_colors = np.concatenate([item[1] for item in valid_sets], axis=0)
    _write_ply(path, all_points, all_colors)
    return {"path": path, "point_count": int(all_points.shape[0])}


def _safe_filename_token(value: Any) -> str:
    text = str(value) if value is not None else "unknown"
    out = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            out.append(char)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or "unknown"


def _pair_warning_counts(pair_reports: List[dict]) -> dict:
    counts: Dict[str, int] = {}
    for report in pair_reports:
        for warning in report.get("warnings", []):
            key = str(warning)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _summarize_pair_reports(pair_reports: List[dict]) -> dict:
    import numpy as np

    pair_count = len(pair_reports)
    triangulated = np.asarray([int(report.get("triangulated_points", 0)) for report in pair_reports], dtype=np.int64)
    ratio_matches = np.asarray([int(report.get("ratio_matches", 0)) for report in pair_reports], dtype=np.int64)
    inliers = np.asarray([int(report.get("ransac_inliers", 0)) for report in pair_reports], dtype=np.int64)
    nonzero = triangulated[triangulated > 0]
    return {
        "pair_count": int(pair_count),
        "triangulated_pair_count": int(nonzero.shape[0]),
        "triangulated_pair_fraction": float(nonzero.shape[0] / max(pair_count, 1)),
        "triangulated_points_total": int(np.sum(triangulated)) if triangulated.shape[0] else 0,
        "triangulated_points_median_all_pairs": float(np.median(triangulated)) if triangulated.shape[0] else None,
        "triangulated_points_median_nonzero_pairs": float(np.median(nonzero)) if nonzero.shape[0] else None,
        "ratio_matches_median": float(np.median(ratio_matches)) if ratio_matches.shape[0] else None,
        "ransac_inliers_median": float(np.median(inliers)) if inliers.shape[0] else None,
        "warning_counts": _pair_warning_counts(pair_reports),
    }


def _write_heatmap_png(
    path: str,
    labels: List[str],
    matrix: "np.ndarray",
    title: str,
) -> Optional[str]:
    import cv2
    import numpy as np

    if not labels:
        return None
    labels_short = [str(label)[-5:] for label in labels]
    n = len(labels)
    cell = 30
    left = 84
    top = 58
    right = 18
    bottom = 54
    width = left + n * cell + right
    height = top + n * cell + bottom
    image = np.full((height, width, 3), 245, dtype=np.uint8)

    values = np.asarray(matrix, dtype=np.float64)
    finite = values[np.isfinite(values) & (values > 0)]
    vmax = float(np.max(finite)) if finite.size else 1.0

    for row in range(n):
        for col in range(n):
            x0 = left + col * cell
            y0 = top + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            if row == col:
                color = (225, 225, 225)
            else:
                value = float(values[row, col]) if np.isfinite(values[row, col]) else 0.0
                ratio = min(1.0, np.log1p(max(0.0, value)) / np.log1p(vmax))
                color = (
                    int(245 - 180 * ratio),
                    int(238 - 125 * ratio),
                    int(230 + 25 * ratio),
                )
            cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=-1)
            cv2.rectangle(image, (x0, y0), (x1, y1), (180, 180, 180), thickness=1)
            if row != col and values[row, col] > 0:
                text = str(int(values[row, col]))
                size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
                cv2.putText(
                    image,
                    text,
                    (x0 + max(2, (cell - size[0]) // 2), y0 + (cell + size[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (25, 25, 25),
                    1,
                    cv2.LINE_AA,
                )

    cv2.putText(image, title, (left, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    for idx, label in enumerate(labels_short):
        x = left + idx * cell + 2
        y = top - 10
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (10, top + idx * cell + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        image,
        f"max={int(vmax)}; darker cells are stronger pairs",
        (left, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if cv2.imwrite(path, image):
        return path
    return None


def _write_multiview_pair_diagnostics(
    multiview_dir: str,
    serials: List[str],
    pair_reports: List[dict],
) -> dict:
    import numpy as np

    pair_rows = []
    heatmap: Dict[Tuple[str, str], int] = {}
    ransac_heatmap: Dict[Tuple[str, str], int] = {}
    for report in pair_reports:
        pair_serials = report.get("serials", [None, None])
        serial1 = pair_serials[0] if len(pair_serials) > 0 else None
        serial2 = pair_serials[1] if len(pair_serials) > 1 else None
        keypoints = report.get("keypoints", [None, None])
        triangulated = int(report.get("triangulated_points", 0))
        ransac_inliers = int(report.get("ransac_inliers", 0))
        if serial1 is not None and serial2 is not None:
            heatmap[(serial1, serial2)] = triangulated
            heatmap[(serial2, serial1)] = triangulated
            ransac_heatmap[(serial1, serial2)] = ransac_inliers
            ransac_heatmap[(serial2, serial1)] = ransac_inliers
        pair_rows.append(
            {
                "serial1": serial1,
                "serial2": serial2,
                "keypoints1": keypoints[0] if len(keypoints) > 0 else None,
                "keypoints2": keypoints[1] if len(keypoints) > 1 else None,
                "raw_matches": report.get("raw_matches"),
                "ratio_matches": report.get("ratio_matches"),
                "ransac_inliers": report.get("ransac_inliers"),
                "triangulated_points": triangulated,
                "mean_reprojection_error_px": report.get("mean_reprojection_error_px"),
                "mean_triangulation_angle_deg": report.get("mean_triangulation_angle_deg"),
                "warnings": report.get("warnings", []),
            }
        )

    pairs_csv = os.path.join(multiview_dir, "pairs.csv")
    _write_csv_rows(
        pairs_csv,
        [
            "serial1",
            "serial2",
            "keypoints1",
            "keypoints2",
            "raw_matches",
            "ratio_matches",
            "ransac_inliers",
            "triangulated_points",
            "mean_reprojection_error_px",
            "mean_triangulation_angle_deg",
            "warnings",
        ],
        pair_rows,
    )
    ranked_csv = os.path.join(multiview_dir, "pairs_ranked.csv")
    ranked_rows = sorted(
        pair_rows,
        key=lambda row: (
            int(row.get("triangulated_points") or 0),
            int(row.get("ransac_inliers") or 0),
            int(row.get("ratio_matches") or 0),
        ),
        reverse=True,
    )
    _write_csv_rows(
        ranked_csv,
        [
            "serial1",
            "serial2",
            "triangulated_points",
            "ransac_inliers",
            "ratio_matches",
            "raw_matches",
            "mean_reprojection_error_px",
            "mean_triangulation_angle_deg",
            "warnings",
        ],
        ranked_rows,
    )

    heatmap_csv = os.path.join(multiview_dir, "triangulation_heatmap.csv")
    os.makedirs(os.path.dirname(heatmap_csv), exist_ok=True)
    with open(heatmap_csv, "w", encoding="utf-8") as f:
        f.write(",".join(["serial"] + serials) + "\n")
        for row_serial in serials:
            row = [row_serial]
            for col_serial in serials:
                row.append("" if row_serial == col_serial else str(heatmap.get((row_serial, col_serial), 0)))
            f.write(",".join(_csv_cell(value) for value in row) + "\n")

    triangulation_matrix = np.zeros((len(serials), len(serials)), dtype=np.float64)
    ransac_matrix = np.zeros((len(serials), len(serials)), dtype=np.float64)
    for row_index, row_serial in enumerate(serials):
        for col_index, col_serial in enumerate(serials):
            if row_index == col_index:
                continue
            triangulation_matrix[row_index, col_index] = heatmap.get((row_serial, col_serial), 0)
            ransac_matrix[row_index, col_index] = ransac_heatmap.get((row_serial, col_serial), 0)
    triangulation_png = _write_heatmap_png(
        os.path.join(multiview_dir, "triangulation_heatmap.png"),
        serials,
        triangulation_matrix,
        "triangulated points per camera pair",
    )
    ransac_png = _write_heatmap_png(
        os.path.join(multiview_dir, "ransac_inlier_heatmap.png"),
        serials,
        ransac_matrix,
        "RANSAC inliers per camera pair",
    )

    summary = _summarize_pair_reports(pair_reports)
    summary.update(
        {
            "pairs_csv": pairs_csv,
            "pairs_ranked_csv": ranked_csv,
            "triangulation_heatmap_csv": heatmap_csv,
            "triangulation_heatmap_png": triangulation_png,
            "ransac_inlier_heatmap_png": ransac_png,
        }
    )
    _write_json(os.path.join(multiview_dir, "pair_diagnostics.json"), summary)
    return summary


def _write_multiview_session_diagnostics(output_dir: str, summaries: List[dict]) -> dict:
    multiview_root = os.path.join(output_dir, "multiview")
    rows = []
    for summary in summaries:
        pair_diag = summary.get("pair_diagnostics") or {}
        rows.append(
            {
                "state_id": summary.get("state_id"),
                "status": summary.get("status"),
                "camera_count": summary.get("camera_count"),
                "point_count": summary.get("point_count"),
                "pair_count": pair_diag.get("pair_count"),
                "triangulated_pair_count": pair_diag.get("triangulated_pair_count"),
                "triangulated_pair_fraction": pair_diag.get("triangulated_pair_fraction"),
                "triangulated_points_median_all_pairs": pair_diag.get("triangulated_points_median_all_pairs"),
                "triangulated_points_median_nonzero_pairs": pair_diag.get("triangulated_points_median_nonzero_pairs"),
                "ratio_matches_median": pair_diag.get("ratio_matches_median"),
                "ransac_inliers_median": pair_diag.get("ransac_inliers_median"),
                "warnings": summary.get("warnings", []),
            }
        )

    csv_path = os.path.join(multiview_root, "multiview_diagnostics.csv")
    _write_csv_rows(
        csv_path,
        [
            "state_id",
            "status",
            "camera_count",
            "point_count",
            "pair_count",
            "triangulated_pair_count",
            "triangulated_pair_fraction",
            "triangulated_points_median_all_pairs",
            "triangulated_points_median_nonzero_pairs",
            "ratio_matches_median",
            "ransac_inliers_median",
            "warnings",
        ],
        rows,
    )
    manifest = {"state_count": len(summaries), "csv_path": csv_path, "states": rows}
    json_path = os.path.join(multiview_root, "multiview_diagnostics.json")
    _write_json(json_path, manifest)
    manifest["json_path"] = json_path
    return manifest


def _find_state_pointcloud(report: dict, output_dir: str) -> Optional[str]:
    # Prefer the automatically cleaned object cloud (world-volume ROI + ground
    # plane removal) when it exists, so every downstream consumer (registration,
    # canonical loading, segmentation) works on the object-only cloud.
    roi_candidate = os.path.join(output_dir, "multiview", report["state_id"], "object_points_roi.ply")
    if os.path.exists(roi_candidate):
        return roi_candidate

    multiview = report.get("multiview") or {}
    pointcloud_path = multiview.get("pointcloud_path")
    if pointcloud_path and os.path.exists(pointcloud_path):
        return pointcloud_path

    candidate = os.path.join(output_dir, "multiview", report["state_id"], "sparse_points.ply")
    if os.path.exists(candidate):
        return candidate
    return None


def _find_raw_state_pointcloud(report: dict, output_dir: str) -> Optional[str]:
    # Raw whole-scene triangulation cloud, ignoring any ROI-cleaned cloud. Used
    # by the ROI front-end itself so it never consumes its own output.
    multiview = report.get("multiview") or {}
    pointcloud_path = multiview.get("pointcloud_path")
    if (
        pointcloud_path
        and os.path.exists(pointcloud_path)
        and os.path.basename(pointcloud_path) != "object_points_roi.ply"
    ):
        return pointcloud_path

    candidate = os.path.join(output_dir, "multiview", report["state_id"], "sparse_points.ply")
    if os.path.exists(candidate):
        return candidate
    return None


def _estimate_working_volume_center(bundle: Dict[str, dict]) -> Tuple["np.ndarray", bool]:
    """Estimate the scene/working-volume center from camera geometry alone.

    Cameras looking inward at the object have optical axes that converge near the
    object. The least-squares intersection of those axes gives a center that is
    independent of the (per-session, arbitrary) calibration world frame, so no
    known coordinates are required. Returns (center, converged_ok).
    """
    import numpy as np

    centers = []
    dirs = []
    for cam in bundle.values():
        R = np.asarray(cam["cam_from_world"], dtype=np.float64)[:3, :3]
        center = np.asarray(cam["camera_center_world"], dtype=np.float64).reshape(3)
        axis = R[2, :].astype(np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-9:
            continue
        centers.append(center)
        dirs.append(axis / norm)

    if len(centers) == 0:
        return np.zeros(3, dtype=np.float64), False
    centers_arr = np.asarray(centers, dtype=np.float64)
    cam_centroid = np.mean(centers_arr, axis=0)
    if len(centers) < 2:
        return cam_centroid, False

    eye = np.eye(3, dtype=np.float64)
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for c, d in zip(centers, dirs):
        proj = eye - np.outer(d, d)
        A += proj
        b += proj @ c

    ok = True
    try:
        center = np.linalg.solve(A, b)
        if not np.all(np.isfinite(center)):
            raise ValueError("non-finite convergence")
    except Exception:
        center = cam_centroid
        ok = False

    spread = float(np.median(np.linalg.norm(centers_arr - cam_centroid, axis=1))) + 1.0e-9
    if np.linalg.norm(center - cam_centroid) > 5.0 * spread:
        center = cam_centroid
        ok = False
    return center, ok


def _ransac_dominant_plane(
    points: "np.ndarray",
    tolerance: float,
    iterations: int,
    seed: int,
) -> Optional[Tuple["np.ndarray", float, "np.ndarray"]]:
    import numpy as np

    n = int(points.shape[0])
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    best_count = -1
    best_plane = None
    best_inliers = None
    for _ in range(int(iterations)):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1.0e-12:
            continue
        normal = normal / norm
        offset = -float(normal @ p0)
        dist = np.abs(points @ normal + offset)
        inliers = dist <= tolerance
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_plane = (normal, offset)
            best_inliers = inliers
    if best_plane is None:
        return None
    return best_plane[0], best_plane[1], best_inliers


def _build_object_roi_clouds(
    state_reports: List[dict],
    output_dir: str,
    mesh_info: dict,
    args: argparse.Namespace,
) -> dict:
    """Automatic background removal front-end for every per-state cloud.

    For each state: crop the whole-scene triangulation to a world-volume sphere
    (center from camera-axis convergence, radius from the known mesh size), then
    remove the dominant supporting plane (the floor cloth), keeping the side the
    cameras look from. Writes ``object_points_roi.ply`` which downstream stages
    pick up automatically. Fully automatic and registration-independent; no known
    coordinates, no per-view masks, no empty-background capture required.
    """
    import numpy as np

    mode = getattr(args, "roi_crop_mode", "auto")
    summary = {"enabled": False, "mode": mode, "states": [], "warnings": []}
    if bool(getattr(args, "check_only", False)):
        return summary

    if mode == "off":
        for report in state_reports:
            stale = os.path.join(output_dir, "multiview", report["state_id"], "object_points_roi.ply")
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass
        return summary
    summary["enabled"] = True

    mesh_path = mesh_info.get("mesh_path")
    mesh_diag = None
    if mesh_path and os.path.exists(mesh_path):
        try:
            mesh_points = _load_mesh_sample_points(mesh_path, 0)
            mesh_diag = float(np.linalg.norm(np.max(mesh_points, axis=0) - np.min(mesh_points, axis=0)))
        except Exception as exc:  # pragma: no cover - defensive
            summary["warnings"].append(f"mesh diagonal computation failed: {exc}")
    summary["mesh_diagonal"] = mesh_diag

    radius_factor = float(getattr(args, "roi_radius_factor", 1.5))
    ground_tol_fraction = float(getattr(args, "roi_ground_plane_fraction", 0.02))
    ground_min_inlier_fraction = float(getattr(args, "roi_ground_min_inlier_fraction", 0.15))
    ground_side_fraction = float(getattr(args, "roi_ground_side_fraction", 0.6))
    ransac_iters = int(getattr(args, "roi_ground_ransac_iterations", 300))
    min_keep_fraction = float(getattr(args, "roi_min_keep_fraction", 0.05))

    for report in state_reports:
        state_id = report["state_id"]
        info = {"state_id": state_id, "status": "skipped", "warnings": []}
        roi_out_path = os.path.join(output_dir, "multiview", state_id, "object_points_roi.ply")

        raw_path = _find_raw_state_pointcloud(report, output_dir)
        if raw_path is None:
            info["status"] = "no_pointcloud"
            summary["states"].append(info)
            continue
        try:
            points, colors = _read_ascii_ply_points(raw_path)
        except Exception as exc:
            info["status"] = "read_failed"
            info["warnings"].append(str(exc))
            summary["states"].append(info)
            continue
        raw_count = int(points.shape[0])
        info["raw_point_count"] = raw_count
        if raw_count < 10:
            info["status"] = "too_few_points"
            summary["states"].append(info)
            continue
        if colors is None:
            colors = np.full((raw_count, 3), 200, dtype=np.uint8)

        bundle, _ = _load_state_camera_bundle(report["path"], report.get("undistorted_cameras", []))
        if not bundle:
            info["status"] = "no_cameras"
            summary["states"].append(info)
            continue
        center, center_ok = _estimate_working_volume_center(bundle)
        cam_centroid = np.mean(
            np.asarray([cam["camera_center_world"] for cam in bundle.values()], dtype=np.float64),
            axis=0,
        )
        info["center_from_camera_convergence"] = bool(center_ok)
        info["roi_center"] = center.tolist()

        keep = np.ones((raw_count,), dtype=bool)
        if mesh_diag is not None and mesh_diag > 0.0:
            radius = radius_factor * mesh_diag
            dist_to_center = np.linalg.norm(points - center.reshape(1, 3), axis=1)
            roi_keep = dist_to_center <= radius
            if int(np.count_nonzero(roi_keep)) >= max(10, int(min_keep_fraction * raw_count)):
                keep &= roi_keep
                info["roi_radius"] = float(radius)
            else:
                info["warnings"].append("ROI sphere kept too few points; sphere crop skipped.")
        else:
            info["warnings"].append("mesh diagonal unavailable; sphere crop skipped.")

        roi_points = points[keep]
        roi_colors = colors[keep]
        info["after_roi_point_count"] = int(roi_points.shape[0])

        removed_points = None
        removed_colors = None
        ground_removed = 0
        if roi_points.shape[0] >= 50 and mesh_diag is not None and mesh_diag > 0.0:
            tolerance = ground_tol_fraction * mesh_diag
            sample = _subsample_points(roi_points, 6000)
            plane = _ransac_dominant_plane(sample, tolerance, ransac_iters, seed=12345)
            if plane is not None:
                normal, offset, _ = plane
                # Orient normal toward the cameras ("up") so the object sits on the +side.
                if float(normal @ cam_centroid + offset) < 0.0:
                    normal = -normal
                    offset = -offset
                signed = roi_points @ normal + offset
                inlier_mask = np.abs(signed) <= tolerance
                inlier_fraction = float(np.count_nonzero(inlier_mask) / max(roi_points.shape[0], 1))
                above = signed > tolerance
                below = signed < -tolerance
                above_count = int(np.count_nonzero(above))
                below_count = int(np.count_nonzero(below))
                outside = max(above_count + below_count, 1)
                side_ok = above_count >= ground_side_fraction * outside
                enough_object = above_count >= max(10, int(min_keep_fraction * roi_points.shape[0]))
                if inlier_fraction >= ground_min_inlier_fraction and side_ok and enough_object:
                    removed_points = roi_points[~above]
                    removed_colors = roi_colors[~above]
                    ground_removed = int(roi_points.shape[0] - above_count)
                    roi_points = roi_points[above]
                    roi_colors = roi_colors[above]
                    info["ground_plane"] = {
                        "normal": normal.tolist(),
                        "offset": float(offset),
                        "inlier_fraction": inlier_fraction,
                        "above_count": above_count,
                        "below_count": below_count,
                    }
                else:
                    info["warnings"].append(
                        "dominant plane rejected as ground "
                        f"(inlier_fraction={inlier_fraction:.3f}, above={above_count}, below={below_count})."
                    )
        info["ground_removed_point_count"] = ground_removed
        info["object_point_count"] = int(roi_points.shape[0])

        if roi_points.shape[0] < max(10, int(min_keep_fraction * raw_count)):
            info["status"] = "insufficient_object_points"
            info["warnings"].append("object cloud too small after ROI+ground; ROI file not written (raw cloud kept).")
            if os.path.exists(roi_out_path):
                try:
                    os.remove(roi_out_path)
                except OSError:
                    pass
            summary["states"].append(info)
            continue

        _write_ply(roi_out_path, roi_points, roi_colors)
        info["object_points_roi_path"] = roi_out_path
        if removed_points is not None and removed_points.shape[0] > 0:
            removed_path = os.path.join(output_dir, "multiview", state_id, "object_roi_removed.ply")
            _write_ply(removed_path, removed_points, removed_colors)
            info["removed_path"] = removed_path
        info["status"] = "ok"
        summary["states"].append(info)
        print(
            f"[ROI] {state_id}: raw={raw_count} -> roi={info['after_roi_point_count']} "
            f"-> object={info['object_point_count']} (ground_removed={ground_removed}, "
            f"center_ok={center_ok})"
        )

    return summary


def _write_target_points(path: str, points: "np.ndarray", colors: Optional["np.ndarray"]) -> None:
    if colors is not None and colors.shape[0] == points.shape[0]:
        _write_ply(path, points, colors)
    else:
        _write_points_ply(path, points, (200, 220, 255))


def _part_mask_base_roots(args: argparse.Namespace, report: dict, session_path: str) -> List[str]:
    if args.part_mask_root:
        return _expand_path_candidates(args.part_mask_root, session_path)
    return [os.path.join(report["path"], "masks")]


def _part_mask_search_dirs(
    args: argparse.Namespace,
    report: dict,
    part: dict,
    session_path: str,
) -> List[str]:
    state_id = report["state_id"]
    part_keys = [part["part_id"]]
    label = part.get("label")
    if label and label not in part_keys:
        part_keys.append(label)

    dirs = []
    for root in _part_mask_base_roots(args, report, session_path):
        for key in part_keys:
            dirs.append(os.path.join(root, state_id, key))
            dirs.append(os.path.join(root, key))
    deduped = []
    for path in dirs:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _find_part_mask_paths(
    args: argparse.Namespace,
    report: dict,
    part: dict,
    bundle: Dict[str, dict],
    session_path: str,
) -> Tuple[Dict[str, str], List[str]]:
    search_dirs = _part_mask_search_dirs(args, report, part, session_path)
    mask_paths: Dict[str, str] = {}
    for serial in sorted(bundle.keys()):
        for mask_dir in search_dirs:
            mask_path = _find_image_path(mask_dir, serial)
            if mask_path is not None:
                mask_paths[serial] = mask_path
                break
    return mask_paths, search_dirs


def _filter_points_by_masks(
    points: "np.ndarray",
    colors: Optional["np.ndarray"],
    bundle: Dict[str, dict],
    mask_paths: Dict[str, str],
    args: argparse.Namespace,
) -> Tuple["np.ndarray", Optional["np.ndarray"], dict]:
    import cv2
    import numpy as np

    vote_count = np.zeros((points.shape[0],), dtype=np.int32)
    visible_count = np.zeros((points.shape[0],), dtype=np.int32)
    per_camera = []
    threshold = float(args.part_mask_threshold)

    for serial, mask_path in sorted(mask_paths.items()):
        cam = bundle.get(serial)
        if cam is None:
            continue

        mask = _cv2_imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            per_camera.append({"serial": serial, "mask_path": mask_path, "status": "unreadable"})
            continue

        pixels = _project_points(cam["projection"], points)
        depths = _depths_in_camera(cam["cam_from_world"], points)
        xy = np.rint(pixels).astype(np.int64)
        h, w = mask.shape[:2]
        valid = (
            np.isfinite(pixels).all(axis=1)
            & (depths > 0.0)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < w)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < h)
        )
        visible_count[valid] += 1

        inside = np.zeros((points.shape[0],), dtype=bool)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size > 0:
            values = mask[xy[valid_indices, 1], xy[valid_indices, 0]]
            inside[valid_indices] = values >= threshold
            vote_count[inside] += 1

        per_camera.append(
            {
                "serial": serial,
                "mask_path": mask_path,
                "status": "ok",
                "visible_points": int(np.count_nonzero(valid)),
                "inside_points": int(np.count_nonzero(inside)),
                "width": int(w),
                "height": int(h),
            }
        )

    min_views = max(1, int(args.part_mask_min_views))
    keep = vote_count >= min_views
    filtered_points = points[keep]
    filtered_colors = colors[keep] if colors is not None and colors.shape[0] == points.shape[0] else None
    info = {
        "mask_view_count": len(mask_paths),
        "source_point_count": int(points.shape[0]),
        "selected_point_count": int(filtered_points.shape[0]),
        "min_views": min_views,
        "threshold": threshold,
        "per_camera": per_camera,
    }
    return filtered_points, filtered_colors, info


def _select_part_target_points(
    report: dict,
    part: dict,
    part_dir: str,
    source_points: "np.ndarray",
    source_colors: Optional["np.ndarray"],
    bundle: Dict[str, dict],
    segmented_targets: Dict[str, dict],
    args: argparse.Namespace,
    session_path: str,
) -> Tuple["np.ndarray", Optional["np.ndarray"], dict]:
    import numpy as np

    info = {
        "scope": "whole_state_sparse_pointcloud",
        "pointcloud_path": None,
        "source_point_count": int(source_points.shape[0]),
        "target_point_count": int(source_points.shape[0]),
        "mask_paths": {},
        "mask_search_dirs": [],
        "warnings": [],
    }

    mask_paths, search_dirs = _find_part_mask_paths(args, report, part, bundle, session_path)
    info["mask_paths"] = mask_paths
    info["mask_search_dirs"] = search_dirs
    min_mask_points = max(3, int(args.part_mask_min_points))

    if mask_paths:
        masked_points, masked_colors, mask_info = _filter_points_by_masks(
            source_points,
            source_colors,
            bundle,
            mask_paths,
            args,
        )
        info["mask_filter"] = mask_info
        if masked_points.shape[0] >= min_mask_points:
            target_path = os.path.join(part_dir, "target_masked_sparse_points.ply")
            _write_target_points(target_path, masked_points, masked_colors)
            info["scope"] = "part_mask_sparse_pointcloud"
            info["pointcloud_path"] = target_path
            info["target_point_count"] = int(masked_points.shape[0])
            return masked_points, masked_colors, info

        info["warnings"].append(
            f"part masks selected only {masked_points.shape[0]} points; {min_mask_points}+ required."
        )
        if args.require_part_masks:
            empty_colors = None if source_colors is None else source_colors[:0]
            info["scope"] = "part_mask_sparse_pointcloud"
            info["target_point_count"] = int(masked_points.shape[0])
            return masked_points, empty_colors, info
    else:
        default_root = os.path.join(report["path"], "masks")
        if args.part_mask_root or os.path.isdir(default_root):
            info["warnings"].append("no mask files found for this part and state.")
        if args.require_part_masks:
            empty_points = np.zeros((0, 3), dtype=source_points.dtype)
            empty_colors = None if source_colors is None else source_colors[:0]
            info["target_point_count"] = 0
            return empty_points, empty_colors, info

    segmented_target = segmented_targets.get(part["part_id"])
    min_segmented_points = max(3, int(args.part_target_min_points))
    if segmented_target:
        target_points = segmented_target["points"]
        target_colors = segmented_target.get("colors")
        info["segmented_mesh_target"] = segmented_target["info"]
        if target_points.shape[0] >= min_segmented_points:
            target_path = os.path.join(part_dir, "target_segmented_sparse_points.ply")
            _write_target_points(target_path, target_points, target_colors)
            info["scope"] = "segmented_mesh_sparse_pointcloud"
            info["pointcloud_path"] = target_path
            info["target_point_count"] = int(target_points.shape[0])
            return target_points, target_colors, info
        info["warnings"].append(
            f"segmented mesh target selected only {target_points.shape[0]} points; {min_segmented_points}+ required."
        )

    target_path = os.path.join(part_dir, "target_whole_sparse_points.ply")
    _write_target_points(target_path, source_points, source_colors)
    info["pointcloud_path"] = target_path
    info["warnings"].append("falling back to whole-state sparse pointcloud.")
    return source_points, source_colors, info


def _object_space_rotation_transform(
    center: "np.ndarray",
    axis: "np.ndarray",
    angle_rad: float,
) -> "np.ndarray":
    import numpy as np

    R = _rotation_about_axis(axis, angle_rad)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(center, dtype=np.float64).reshape(3) - R @ np.asarray(center, dtype=np.float64).reshape(3)
    return T


def _object_space_scale_transform(center: "np.ndarray", scale: float) -> "np.ndarray":
    import numpy as np

    scale = float(scale)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] *= scale
    c = np.asarray(center, dtype=np.float64).reshape(3)
    T[:3, 3] = c - scale * c
    return T


def _object_space_translation_transform(offset: "np.ndarray") -> "np.ndarray":
    import numpy as np

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(offset, dtype=np.float64).reshape(3)
    return T


def _same_placement_group(report_a: dict, report_b: dict) -> bool:
    group_a = report_a.get("placement_group_id")
    group_b = report_b.get("placement_group_id")
    if group_a is None or group_b is None:
        return True
    return int(group_a) == int(group_b)


def _parse_manual_placement_groups(spec: str, state_ids: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    groups = [group for group in (spec or "").split(";") if group.strip()]
    for group_id, group in enumerate(groups):
        for token in group.split(","):
            token = token.strip()
            if token:
                mapping[token] = group_id
    next_group_id = len(groups)
    for state_id in state_ids:
        if state_id not in mapping:
            mapping[state_id] = next_group_id
            next_group_id += 1
    return mapping


def _detect_placement_groups(
    state_reports: List[dict],
    output_dir: str,
    mesh_info: dict,
    args: argparse.Namespace,
) -> dict:
    """Group states by object placement (hybrid capture support).

    A hybrid session repositions the whole object between placement groups and only
    articulates the moving part inside a group. Image-difference channels (image_diff
    evidence, silhouette evidence, moving-part carve) are only meaningful between
    states of the same placement, so every state report is annotated with
    ``placement_group_id``. Auto mode clusters per-state object-cloud centroids
    (states whose object stayed in place share a centroid up to the moving-part
    displacement); ``manual`` takes explicit groups; ``single`` restores the legacy
    one-group behavior.
    """
    import numpy as np

    mode = getattr(args, "placement_group_mode", "auto")
    summary: dict = {"enabled": True, "mode": mode, "groups": [], "warnings": []}
    state_ids = [str(report.get("state_id")) for report in state_reports]

    if bool(getattr(args, "check_only", False)) or mode == "single":
        for report in state_reports:
            report["placement_group_id"] = 0
        summary["groups"] = [{"group_id": 0, "state_ids": state_ids}]
        return summary

    if mode == "manual":
        mapping = _parse_manual_placement_groups(getattr(args, "placement_groups", "") or "", state_ids)
        for report in state_reports:
            report["placement_group_id"] = int(mapping[str(report.get("state_id"))])
        group_states: Dict[int, List[str]] = {}
        for state_id in state_ids:
            group_states.setdefault(int(mapping[state_id]), []).append(state_id)
        summary["groups"] = [
            {"group_id": group_id, "state_ids": members}
            for group_id, members in sorted(group_states.items())
        ]
        for group in summary["groups"]:
            print(f"[GROUP] manual group {group['group_id']}: states={group['state_ids']}")
        return summary

    mesh_diag = None
    mesh_path = mesh_info.get("mesh_path")
    if mesh_path and os.path.exists(mesh_path):
        try:
            mesh_points = _load_mesh_sample_points(mesh_path, 0)
            mesh_diag = float(np.linalg.norm(np.max(mesh_points, axis=0) - np.min(mesh_points, axis=0)))
        except Exception as exc:  # pragma: no cover - defensive
            summary["warnings"].append(f"mesh diagonal computation failed: {exc}")

    centroids: List[Optional["np.ndarray"]] = []
    for report in state_reports:
        centroid = None
        cloud_path = _find_state_pointcloud(report, output_dir)
        if cloud_path is not None:
            try:
                points, _colors = _read_ascii_ply_points(cloud_path)
                if points.shape[0] >= 10:
                    centroid = np.median(points, axis=0)
            except Exception as exc:  # pragma: no cover - defensive
                summary["warnings"].append(f"{report.get('state_id')}: centroid read failed: {exc}")
        centroids.append(centroid)

    if mesh_diag is None or mesh_diag <= 0.0:
        summary["warnings"].append(
            "mesh diagonal unavailable; placement auto-detection skipped (all states in one group)."
        )
        for report in state_reports:
            report["placement_group_id"] = 0
        summary["groups"] = [{"group_id": 0, "state_ids": state_ids}]
        return summary

    distance_fraction = float(getattr(args, "placement_group_distance_fraction", 0.35))
    threshold = distance_fraction * mesh_diag
    summary["distance_threshold"] = threshold
    summary["mesh_diagonal"] = mesh_diag

    count = len(state_reports)
    group_of = [-1] * count
    next_group_id = 0
    for index in range(count):
        if group_of[index] >= 0:
            continue
        group_of[index] = next_group_id
        if centroids[index] is None:
            summary["warnings"].append(
                f"{state_ids[index]}: no object cloud centroid; placed in its own group."
            )
            next_group_id += 1
            continue
        stack = [index]
        while stack:
            current = stack.pop()
            for other in range(count):
                if group_of[other] >= 0 or centroids[other] is None:
                    continue
                if float(np.linalg.norm(centroids[current] - centroids[other])) <= threshold:
                    group_of[other] = next_group_id
                    stack.append(other)
        next_group_id += 1

    group_states = {}
    for index, report in enumerate(state_reports):
        report["placement_group_id"] = int(group_of[index])
        entry = group_states.setdefault(int(group_of[index]), {"state_ids": [], "centroid": None})
        entry["state_ids"].append(state_ids[index])
        if centroids[index] is not None and entry["centroid"] is None:
            entry["centroid"] = centroids[index]

    summary["groups"] = []
    for group_id in sorted(group_states.keys()):
        entry = group_states[group_id]
        member_centroids = [
            centroids[index]
            for index in range(count)
            if group_of[index] == group_id and centroids[index] is not None
        ]
        group_info = {
            "group_id": group_id,
            "state_ids": entry["state_ids"],
            "centroid": np.mean(np.asarray(member_centroids), axis=0).tolist() if member_centroids else None,
            "max_intra_distance": (
                float(
                    max(
                        np.linalg.norm(a - b)
                        for a in member_centroids
                        for b in member_centroids
                    )
                )
                if len(member_centroids) >= 2
                else 0.0
            ),
        }
        summary["groups"].append(group_info)
        print(
            f"[GROUP] auto group {group_id}: states={group_info['state_ids']} "
            f"max_intra={group_info['max_intra_distance']:.4f} thr={threshold:.4f}"
        )
    if len(summary["groups"]) == 1:
        summary["warnings"].append(
            "only one placement group detected; background-plate masks need 2+ placements."
        )
    return summary


def _find_object_mask_path(output_dir: Optional[str], state_id: Optional[str], serial: str) -> Optional[str]:
    if not output_dir or not state_id:
        return None
    path = os.path.join(output_dir, "object_masks", str(state_id), f"{serial}.png")
    return path if os.path.exists(path) else None


def _find_trust_mask_path(output_dir: Optional[str], serial: str) -> Optional[str]:
    if not output_dir:
        return None
    path = os.path.join(output_dir, "background_plates", f"trust_{serial}.png")
    return path if os.path.exists(path) else None


def _plate_cloth_like_mask(plate: "np.ndarray", args: argparse.Namespace) -> "np.ndarray":
    """Plate pixels that look like the capture cloth (bright and unsaturated).

    |image - plate| is only a reliable object/background test where the background
    is the controlled cloth; against the cluttered room the diff is fragmentary
    (holes where object color matches the background, background-shaped blobs
    where the object occludes structure). The same test also bounds where cast
    shadows can be suppressed.
    """
    import numpy as np

    plate_f = plate.astype(np.float64)
    plate_mean = np.mean(plate_f, axis=2)
    plate_spread = np.max(plate_f, axis=2) - np.min(plate_f, axis=2)
    return (
        plate_mean >= float(getattr(args, "object_mask_shadow_plate_min_brightness", 110.0))
    ) & (
        plate_spread <= float(getattr(args, "object_mask_shadow_plate_max_chroma", 45.0))
    )


def _plate_trust_mask(plate: "np.ndarray", args: argparse.Namespace) -> "np.ndarray":
    import cv2
    import numpy as np

    trust = _plate_cloth_like_mask(plate, args)
    erode = max(0, int(getattr(args, "mask_trust_erode", 1)))
    if erode > 0:
        # The cloth boundary in the plate is a soft gradient; eroding keeps those
        # ambiguous edge pixels out of the trusted region.
        trust = cv2.erode(trust.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=erode) > 0
    return trust


def _object_mask_from_plate(
    image: "np.ndarray",
    plate: "np.ndarray",
    args: argparse.Namespace,
) -> Tuple["np.ndarray", dict]:
    """Object mask for one view: threshold |image - background plate| with cleanup.

    The plate is the per-pixel median over all states, so anything that differs is
    object (or its shadow). Shadow suppression removes pixels that are darker than
    the plate but keep the plate's chromaticity (a shadow on the white cloth), which
    would otherwise inflate the mask.
    """
    import cv2
    import numpy as np

    diff = np.max(cv2.absdiff(image, plate), axis=2)
    blur = max(0, int(getattr(args, "object_mask_blur", 5)))
    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        diff = cv2.GaussianBlur(diff, (blur, blur), 0)
    threshold = float(getattr(args, "object_mask_threshold", 30.0))
    mask = diff >= threshold
    raw_fraction = float(np.mean(mask)) if mask.size else 0.0

    shadow_removed_fraction = 0.0
    if str(getattr(args, "object_mask_suppress_shadows", "on")) == "on":
        image_f = image.astype(np.float64)
        plate_f = plate.astype(np.float64)
        image_sum = np.sum(image_f, axis=2) + 1.0e-6
        plate_sum = np.sum(plate_f, axis=2) + 1.0e-6
        ratio = image_sum / plate_sum
        chroma_dist = np.sum(
            np.abs(image_f / image_sum[..., None] - plate_f / plate_sum[..., None]),
            axis=2,
        )
        low = float(getattr(args, "object_mask_shadow_brightness_low", 0.35))
        high = float(getattr(args, "object_mask_shadow_brightness_high", 0.97))
        chroma_threshold = float(getattr(args, "object_mask_shadow_chroma_threshold", 0.10))
        shadow = (ratio >= low) & (ratio <= high) & (chroma_dist <= chroma_threshold)
        # A cast shadow only looks like "darker background" where the background is the
        # bright, unsaturated cloth. Against other backgrounds (walls, equipment) the
        # same rule matches object pixels and punches holes in the mask, so the
        # suppression is restricted to cloth-like plate pixels.
        cloth_like = _plate_cloth_like_mask(plate, args)
        shadow = shadow & cloth_like
        removed = mask & shadow
        shadow_removed_fraction = float(np.mean(removed)) if removed.size else 0.0
        mask = mask & (~shadow)

    mask_u8 = mask.astype(np.uint8)
    morph_iterations = max(0, int(getattr(args, "object_mask_morph_iterations", 2)))
    if morph_iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    min_component_fraction = float(getattr(args, "object_mask_min_component_fraction", 0.0005))
    if min_component_fraction > 0.0 and bool(np.any(mask_u8)):
        min_area = int(min_component_fraction * mask_u8.shape[0] * mask_u8.shape[1])
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        keep = np.zeros_like(mask_u8)
        for component_id in range(1, component_count):
            if int(stats[component_id, cv2.CC_STAT_AREA]) >= max(1, min_area):
                keep[labels == component_id] = 1
        mask_u8 = keep

    final_mask = mask_u8 > 0
    info = {
        "threshold": threshold,
        "raw_fraction": raw_fraction,
        "shadow_removed_fraction": shadow_removed_fraction,
        "mask_fraction": float(np.mean(final_mask)) if final_mask.size else 0.0,
    }
    return final_mask, info


def _external_mask_summary(
    state_reports: List[dict],
    output_dir: str,
    args: argparse.Namespace,
) -> Optional[dict]:
    """Consume externally generated object masks (e.g. generate_masks_sam3.py).

    Returns a summary when the external source applies, else None (the caller
    falls back to plate building). External masks follow the same disk contract
    as the plate path (object_masks/<state>/<serial>.png, 255 = object), so hull
    voting and mask-scored registration consume them unchanged. Plate trust
    gating is force-disabled here: trust maps describe where *plate* content is
    reliable, which says nothing about a learned mask, and stale trust maps from
    an earlier plate run in the same output dir must not gate the new masks.
    """
    source = str(getattr(args, "object_mask_source", "auto"))
    if source == "plate":
        return None
    masks_root = os.path.join(output_dir, "object_masks")
    manifest_path = os.path.join(masks_root, "masks_manifest.json")
    manifest = _load_optional_json(manifest_path) if os.path.exists(manifest_path) else None
    provider = str((manifest or {}).get("provider", "")).strip() or None
    if source == "auto" and (provider is None or provider == "plate"):
        return None
    if not os.path.isdir(masks_root):
        raise FileNotFoundError(
            f"--object-mask-source external: {masks_root} does not exist; "
            "generate masks first (see generate_masks_sam3.py in this folder)."
        )

    summary: dict = {
        "enabled": True,
        "mode": "external",
        "provider": provider or "unknown",
        "prompt": (manifest or {}).get("prompt"),
        "manifest_path": manifest_path if manifest is not None else None,
        "serials": [],
        "states": [],
        "warnings": [],
    }
    if manifest is None:
        summary["warnings"].append("external masks without masks_manifest.json; provider unknown.")

    total_masks = 0
    for report in state_reports:
        state_id = str(report.get("state_id"))
        serials = sorted(report.get("undistorted_cameras", []))
        mask_count = sum(
            1 for serial in serials if _find_object_mask_path(output_dir, state_id, serial) is not None
        )
        total_masks += mask_count
        summary["states"].append(
            {"state_id": state_id, "camera_count": len(serials), "mask_count": mask_count}
        )
        if mask_count < 3:
            summary["warnings"].append(
                f"{state_id}: only {mask_count} external masks; visual hull needs 3+ masked cameras."
            )
    if total_masks == 0:
        raise FileNotFoundError(
            f"External mask source selected but no per-state masks found under {masks_root}; "
            "check that the generator used the same --output-dir and state ids."
        )

    if str(getattr(args, "mask_trust_mode", "auto")) != "off":
        args.mask_trust_mode = "off"
        print("[MASKS] external masks: plate trust gating disabled (trust maps only describe plate reliability).")
    print(
        f"[MASKS] external provider={summary['provider']} states={len(summary['states'])} "
        f"masks={total_masks} (plate building skipped)"
    )
    return summary


def _build_background_plates_and_masks(
    state_reports: List[dict],
    output_dir: str,
    args: argparse.Namespace,
) -> dict:
    """Median background plates + per-state per-view object masks (hybrid capture).

    With the object at 2+ different placements and fixed cameras, the per-pixel
    median over all states of one camera recovers the empty background (each pixel
    is covered by the object in only a minority of states). Each state's object
    mask is then |image - plate| with shadow suppression and cleanup. These masks
    are texture-independent full silhouettes in every view, and are consumed by the
    mask-scored registration refinement.

    When ``--object-mask-source`` resolves to external (a masks_manifest.json from
    generate_masks_sam3.py, or the flag set explicitly), plate building is skipped
    entirely and the pre-generated masks are summarized instead.
    """
    import cv2
    import numpy as np

    mode = getattr(args, "background_plate_mode", "auto")
    summary: dict = {"enabled": False, "mode": mode, "serials": [], "states": [], "warnings": []}
    if bool(getattr(args, "check_only", False)):
        return summary
    external_summary = _external_mask_summary(state_reports, output_dir, args)
    if external_summary is not None:
        return external_summary
    if mode == "off":
        return summary

    group_ids = {
        report.get("placement_group_id")
        for report in state_reports
        if report.get("placement_group_id") is not None
    }
    if len(group_ids) < 2:
        summary["warnings"].append(
            "fewer than 2 placement groups; a median background plate would still contain the object. "
            "Capture the object at 2+ (ideally 3+) placements, or force --placement-group-mode manual."
        )
        return summary
    summary["enabled"] = True

    bundles = {}
    for report in state_reports:
        bundle, _warnings = _load_state_camera_bundle(report["path"], report.get("undistorted_cameras", []))
        if bundle:
            bundles[str(report.get("state_id"))] = bundle

    serial_states: Dict[str, List[str]] = {}
    for state_id, bundle in bundles.items():
        for serial in bundle.keys():
            serial_states.setdefault(serial, []).append(state_id)

    plates_dir = os.path.join(output_dir, "background_plates")
    masks_root = os.path.join(output_dir, "object_masks")
    overlay_cameras = max(0, int(getattr(args, "object_mask_overlay_cameras", 4)))
    state_rows: Dict[str, dict] = {
        str(report.get("state_id")): {
            "state_id": str(report.get("state_id")),
            "mask_count": 0,
            "mean_mask_fraction": 0.0,
        }
        for report in state_reports
    }
    csv_rows: List[dict] = []

    for serial_index, serial in enumerate(sorted(serial_states.keys())):
        state_ids = sorted(serial_states[serial])
        if len(state_ids) < 3:
            summary["warnings"].append(f"{serial}: only {len(state_ids)} states; plate skipped.")
            continue
        images = {}
        shape = None
        for state_id in state_ids:
            image = _cv2_imread(bundles[state_id][serial]["image_path"], cv2.IMREAD_COLOR)
            if image is None:
                continue
            if shape is None:
                shape = image.shape
            if image.shape != shape:
                summary["warnings"].append(f"{serial}: {state_id} image shape mismatch; excluded from plate.")
                continue
            images[state_id] = image
        if len(images) < 3:
            summary["warnings"].append(f"{serial}: fewer than 3 readable images; plate skipped.")
            continue

        stack = np.stack([images[state_id] for state_id in sorted(images.keys())], axis=0)
        plate = np.median(stack, axis=0).astype(np.uint8)
        del stack
        os.makedirs(plates_dir, exist_ok=True)
        plate_path = os.path.join(plates_dir, f"{serial}.png")
        _cv2_imwrite(plate_path, plate)

        trust = None
        trust_path = None
        trusted_fraction = None
        if str(getattr(args, "mask_trust_mode", "auto")) != "off":
            trust = _plate_trust_mask(plate, args)
            trust_path = os.path.join(plates_dir, f"trust_{serial}.png")
            _cv2_imwrite(trust_path, (trust.astype(np.uint8) * 255))
            trusted_fraction = float(np.mean(trust))
            if serial_index < overlay_cameras:
                dimmed = plate.copy()
                dimmed[~trust] = (0.35 * dimmed[~trust]).astype(np.uint8)
                _cv2_imwrite(os.path.join(plates_dir, f"trust_overlay_{serial}.jpg"), dimmed)

        serial_info = {
            "serial": serial,
            "plate_path": plate_path,
            "trust_path": trust_path,
            "trusted_fraction": trusted_fraction,
            "state_count": int(len(images)),
            "masks": 0,
        }

        for state_id in sorted(images.keys()):
            mask, info = _object_mask_from_plate(images[state_id], plate, args)
            state_mask_dir = os.path.join(masks_root, state_id)
            os.makedirs(state_mask_dir, exist_ok=True)
            mask_path = os.path.join(state_mask_dir, f"{serial}.png")
            _cv2_imwrite(mask_path, (mask.astype(np.uint8) * 255))
            serial_info["masks"] += 1
            row = state_rows.get(state_id)
            if row is not None:
                row["mean_mask_fraction"] = (
                    (row["mean_mask_fraction"] * row["mask_count"] + info["mask_fraction"])
                    / max(row["mask_count"] + 1, 1)
                )
                row["mask_count"] += 1
            mask_trusted_fraction = float("nan")
            if trust is not None and bool(np.any(mask)):
                mask_trusted_fraction = float(np.mean(trust[mask]))
            csv_rows.append(
                {
                    "state_id": state_id,
                    "serial": serial,
                    "raw_fraction": info["raw_fraction"],
                    "shadow_removed_fraction": info["shadow_removed_fraction"],
                    "mask_fraction": info["mask_fraction"],
                    "mask_trusted_fraction": mask_trusted_fraction,
                    "mask_path": mask_path,
                }
            )
            if serial_index < overlay_cameras:
                overlay = images[state_id].copy()
                overlay[mask] = (
                    0.45 * overlay[mask] + 0.55 * np.array([0, 200, 255], dtype=np.float64)
                ).astype(np.uint8)
                panel = np.hstack([images[state_id], overlay])
                _cv2_imwrite(os.path.join(state_mask_dir, f"overlay_{serial}.jpg"), panel)

        summary["serials"].append(serial_info)
        trust_note = f" trusted={trusted_fraction:.2f}" if trusted_fraction is not None else ""
        print(f"[PLATE] {serial}: states={serial_info['state_count']} masks={serial_info['masks']}{trust_note}")

    summary["states"] = list(state_rows.values())
    if csv_rows:
        stats_path = os.path.join(masks_root, "mask_stats.csv")
        _write_csv_rows(
            stats_path,
            [
                "state_id",
                "serial",
                "raw_fraction",
                "shadow_removed_fraction",
                "mask_fraction",
                "mask_trusted_fraction",
                "mask_path",
            ],
            csv_rows,
        )
        summary["mask_stats_csv"] = stats_path
    if not summary["serials"]:
        summary["warnings"].append("no background plates could be built.")
    return summary


def _build_visual_hull_clouds(
    state_reports: List[dict],
    output_dir: str,
    object_roi_summary: dict,
    args: argparse.Namespace,
) -> dict:
    """Per-state visual hull from the background-plate object masks.

    Voxels of the camera-derived working volume are kept when they project inside
    the object mask in at least ``--hull-min-view-fraction`` of the cameras that
    see them. Majority voting absorbs per-view mask defects (holes where the
    object color matches the background behind it, false positives from moved
    clutter): a bad mask in one view is vetoed by the other views. With trust
    gating (``--mask-trust-mode auto``) a camera only judges a voxel when it
    projects onto trusted (cloth-backed) pixels, so off-cloth mask holes cannot
    carve real object voxels. The exported hull surface is a dense, texture-free,
    volumetric observation of the whole object in world coordinates — the
    registration target that a flat sparse cluster could never provide.
    """
    import cv2
    import numpy as np

    mode = getattr(args, "visual_hull_mode", "auto")
    summary: dict = {"enabled": False, "mode": mode, "states": [], "warnings": []}
    if mode == "off" or bool(getattr(args, "check_only", False)):
        return summary
    masks_root = os.path.join(output_dir, "object_masks")
    if not os.path.isdir(masks_root):
        summary["warnings"].append("no object masks (needs 2+ placements); visual hull skipped.")
        return summary
    summary["enabled"] = True

    roi_states = {s.get("state_id"): s for s in (object_roi_summary or {}).get("states", [])}
    res = max(16, int(getattr(args, "hull_grid_resolution", 96)))
    min_view_fraction = float(getattr(args, "hull_min_view_fraction", 0.7))
    min_opportunities = int(getattr(args, "hull_min_opportunities", 3))
    ground_margin_fraction = float(getattr(args, "hull_ground_margin_fraction", 0.01))
    out_root = os.path.join(output_dir, "visual_hull")
    identity = np.eye(4, dtype=np.float64)

    for report in state_reports:
        state_id = str(report.get("state_id"))
        info = {"state_id": state_id, "status": "skipped"}
        bundle, _warnings = _load_state_camera_bundle(report["path"], report.get("undistorted_cameras", []))
        mask_serials = [
            serial for serial in sorted(bundle.keys())
            if _find_object_mask_path(output_dir, state_id, serial) is not None
        ]
        if len(mask_serials) < max(3, min_opportunities):
            info["status"] = "too_few_masked_cameras"
            info["masked_camera_count"] = len(mask_serials)
            summary["states"].append(info)
            continue

        state_roi = roi_states.get(report.get("state_id")) or roi_states.get(state_id) or {}
        center = np.asarray(state_roi.get("roi_center", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        radius = float(state_roi.get("roi_radius", 0.0))
        if radius <= 0.0:
            info["status"] = "no_roi_radius"
            summary["states"].append(info)
            summary["warnings"].append(f"{state_id}: ROI radius unavailable for hull volume.")
            continue
        ground = state_roi.get("ground_plane")

        lin = np.linspace(-radius, radius, res)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1) + center.reshape(1, 3)
        inside = np.linalg.norm(grid - center.reshape(1, 3), axis=1) <= radius
        if ground is not None and ground.get("normal") is not None:
            normal = np.asarray(ground["normal"], dtype=np.float64)
            offset = float(ground.get("offset", 0.0))
            inside &= (grid @ normal + offset) > ground_margin_fraction * radius
        voxels = grid[inside]
        if voxels.shape[0] == 0:
            info["status"] = "no_voxels"
            summary["states"].append(info)
            continue

        votes = np.zeros((voxels.shape[0],), dtype=np.float64)
        opportunities = np.zeros((voxels.shape[0],), dtype=np.float64)
        used_cameras = 0
        trust_gated_cameras = 0
        mask_erode = max(0, int(getattr(args, "hull_mask_erode", 0)))
        trust_mode_on = str(getattr(args, "mask_trust_mode", "auto")) != "off"
        for serial in mask_serials:
            mask_path = _find_object_mask_path(output_dir, state_id, serial)
            mask_image = _cv2_imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                continue
            mask = mask_image > 127
            if mask_erode > 0:
                # Blur/close in the mask stage systematically inflates the silhouette;
                # eroding before voting thins the hull (holes are absorbed by the
                # multi-view vote, bloat is not).
                mask = cv2.erode(
                    mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=mask_erode
                ) > 0
            trust = None
            if trust_mode_on:
                trust_path = _find_trust_mask_path(output_dir, serial)
                if trust_path is not None:
                    trust_image = _cv2_imread(trust_path, cv2.IMREAD_GRAYSCALE)
                    if trust_image is not None and trust_image.shape[:2] == mask.shape[:2]:
                        trust = trust_image > 127
            visible, hits = _vote_projected_vertices_from_mask(
                voxels, identity, bundle[serial], mask, valid_mask=trust
            )
            opportunities[visible] += 1.0
            votes[hits] += 1.0
            used_cameras += 1
            if trust is not None:
                trust_gated_cameras += 1

        occupied = (opportunities >= float(min_opportunities)) & (
            votes >= min_view_fraction * np.maximum(opportunities, 1.0)
        )
        filled_count = int(np.count_nonzero(occupied))
        info["used_cameras"] = used_cameras
        info["trust_gated_cameras"] = trust_gated_cameras
        info["filled_voxel_count"] = filled_count
        if filled_count == 0:
            info["status"] = "empty_hull"
            summary["states"].append(info)
            print(f"[HULL] {state_id}: cameras={used_cameras} filled=0 (empty hull)")
            continue

        occupancy = np.zeros((res * res * res,), dtype=bool)
        inside_indices = np.flatnonzero(inside)
        occupancy[inside_indices[occupied]] = True
        occupancy = occupancy.reshape(res, res, res)
        padded = np.pad(occupancy, 1, mode="constant", constant_values=False)
        neighbors_full = (
            padded[2:, 1:-1, 1:-1]
            & padded[:-2, 1:-1, 1:-1]
            & padded[1:-1, 2:, 1:-1]
            & padded[1:-1, :-2, 1:-1]
            & padded[1:-1, 1:-1, 2:]
            & padded[1:-1, 1:-1, :-2]
        )
        surface = occupancy & (~neighbors_full)
        surface_points = grid[surface.ravel()]
        info["surface_voxel_count"] = int(surface_points.shape[0])

        os.makedirs(out_root, exist_ok=True)
        hull_path = os.path.join(out_root, f"hull_{state_id}.ply")
        _write_points_ply(hull_path, surface_points, (80, 220, 80))
        info["path"] = hull_path
        info["status"] = "ok"
        info["voxel_size"] = float(2.0 * radius / max(res - 1, 1))
        summary["states"].append(info)
        print(
            f"[HULL] {state_id}: cameras={used_cameras} trust={trust_gated_cameras}/{used_cameras} "
            f"filled={filled_count} surface={int(surface_points.shape[0])} -> {hull_path}"
        )

    return summary


def _find_state_hull_pointcloud(output_dir: str, state_id: str) -> Optional[str]:
    path = os.path.join(output_dir, "visual_hull", f"hull_{state_id}.ply")
    return path if os.path.exists(path) else None


def _silhouette_boundary(mask: "np.ndarray") -> "np.ndarray":
    import cv2
    import numpy as np

    if mask is None or not bool(np.any(mask)):
        return np.zeros_like(mask, dtype=bool)
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return mask & (~eroded)


def _silhouette_edge_score(
    body_mask: "np.ndarray",
    edge_distance: "np.ndarray",
    trim_fraction: float,
    roi_mask: Optional["np.ndarray"] = None,
) -> Tuple[float, dict]:
    import numpy as np

    body_px = int(np.count_nonzero(body_mask))
    boundary = _silhouette_boundary(body_mask)
    boundary_px = int(np.count_nonzero(boundary))
    if roi_mask is not None:
        roi_mask = np.asarray(roi_mask, dtype=bool)
        boundary_for_score = boundary & roi_mask
    else:
        boundary_for_score = boundary
    boundary_roi_px = int(np.count_nonzero(boundary_for_score))
    if body_px <= 0 or boundary_px <= 0:
        return float("inf"), {
            "body_px": body_px,
            "boundary_px": boundary_px,
            "boundary_roi_px": boundary_roi_px,
            "boundary_roi_fraction": 0.0,
            "kept_boundary_px": 0,
        }
    if boundary_roi_px <= 0:
        return float("inf"), {
            "body_px": body_px,
            "boundary_px": boundary_px,
            "boundary_roi_px": boundary_roi_px,
            "boundary_roi_fraction": 0.0,
            "kept_boundary_px": 0,
        }
    values = edge_distance[boundary_for_score].astype(np.float64)
    values = values[np.isfinite(values)]
    if values.shape[0] == 0:
        return float("inf"), {
            "body_px": body_px,
            "boundary_px": boundary_px,
            "boundary_roi_px": boundary_roi_px,
            "boundary_roi_fraction": float(boundary_roi_px / max(boundary_px, 1)),
            "kept_boundary_px": 0,
        }
    keep_fraction = min(1.0, max(0.05, float(trim_fraction)))
    keep_count = max(1, int(np.ceil(values.shape[0] * keep_fraction)))
    kept = np.sort(values)[:keep_count]
    return float(np.mean(kept)), {
        "body_px": body_px,
        "boundary_px": boundary_px,
        "boundary_roi_px": boundary_roi_px,
        "boundary_roi_fraction": float(boundary_roi_px / max(boundary_px, 1)),
        "kept_boundary_px": int(keep_count),
        "edge_distance_mean_px": float(np.mean(values)),
        "edge_distance_trimmed_mean_px": float(np.mean(kept)),
        "edge_distance_p50_px": float(np.percentile(values, 50.0)),
        "edge_distance_p90_px": float(np.percentile(values, 90.0)),
    }


def _silhouette_mask_score(
    body_mask: "np.ndarray",
    object_mask: "np.ndarray",
    precision_weight: float,
    recall_weight: float,
) -> Tuple[float, dict]:
    """Pose score against a real object mask (lower is better).

    ``precision`` is the fraction of the rendered silhouette inside the object
    mask (a wrong in-plane rotation pushes silhouette corners out of the mask);
    ``recall`` is the fraction of the object mask covered by the silhouette.
    Precision is weighted higher so extra mask regions (residual shadow, the
    moving part beyond the body) do not force the body onto them.
    """
    import numpy as np

    body_px = int(np.count_nonzero(body_mask))
    mask_px = int(np.count_nonzero(object_mask))
    if mask_px <= 0:
        return float("inf"), {"body_px": body_px, "object_mask_px": mask_px}
    if body_px <= 0:
        # Silhouette entirely outside the scored region (off-image, or fully on
        # untrusted background). Worst finite score: the camera must keep
        # penalizing such poses instead of dropping out of the candidate mean,
        # which would let a bad pose escape this camera's judgement.
        return float(precision_weight + recall_weight), {
            "body_px": body_px,
            "object_mask_px": mask_px,
            "mask_intersection_px": 0,
            "mask_precision": 0.0,
            "mask_recall": 0.0,
            "mask_iou": 0.0,
        }
    intersection = int(np.count_nonzero(body_mask & object_mask))
    precision = float(intersection / max(body_px, 1))
    recall = float(intersection / max(mask_px, 1))
    union = body_px + mask_px - intersection
    score = float(precision_weight * (1.0 - precision) + recall_weight * (1.0 - recall))
    return score, {
        "body_px": body_px,
        "object_mask_px": mask_px,
        "mask_intersection_px": intersection,
        "mask_precision": precision,
        "mask_recall": recall,
        "mask_iou": float(intersection / max(union, 1)),
    }


def _mask_bbox(mask: "np.ndarray") -> Optional[Tuple[int, int, int, int]]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if xs.shape[0] == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_metrics(
    body_mask: "np.ndarray",
    roi_bbox: Optional[Tuple[int, int, int, int]],
) -> dict:
    import numpy as np

    body_bbox = _mask_bbox(body_mask)
    if roi_bbox is None or body_bbox is None:
        return {
            "body_bbox": body_bbox,
            "roi_bbox": roi_bbox,
            "bbox_center_norm": 1.0,
            "bbox_size_norm": 1.0,
            "bbox_area_ratio": 0.0,
        }
    bx0, by0, bx1, by1 = body_bbox
    rx0, ry0, rx1, ry1 = roi_bbox
    bw = max(1.0, float(bx1 - bx0))
    bh = max(1.0, float(by1 - by0))
    rw = max(1.0, float(rx1 - rx0))
    rh = max(1.0, float(ry1 - ry0))
    body_center = np.asarray([(bx0 + bx1) * 0.5, (by0 + by1) * 0.5], dtype=np.float64)
    roi_center = np.asarray([(rx0 + rx1) * 0.5, (ry0 + ry1) * 0.5], dtype=np.float64)
    roi_diag = float(np.linalg.norm([rw, rh]))
    center_norm = float(np.linalg.norm(body_center - roi_center) / max(roi_diag, 1.0))
    size_norm = float((abs(np.log(bw / rw)) + abs(np.log(bh / rh))) * 0.5)
    area_ratio = float((bw * bh) / max(rw * rh, 1.0))
    return {
        "body_bbox": body_bbox,
        "roi_bbox": roi_bbox,
        "bbox_center_norm": center_norm,
        "bbox_size_norm": size_norm,
        "bbox_area_ratio": area_ratio,
    }


def _project_sparse_points_to_image_roi(
    points_world: "np.ndarray",
    cam: dict,
    image_shape: Tuple[int, int],
    args: argparse.Namespace,
) -> dict:
    import cv2
    import numpy as np

    height, width = int(image_shape[0]), int(image_shape[1])
    info = {
        "enabled": False,
        "source": "projected_sparse",
        "point_count": 0,
        "bbox": None,
        "mask": None,
        "warning": None,
    }
    if points_world is None or points_world.shape[0] == 0 or height <= 0 or width <= 0:
        info["warning"] = "no sparse points for image ROI."
        return info

    points = np.asarray(points_world, dtype=np.float64)
    pixels = _project_points(cam["projection"], points)
    depths = _depths_in_camera(cam["cam_from_world"], points)
    valid = (
        np.isfinite(pixels).all(axis=1)
        & np.isfinite(depths)
        & (depths > 0.0)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(height))
    )
    xy = pixels[valid]
    min_points = max(1, int(getattr(args, "registration_image_roi_min_points", 8)))
    if xy.shape[0] < min_points:
        info["warning"] = f"only {xy.shape[0]} projected sparse ROI points; {min_points}+ required."
        info["point_count"] = int(xy.shape[0])
        return info

    x0 = float(np.percentile(xy[:, 0], 2.0))
    y0 = float(np.percentile(xy[:, 1], 2.0))
    x1 = float(np.percentile(xy[:, 0], 98.0))
    y1 = float(np.percentile(xy[:, 1], 98.0))
    pad_fraction = max(0.0, float(getattr(args, "registration_image_roi_padding_fraction", 0.35)))
    span = max(x1 - x0, y1 - y0, 1.0)
    pad = pad_fraction * span
    x0 = int(max(0, np.floor(x0 - pad)))
    y0 = int(max(0, np.floor(y0 - pad)))
    x1 = int(min(width, np.ceil(x1 + pad)))
    y1 = int(min(height, np.ceil(y1 + pad)))
    if x1 <= x0 or y1 <= y0:
        info["warning"] = "projected sparse ROI bbox is empty."
        return info

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    point_mask = np.zeros((height, width), dtype=np.uint8)
    rounded = np.rint(xy).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    point_mask[rounded[:, 1], rounded[:, 0]] = 1
    dilate = max(0, int(getattr(args, "registration_image_roi_point_dilate", 11)))
    if dilate > 0:
        kernel_size = 2 * dilate + 1
        point_mask = cv2.dilate(point_mask, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
    mask = np.maximum(mask, point_mask)
    info.update(
        {
            "enabled": True,
            "point_count": int(xy.shape[0]),
            "bbox": (x0, y0, x1, y1),
            "mask": mask.astype(bool),
        }
    )
    return info


def _prepare_registration_image_contexts(
    bundle: Dict[str, dict],
    serials: List[str],
    sparse_points_world: Optional["np.ndarray"],
    args: argparse.Namespace,
    output_dir: Optional[str] = None,
    state_id: Optional[str] = None,
) -> Dict[str, dict]:
    import cv2
    import numpy as np

    contexts = {}
    roi_mode = getattr(args, "registration_image_roi_mode", "projected_sparse")
    score_mode = getattr(args, "registration_image_score_mode", "auto")
    mask_downscale = max(1, int(getattr(args, "registration_image_mask_downscale", 2)))
    for serial in serials:
        cam = bundle[serial]
        img = _cv2_imread(cam["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            int(getattr(args, "registration_image_canny_low", 50)),
            int(getattr(args, "registration_image_canny_high", 150)),
        )
        edge_distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
        roi = {"enabled": False, "bbox": None, "mask": None, "point_count": 0, "warning": None}
        if roi_mode == "projected_sparse":
            roi = _project_sparse_points_to_image_roi(sparse_points_world, cam, img.shape[:2], args)

        object_mask = None
        mask_small = None
        cam_small = None
        mask_shape = None
        trust = None
        trust_small = None
        if score_mode in ("auto", "mask"):
            mask_path = _find_object_mask_path(output_dir, state_id, serial)
            if mask_path is not None:
                mask_image = _cv2_imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_image is not None and mask_image.shape[:2] == img.shape[:2]:
                    object_mask = mask_image > 127
                    trust_image = None
                    if str(getattr(args, "mask_trust_mode", "auto")) != "off":
                        trust_path = _find_trust_mask_path(output_dir, serial)
                        if trust_path is not None:
                            trust_image = _cv2_imread(trust_path, cv2.IMREAD_GRAYSCALE)
                            if trust_image is not None and trust_image.shape[:2] != img.shape[:2]:
                                trust_image = None
                    if trust_image is not None:
                        trust = trust_image > 127
                    if mask_downscale > 1:
                        small_size = (
                            max(1, img.shape[1] // mask_downscale),
                            max(1, img.shape[0] // mask_downscale),
                        )
                        mask_small = (
                            cv2.resize(mask_image, small_size, interpolation=cv2.INTER_NEAREST) > 127
                        )
                        if trust_image is not None:
                            trust_small = (
                                cv2.resize(trust_image, small_size, interpolation=cv2.INTER_NEAREST) > 127
                            )
                        scale_x = small_size[0] / float(img.shape[1])
                        scale_y = small_size[1] / float(img.shape[0])
                        scale_T = np.array(
                            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
                            dtype=np.float64,
                        )
                        cam_small = dict(cam)
                        cam_small["projection"] = scale_T @ np.asarray(cam["projection"], dtype=np.float64)
                        mask_shape = (small_size[1], small_size[0])
                    else:
                        mask_small = object_mask
                        trust_small = trust
                        cam_small = cam
                        mask_shape = img.shape[:2]
                    if trust_small is not None and mask_small is not None:
                        mask_small = mask_small & trust_small
                        min_trusted_px = max(0, int(getattr(args, "registration_mask_min_trusted_px", 150)))
                        if int(np.count_nonzero(mask_small)) < min_trusted_px:
                            # The object barely overlaps the trusted (cloth) region in
                            # this view; the fragmentary remainder would mis-rank poses,
                            # so the camera is dropped from mask scoring entirely.
                            mask_small = None
                            cam_small = None
                            mask_shape = None
                            trust_small = None

        contexts[serial] = {
            "serial": serial,
            "cam": cam,
            "image": img,
            "edge_distance": edge_distance,
            "roi": roi,
            "object_mask": object_mask,
            "mask_small": mask_small,
            "cam_small": cam_small,
            "mask_shape": mask_shape,
            "trust": trust,
            "trust_small": trust_small,
        }
    return contexts


def _registration_image_refinement_candidates(
    vertices: "np.ndarray",
    args: argparse.Namespace,
) -> List[dict]:
    import numpy as np

    centroid = np.mean(vertices, axis=0)
    mesh_diag = _mesh_diagonal_from_vertices(vertices)
    cov = np.cov((vertices - centroid).T)
    _eigvals, eigvecs = np.linalg.eigh(cov)
    axes = {
        "normal": eigvecs[:, 0],
        "wide": eigvecs[:, 2],
        "mid": eigvecs[:, 1],
    }
    rotation_candidates = [
        {
            "name": "base",
            "object_T_delta": np.eye(4, dtype=np.float64),
            "axis": "none",
            "angle_deg": 0.0,
        }
    ]

    step = int(getattr(args, "registration_image_normal_step_deg", 0))
    if step > 0:
        for angle_deg in range(step, 360, step):
            rotation_candidates.append(
                {
                    "name": f"normal_{angle_deg:03d}",
                    "object_T_delta": _object_space_rotation_transform(
                        centroid, axes["normal"], np.deg2rad(float(angle_deg))
                    ),
                    "axis": "normal",
                    "angle_deg": float(angle_deg),
                }
            )

    tilt_deg = float(getattr(args, "registration_image_tilt_deg", 0.0))
    tilt_multiples = max(1, int(getattr(args, "registration_image_tilt_multiples", 1)))
    if tilt_deg > 0.0:
        for axis_name in ("wide", "mid", "normal"):
            for multiple in range(1, tilt_multiples + 1):
                for sign in (-1.0, 1.0):
                    angle_deg = sign * tilt_deg * multiple
                    rotation_candidates.append(
                        {
                            "name": f"{axis_name}_{angle_deg:+.1f}",
                            "object_T_delta": _object_space_rotation_transform(
                                centroid, axes[axis_name], np.deg2rad(angle_deg)
                            ),
                            "axis": axis_name,
                            "angle_deg": float(angle_deg),
                        }
                    )

    if bool(getattr(args, "registration_image_frontback_flips", True)):
        for axis_name in ("wide", "mid"):
            rotation_candidates.append(
                {
                    "name": f"{axis_name}_flip_180",
                    "object_T_delta": _object_space_rotation_transform(
                        centroid, axes[axis_name], np.deg2rad(180.0)
                    ),
                    "axis": axis_name,
                    "angle_deg": 180.0,
                }
            )

    scale_span = max(0.0, float(getattr(args, "registration_image_scale_sweep", 0.0)))
    if scale_span > 0.0:
        scale_values = [max(0.05, 1.0 - scale_span), 1.0, 1.0 + scale_span]
    else:
        scale_values = [1.0]

    translation_fraction = max(0.0, float(getattr(args, "registration_image_translation_fraction", 0.0)))
    translation_steps = max(0, int(getattr(args, "registration_image_translation_steps", 1)))
    if translation_fraction > 0.0 and translation_steps > 0:
        span = mesh_diag * translation_fraction
        values = np.linspace(-span, span, 2 * translation_steps + 1)
    else:
        values = np.asarray([0.0], dtype=np.float64)
    translation_offsets = [
        np.asarray([dx, dy, dz], dtype=np.float64)
        for dx in values
        for dy in values
        for dz in values
    ]

    candidates = []
    for base in rotation_candidates:
        for scale in scale_values:
            scale_T = _object_space_scale_transform(centroid, scale)
            for offset in translation_offsets:
                translate_T = _object_space_translation_transform(offset)
                delta = translate_T @ scale_T @ base["object_T_delta"]
                suffix = []
                if abs(float(scale) - 1.0) > 1.0e-9:
                    suffix.append(f"s{float(scale):.3f}")
                if float(np.linalg.norm(offset)) > 1.0e-12:
                    suffix.append(
                        "t{:+.3f}_{:+.3f}_{:+.3f}".format(
                            float(offset[0] / max(mesh_diag, 1.0e-12)),
                            float(offset[1] / max(mesh_diag, 1.0e-12)),
                            float(offset[2] / max(mesh_diag, 1.0e-12)),
                        )
                    )
                name = base["name"] if not suffix else f"{base['name']}:{':'.join(suffix)}"
                candidates.append(
                    {
                        "name": name,
                        "object_T_delta": delta,
                        "axis": base["axis"],
                        "angle_deg": base["angle_deg"],
                        "scale": float(scale),
                        "translation_object": offset.tolist(),
                        "translation_fraction": (
                            (offset / max(mesh_diag, 1.0e-12)).astype(float).tolist()
                        ),
                    }
                )

    return candidates


def _score_registration_image_pose(
    vertices: "np.ndarray",
    faces: "np.ndarray",
    world_T_object: "np.ndarray",
    contexts: Dict[str, dict],
    serials: List[str],
    args: argparse.Namespace,
    channel: str = "edge",
) -> Tuple[float, List[dict]]:
    import numpy as np

    trim_fraction = float(getattr(args, "registration_image_edge_trim_fraction", 0.70))
    bbox_weight = max(0.0, float(getattr(args, "registration_image_bbox_weight", 12.0)))
    min_bbox_area_ratio = max(0.0, float(getattr(args, "registration_image_min_bbox_area_ratio", 0.25)))
    max_bbox_area_ratio = max(min_bbox_area_ratio, float(getattr(args, "registration_image_max_bbox_area_ratio", 3.0)))
    precision_weight = max(0.0, float(getattr(args, "registration_mask_precision_weight", 1.0)))
    recall_weight = max(0.0, float(getattr(args, "registration_mask_recall_weight", 0.5)))
    camera_rows = []
    scores = []
    for serial in serials:
        context = contexts.get(serial)
        if not context:
            continue
        img = context["image"]
        if channel == "mask" and context.get("mask_small") is not None:
            body = _render_mesh_silhouette_mask(
                vertices, faces, world_T_object, context["cam_small"], context["mask_shape"], 0
            )
            trust_small = context.get("trust_small")
            if trust_small is not None:
                # Silhouette pixels over untrusted background are unknowable in the
                # mask, so they must not count against precision.
                body = body & trust_small
            score, info = _silhouette_mask_score(
                body, context["mask_small"], precision_weight, recall_weight
            )
            row = {
                "serial": serial,
                "score": score,
                "score_channel": "mask",
                "trust_gated": bool(trust_small is not None),
                "edge_score": float("nan"),
                "bbox_penalty": 0.0,
                "roi_enabled": False,
                "roi_point_count": 0,
                "roi_bbox": None,
            }
            row.update(info)
            camera_rows.append(row)
            if np.isfinite(score):
                scores.append(score)
            continue
        roi = context.get("roi") or {}
        roi_mask = roi.get("mask") if roi.get("enabled") else None
        body = _render_mesh_silhouette_mask(vertices, faces, world_T_object, context["cam"], img.shape[:2], 0)
        edge_score, info = _silhouette_edge_score(body, context["edge_distance"], trim_fraction, roi_mask=roi_mask)
        bbox_info = _bbox_metrics(body, roi.get("bbox") if roi.get("enabled") else None)
        bbox_penalty = 0.0
        if roi.get("enabled"):
            bbox_penalty = (
                float(bbox_info["bbox_center_norm"])
                + float(bbox_info["bbox_size_norm"])
                + 0.5 * max(0.0, min_bbox_area_ratio - float(bbox_info["bbox_area_ratio"]))
                + 0.25 * max(0.0, float(bbox_info["bbox_area_ratio"]) - max_bbox_area_ratio)
            )
        score = float(edge_score + bbox_weight * bbox_penalty) if np.isfinite(edge_score) else float("inf")
        row = {
            "serial": serial,
            "score": score,
            "score_channel": "edge",
            "edge_score": edge_score,
            "bbox_penalty": bbox_penalty,
            "roi_enabled": bool(roi.get("enabled")),
            "roi_point_count": int(roi.get("point_count", 0) or 0),
            "roi_bbox": roi.get("bbox"),
        }
        row.update(info)
        row.update(bbox_info)
        camera_rows.append(row)
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        return float("inf"), camera_rows
    return float(np.mean(scores)), camera_rows


def _write_registration_image_refine_overlays(
    out_dir: str,
    vertices: "np.ndarray",
    faces: "np.ndarray",
    base_T: "np.ndarray",
    best_T: "np.ndarray",
    contexts: Dict[str, dict],
    serials: List[str],
) -> List[dict]:
    import cv2
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for serial in serials:
        context = contexts.get(serial)
        if not context:
            continue
        img = context["image"]
        base = _render_mesh_silhouette_mask(vertices, faces, base_T, context["cam"], img.shape[:2], 0)
        best = _render_mesh_silhouette_mask(vertices, faces, best_T, context["cam"], img.shape[:2], 0)

        overlay = img.copy()
        trust = context.get("trust")
        if trust is not None and trust.shape[:2] == overlay.shape[:2]:
            # Show the trust gate: darkened pixels are "unknown" background where
            # mask evidence is neither trusted nor scored.
            untrusted = ~trust
            overlay[untrusted] = (0.55 * overlay[untrusted]).astype(np.uint8)
        base_boundary = _silhouette_boundary(base)
        best_boundary = _silhouette_boundary(best)
        overlay[base] = (0.75 * overlay[base] + 0.25 * np.array([0, 160, 255], dtype=np.float64)).astype(np.uint8)
        overlay[best] = (0.75 * overlay[best] + 0.25 * np.array([0, 220, 0], dtype=np.float64)).astype(np.uint8)
        overlay[base_boundary] = np.array([0, 120, 255], dtype=np.uint8)
        overlay[best_boundary] = np.array([0, 255, 0], dtype=np.uint8)
        object_mask = context.get("object_mask")
        if object_mask is not None:
            mask_boundary = cv2.dilate(
                _silhouette_boundary(object_mask).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
            ).astype(bool)
            overlay[mask_boundary] = np.array([0, 255, 255], dtype=np.uint8)
        roi = context.get("roi") or {}
        if roi.get("enabled") and roi.get("bbox") is not None:
            x0, y0, x1, y1 = roi["bbox"]
            cv2.rectangle(overlay, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), (255, 0, 255), 2)
        panel = np.hstack([img, overlay])
        path = os.path.join(out_dir, f"image_refine_{serial}.png")
        _cv2_imwrite(path, panel)
        rows.append(
            {
                "serial": serial,
                "overlay_path": path,
                "base_body_px": int(np.count_nonzero(base)),
                "best_body_px": int(np.count_nonzero(best)),
                "roi_enabled": bool(roi.get("enabled")),
                "roi_point_count": int(roi.get("point_count", 0) or 0),
                "roi_bbox": roi.get("bbox"),
            }
        )
    return rows


def _refine_registration_with_image_edges(
    report: dict,
    reg_dir: str,
    vertices: "np.ndarray",
    faces: "np.ndarray",
    initial_T: "np.ndarray",
    sparse_points_world: Optional["np.ndarray"],
    args: argparse.Namespace,
    output_dir: Optional[str] = None,
) -> dict:
    import numpy as np

    summary = {
        "method": "single_frame_silhouette_edge_search",
        "enabled": False,
        "accepted": False,
        "initial_score": None,
        "best_score": None,
        "best_candidate": None,
        "candidate_count": 0,
        "warnings": [],
    }
    if not bool(getattr(args, "registration_image_refine", False)):
        return summary
    summary["enabled"] = True

    bundle, warnings = _load_state_camera_bundle(report["path"], report.get("undistorted_cameras", []))
    summary["warnings"].extend(warnings)
    serials = sorted(bundle.keys())
    max_cameras = int(getattr(args, "registration_image_max_cameras", 8))
    if max_cameras > 0:
        serials = serials[:max_cameras]
    if not serials:
        summary["warnings"].append("no usable cameras for image registration refinement.")
        return summary

    contexts = _prepare_registration_image_contexts(
        bundle,
        serials,
        sparse_points_world,
        args,
        output_dir=output_dir,
        state_id=str(report.get("state_id")),
    )
    serials = [serial for serial in serials if serial in contexts]
    if not serials:
        summary["warnings"].append("no images could be loaded for image registration refinement.")
        return summary

    score_mode = getattr(args, "registration_image_score_mode", "auto")
    mask_serials = [serial for serial in serials if contexts[serial].get("mask_small") is not None]
    if score_mode == "mask" and not mask_serials:
        summary["warnings"].append(
            "score mode 'mask' requested but no object masks found; falling back to edge scoring."
        )
    if score_mode in ("auto", "mask") and mask_serials:
        channel = "mask"
        serials = mask_serials
    else:
        channel = "edge"
    summary["score_channel"] = channel
    summary["mask_camera_count"] = len(mask_serials)
    roi_reports = [
        {
            "serial": serial,
            "enabled": bool((contexts[serial].get("roi") or {}).get("enabled")),
            "point_count": int((contexts[serial].get("roi") or {}).get("point_count", 0) or 0),
            "bbox": (contexts[serial].get("roi") or {}).get("bbox"),
            "warning": (contexts[serial].get("roi") or {}).get("warning"),
        }
        for serial in serials
    ]
    summary["image_roi"] = {
        "mode": getattr(args, "registration_image_roi_mode", "projected_sparse"),
        "camera_count": len(serials),
        "enabled_camera_count": int(sum(1 for row in roi_reports if row["enabled"])),
        "reports": roi_reports,
    }

    if channel == "mask":
        summary["method"] = "single_frame_silhouette_mask_search"

    candidates = _registration_image_refinement_candidates(vertices, args)
    rounds = max(1, int(getattr(args, "registration_image_refine_rounds", 1)))
    round_ratio = float(getattr(args, "registration_image_round_acceptance_ratio", 0.999))
    summary["candidate_count"] = len(candidates)
    summary["rounds"] = rounds
    print(
        f"[IMGREG] {report.get('state_id')}: channel={channel} cameras={len(serials)} "
        f"candidates={len(candidates)} rounds={rounds}"
    )

    # Greedy coordinate-descent polish: evaluate the candidate set around the current
    # pose, apply the best improving delta, and repeat. Successive rounds compose
    # deltas about different axes, so an initial error about an arbitrary axis (which
    # no single candidate covers) can still be corrected.
    rows = []
    base_score = None
    base_camera_rows = []
    current_T = np.asarray(initial_T, dtype=np.float64)
    current_score = None
    current_camera_rows = []
    applied_names = []
    for round_index in range(rounds):
        round_best = None
        for idx, candidate in enumerate(candidates):
            T = current_T @ candidate["object_T_delta"]
            score, camera_rows = _score_registration_image_pose(
                vertices, faces, T, contexts, serials, args, channel=channel
            )
            edge_scores = [row["edge_score"] for row in camera_rows if np.isfinite(row.get("edge_score", float("inf")))]
            bbox_penalties = [row["bbox_penalty"] for row in camera_rows if np.isfinite(row.get("bbox_penalty", float("inf")))]
            boundary_roi_fractions = [
                row["boundary_roi_fraction"]
                for row in camera_rows
                if np.isfinite(row.get("boundary_roi_fraction", float("inf")))
            ]
            row = {
                "round": round_index,
                "index": idx,
                "name": candidate["name"],
                "axis": candidate["axis"],
                "angle_deg": candidate["angle_deg"],
                "scale": candidate.get("scale", 1.0),
                "translation_x": (candidate.get("translation_object") or [0.0, 0.0, 0.0])[0],
                "translation_y": (candidate.get("translation_object") or [0.0, 0.0, 0.0])[1],
                "translation_z": (candidate.get("translation_object") or [0.0, 0.0, 0.0])[2],
                "score_channel": channel,
                "score": score,
                "edge_score": float(np.mean(edge_scores)) if edge_scores else float("inf"),
                "bbox_penalty": float(np.mean(bbox_penalties)) if bbox_penalties else float("inf"),
                "boundary_roi_fraction": (
                    float(np.mean(boundary_roi_fractions)) if boundary_roi_fractions else 0.0
                ),
                "camera_count": len(camera_rows),
            }
            rows.append(row)
            if round_index == 0 and candidate["name"] == "base":
                base_score = score
                base_camera_rows = camera_rows
            if round_best is None or score < round_best["score"]:
                round_best = {
                    "index": idx,
                    "candidate": candidate,
                    "score": score,
                    "camera_rows": camera_rows,
                    "T_world_object": T,
                }
        if round_best is None:
            break
        if current_score is None:
            current_score = base_score if base_score is not None else float("inf")
            current_camera_rows = base_camera_rows
        improved = (
            np.isfinite(round_best["score"])
            and round_best["score"] < current_score * round_ratio
            and round_best["candidate"]["name"] != "base"
        )
        if not improved:
            break
        current_T = round_best["T_world_object"]
        current_score = round_best["score"]
        current_camera_rows = round_best["camera_rows"]
        applied_names.append(round_best["candidate"]["name"])
        print(
            f"[IMGREG] {report.get('state_id')}: round {round_index} applied "
            f"{round_best['candidate']['name']} score={round_best['score']:.5f}"
        )

    out_dir = os.path.join(reg_dir, "image_refine")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "candidates.csv")
    _write_csv_rows(
        csv_path,
        [
            "round",
            "index",
            "name",
            "axis",
            "angle_deg",
            "scale",
            "translation_x",
            "translation_y",
            "translation_z",
            "score_channel",
            "score",
            "edge_score",
            "bbox_penalty",
            "boundary_roi_fraction",
            "camera_count",
        ],
        rows,
    )
    summary["candidates_csv"] = csv_path

    if base_score is None:
        base_score = float("inf")
    summary["initial_score"] = base_score
    if current_score is None:
        summary["warnings"].append("no image refinement candidate could be scored.")
        return summary
    summary["best_score"] = current_score
    summary["initial_camera_reports"] = base_camera_rows
    summary["best_camera_reports"] = current_camera_rows
    summary["applied_rounds"] = applied_names
    summary["best_candidate"] = {
        "name": "+".join(applied_names) if applied_names else "base",
        "applied_rounds": applied_names,
    }

    overlay_rows = _write_registration_image_refine_overlays(
        out_dir,
        vertices,
        faces,
        initial_T,
        current_T,
        contexts,
        serials[: min(len(serials), int(getattr(args, "registration_image_overlay_cameras", 6)))],
    )
    summary["overlays"] = overlay_rows

    ratio = float(getattr(args, "registration_image_acceptance_ratio", 0.90))
    finite = np.isfinite(base_score) and np.isfinite(current_score)
    accepted = bool(finite and current_score <= base_score * ratio and applied_names)
    summary["accepted"] = accepted
    if accepted:
        summary["T_world_object"] = current_T.tolist()
    else:
        summary["warnings"].append("image refinement did not improve edge score enough to replace registration pose.")
    return summary


def _fibonacci_directions(count: int) -> "np.ndarray":
    """~Evenly distributed unit vectors on the sphere (spiral / Fibonacci lattice)."""
    import numpy as np

    count = max(1, int(count))
    idx = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * idx / count
    phi = np.arccos(np.clip(z, -1.0, 1.0))
    golden = np.pi * (1.0 + 5.0 ** 0.5)
    theta = golden * idx
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    return np.stack([x, y, np.cos(phi)], axis=1)


def _align_axis_rotation(source_axis: "np.ndarray", target_axis: "np.ndarray") -> "np.ndarray":
    """Minimal 3x3 rotation mapping unit ``source_axis`` onto unit ``target_axis``."""
    import numpy as np

    a = np.asarray(source_axis, dtype=np.float64)
    b = np.asarray(target_axis, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1.0e-12 or nb < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    a = a / na
    b = b / nb
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1.0e-9:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1.0e-9:
        perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, perp)
        return _rotation_about_axis(axis, np.pi)
    return _rotation_about_axis(np.cross(a, b), float(np.arccos(c)))


def _pixel_ray_world(cam: dict, pixel: Tuple[float, float]) -> Optional[Tuple["np.ndarray", "np.ndarray"]]:
    """Back-project an image pixel to a world-space ray (origin, unit direction)."""
    import numpy as np

    K = np.asarray(cam["K"], dtype=np.float64)
    cam_from_world = np.asarray(cam["cam_from_world"], dtype=np.float64)
    rotation = cam_from_world[:, :3]
    try:
        d_cam = np.linalg.inv(K) @ np.array([float(pixel[0]), float(pixel[1]), 1.0], dtype=np.float64)
    except np.linalg.LinAlgError:
        return None
    d_world = rotation.T @ d_cam
    norm = float(np.linalg.norm(d_world))
    if norm < 1.0e-12:
        return None
    origin = np.asarray(cam["camera_center_world"], dtype=np.float64).reshape(3)
    return origin, d_world / norm


def _robust_ray_intersection(
    origins: "np.ndarray",
    directions: "np.ndarray",
    iterations: int = 4,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """IRLS closest point to a bundle of rays (robust to a minority of stray rays)."""
    import numpy as np

    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    count = origins.shape[0]
    weights = np.ones(count, dtype=np.float64)
    point = np.mean(origins, axis=0)
    resid = np.zeros(count, dtype=np.float64)
    projectors = [np.eye(3, dtype=np.float64) - np.outer(directions[i], directions[i]) for i in range(count)]
    for _ in range(max(1, int(iterations))):
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        for i in range(count):
            A += weights[i] * projectors[i]
            b += weights[i] * (projectors[i] @ origins[i])
        try:
            point = np.linalg.solve(A + 1.0e-9 * np.eye(3), b)
        except np.linalg.LinAlgError:
            break
        for i in range(count):
            resid[i] = float(np.linalg.norm(projectors[i] @ (point - origins[i])))
        scale = float(np.median(resid)) + 1.0e-9
        weights = 1.0 / (1.0 + (resid / scale) ** 2)
    inliers = resid <= 3.0 * (float(np.median(resid)) + 1.0e-9)
    return point, resid, inliers


def _scaled_mask_and_cam(object_mask: "np.ndarray", cam: dict, downscale: int) -> Tuple["np.ndarray", dict, Tuple[int, int]]:
    """Downscale a mask and the matching camera projection for cheap silhouette scoring."""
    import cv2
    import numpy as np

    height, width = int(object_mask.shape[0]), int(object_mask.shape[1])
    ds = max(1, int(downscale))
    small_w = max(1, width // ds)
    small_h = max(1, height // ds)
    if (small_w, small_h) == (width, height):
        return object_mask.astype(bool), cam, (height, width)
    small = cv2.resize(object_mask.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST) > 0
    scale = np.array(
        [[small_w / float(width), 0.0, 0.0], [0.0, small_h / float(height), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cam_small = dict(cam)
    cam_small["projection"] = scale @ np.asarray(cam["projection"], dtype=np.float64)
    return small, cam_small, (small_h, small_w)


def _maybe_decimate_faces(vertices: "np.ndarray", faces: Optional["np.ndarray"], max_faces: int) -> Tuple["np.ndarray", Optional["np.ndarray"]]:
    """Cap the triangle count used for silhouette rasterization (bounds search cost)."""
    import numpy as np

    if faces is None or max_faces <= 0 or faces.shape[0] <= int(max_faces):
        return vertices, faces
    rng = np.random.default_rng(12345)
    keep = np.sort(rng.choice(faces.shape[0], size=int(max_faces), replace=False))
    return vertices, faces[keep]


def _multiview_mask_pose_cost(
    vertices: "np.ndarray",
    faces: Optional["np.ndarray"],
    world_T_object: "np.ndarray",
    scaled_views: List[Tuple["np.ndarray", dict, Tuple[int, int]]],
    precision_weight: float,
    recall_weight: float,
    trim_fraction: float,
) -> Tuple[float, int]:
    """Trimmed-mean silhouette-vs-mask cost over views (lower is better).

    Dropping the worst ``trim_fraction`` of views makes the cost tolerant of a
    minority of false-positive/false-negative masks and of the moving articulated
    part, whose view disagrees with the rest-pose mesh silhouette.
    """
    import numpy as np

    scores = []
    for mask_small, cam_small, shape in scaled_views:
        body = _render_mesh_silhouette_mask(vertices, faces, world_T_object, cam_small, shape, 0)
        score, _info = _silhouette_mask_score(body, mask_small, precision_weight, recall_weight)
        if np.isfinite(score):
            scores.append(float(score))
    if not scores:
        return float("inf"), 0
    scores.sort()
    trim = max(0.0, min(0.9, float(trim_fraction)))
    keep = max(1, int(round(len(scores) * (1.0 - trim))))
    return float(np.mean(scores[:keep])), len(scores)


def _build_chamfer_view(object_mask: "np.ndarray", cam: dict, downscale: int) -> dict:
    """Precompute the per-view data for the distance-transform silhouette cost.

    The mask distance field ``dt_out`` (distance from every outside pixel to the
    nearest mask pixel; zero inside) is computed once per view. Candidate poses are
    then scored by projecting a fixed set of mesh surface points and looking up
    this field, so no mesh is rasterized per candidate -- the dominant cost of the
    old fill-poly scorer.
    """
    import cv2
    import numpy as np

    mask_small, cam_small, shape = _scaled_mask_and_cam(object_mask, cam, downscale)
    mask_bool = np.asarray(mask_small, dtype=bool)
    inv = (~mask_bool).astype(np.uint8)
    dt_out = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float64)
    mask_area = int(np.count_nonzero(mask_bool))
    # ``dist_scale`` (~ object radius) only smooths the outside-distance term so the
    # coarse grid / refine have a convergence basin from far away; the strong
    # anti-spill penalty is the scale-free fraction-outside term in the cost.
    return {
        "mask": mask_bool,
        "cam_small": cam_small,
        "shape": (int(shape[0]), int(shape[1])),
        "dt_out": dt_out,
        "mask_area": mask_area,
        "dist_scale": float(np.sqrt(max(mask_area, 1))) + 1.0e-9,
    }


def _chamfer_view_cost(
    world_points: "np.ndarray",
    view: dict,
    precision_weight: float,
    recall_weight: float,
) -> float:
    """Distance-transform silhouette cost for one view (lower is better).

    ``precision`` penalizes projected mesh points that fall outside the mask. It has
    a scale-free **fraction outside** term (the strong anti-spill penalty -- a wrong
    orientation or an inflated pose pushes points out of the mask, and this cannot be
    cheapened by the recall term) plus a small **smooth distance** term so the coarse
    grid and refine still have a convergence basin when points are far outside.
    ``recall`` penalizes uncovered mask area by splatting the projected points into an
    occupancy grid. All derived from a single point projection plus lookups; no
    triangle is rasterized.
    """
    import cv2
    import numpy as np

    cam = view["cam_small"]
    height, width = view["shape"]
    mask = view["mask"]
    dt_out = view["dt_out"]
    mask_area = int(view["mask_area"])
    dist_scale = float(view["dist_scale"])
    if mask_area <= 0:
        return float("inf")

    depths = _depths_in_camera(cam["cam_from_world"], world_points)
    front = np.isfinite(depths) & (depths > 0.0)
    if not np.any(front):
        return float(precision_weight + recall_weight)
    pixels = _project_points(cam["projection"], world_points[front])
    finite = np.isfinite(pixels).all(axis=1)
    pixels = pixels[finite]
    if pixels.shape[0] == 0:
        return float(precision_weight + recall_weight)
    xi = np.rint(pixels[:, 0]).astype(np.int64)
    yi = np.rint(pixels[:, 1]).astype(np.int64)
    inb = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

    # Precision: fraction of points outside the mask (scale-free, strong anti-spill)
    # plus a small smooth distance term for a convergence basin. A boundary tolerance
    # keeps points within ~1.5 px of the mask edge counted as inside, so the
    # point-projection-vs-rasterized-mask discretization mismatch does not create a
    # noise floor for the correct pose. Off-image points count as outside at max dist.
    boundary_tol = 1.5
    inside = np.zeros(pixels.shape[0], dtype=bool)
    dist_norm = np.ones(pixels.shape[0], dtype=np.float64)
    if np.any(inb):
        d = dt_out[yi[inb], xi[inb]]
        inside[inb] = d <= boundary_tol
        dist_norm[inb] = np.minimum(np.maximum(d - boundary_tol, 0.0) / dist_scale, 1.0)
    frac_outside = float(1.0 - inside.mean())
    smooth_outside = float(dist_norm.mean())
    precision_penalty = frac_outside + 0.3 * smooth_outside

    recall = 0.0
    if np.any(inb):
        occ = np.zeros((height, width), dtype=np.uint8)
        occ[yi[inb], xi[inb]] = 1
        occ = cv2.dilate(occ, np.ones((3, 3), dtype=np.uint8), iterations=1)
        covered = int(np.count_nonzero((occ > 0) & mask))
        recall = covered / float(mask_area)
    recall_penalty = 1.0 - recall
    return float(precision_weight * precision_penalty + recall_weight * recall_penalty)


def _multiview_chamfer_cost(
    points_obj: "np.ndarray",
    world_T_object: "np.ndarray",
    chamfer_views: List[dict],
    precision_weight: float,
    recall_weight: float,
    trim_fraction: float,
) -> Tuple[float, int]:
    """Trimmed-mean distance-transform silhouette cost over views (lower is better).

    Drop-in replacement for ``_multiview_mask_pose_cost`` that avoids per-candidate
    rasterization: the mesh surface points are transformed once, then each view is
    scored by cheap projection + distance-field lookups. Trimming the worst
    ``trim_fraction`` of views keeps the cost tolerant of a minority of FP/FN masks
    and of the moving articulated part.
    """
    import numpy as np

    world_points = _apply_transform(np.asarray(points_obj, dtype=np.float64), world_T_object)
    scores = []
    for view in chamfer_views:
        cost = _chamfer_view_cost(world_points, view, precision_weight, recall_weight)
        if np.isfinite(cost):
            scores.append(float(cost))
    if not scores:
        return float("inf"), 0
    scores.sort()
    trim = max(0.0, min(0.9, float(trim_fraction)))
    keep = max(1, int(round(len(scores) * (1.0 - trim))))
    return float(np.mean(scores[:keep])), len(scores)


def _register_mask_silhouette(
    report: dict,
    output_dir: Optional[str],
    reg_dir: str,
    vertices: "np.ndarray",
    faces: Optional["np.ndarray"],
    mesh_points: "np.ndarray",
    args: argparse.Namespace,
) -> Optional[dict]:
    """Fit a known mesh directly to multiview object masks (analysis-by-synthesis).

    Rather than reconstruct a visual hull and align the mesh to it, this fits the
    mesh to the 2D masks: translation from a robust intersection of the per-view
    mask-centroid rays, orientation from a coarse SO(3) grid scored by a
    trimmed-mean multiview silhouette-vs-mask cost, then a local greedy refine of
    rotation and translation. Scale is fixed (the mesh is metric). Returns None
    when too few usable masks exist so the caller can fall back to the hull path.
    """
    import numpy as np

    state_id = str(report.get("state_id"))
    bundle, warnings = _load_state_camera_bundle(report["path"], report.get("undistorted_cameras", []))
    if not bundle:
        return None
    serials_all = sorted(bundle.keys())
    contexts = _prepare_registration_image_contexts(
        bundle, serials_all, None, args, output_dir=output_dir, state_id=state_id
    )
    mask_serials = [s for s in serials_all if contexts.get(s, {}).get("object_mask") is not None]
    min_masks = max(2, int(getattr(args, "registration_silhouette_min_cameras", 3)))
    if len(mask_serials) < min_masks:
        return None

    precision_weight = max(0.0, float(getattr(args, "registration_mask_precision_weight", 1.0)))
    recall_weight = max(0.0, float(getattr(args, "registration_mask_recall_weight", 0.5)))
    trim_fraction = float(getattr(args, "registration_silhouette_trim_fraction", 0.3))
    score_budget = max(64, int(getattr(args, "registration_silhouette_score_points", 2000)))
    scoring_points = _subsample_points(mesh_points, score_budget)

    origins: List["np.ndarray"] = []
    directions: List["np.ndarray"] = []
    for serial in mask_serials:
        mask = contexts[serial]["object_mask"]
        ys, xs = np.nonzero(mask)
        if xs.shape[0] < 1:
            continue
        ray = _pixel_ray_world(contexts[serial]["cam"], (float(xs.mean()), float(ys.mean())))
        if ray is None:
            continue
        origins.append(ray[0])
        directions.append(ray[1])
    if len(origins) < min_masks:
        return None
    translation, _resid, ray_inliers = _robust_ray_intersection(np.asarray(origins), np.asarray(directions))

    # Keep the vertex-mean centroid (identical to the previous scorer) so the
    # translation placement is unchanged from before this round -- only scoring changed.
    mesh_centroid = np.mean(vertices, axis=0)
    mesh_diag = float(np.linalg.norm(np.max(mesh_points, axis=0) - np.min(mesh_points, axis=0)))
    _centroid, mesh_axes, _extents = _pca_axes(mesh_points)
    normal_axis = mesh_axes[:, 2]

    def _make_T(rotation: "np.ndarray") -> "np.ndarray":
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation - rotation @ mesh_centroid
        return transform

    coarse_cameras = int(getattr(args, "registration_silhouette_coarse_cameras", 8))
    coarse_serials = mask_serials if coarse_cameras <= 0 else mask_serials[:coarse_cameras]
    coarse_downscale = int(getattr(args, "registration_silhouette_coarse_downscale", 4))
    coarse_views = [
        _build_chamfer_view(contexts[s]["object_mask"], contexts[s]["cam"], coarse_downscale) for s in coarse_serials
    ]

    normal_dirs = int(getattr(args, "registration_silhouette_normal_dirs", 48))
    inplane_steps = max(1, int(getattr(args, "registration_silhouette_inplane_steps", 12)))
    direction_grid = _fibonacci_directions(normal_dirs)
    inplane_angles = [2.0 * np.pi * k / inplane_steps for k in range(inplane_steps)]

    scored: List[Tuple[float, "np.ndarray"]] = []
    for world_dir in direction_grid:
        align = _align_axis_rotation(normal_axis, world_dir)
        for phi in inplane_angles:
            rotation = _rotation_about_axis(world_dir, phi) @ align
            transform = _make_T(rotation)
            cost, _n = _multiview_chamfer_cost(
                scoring_points, transform, coarse_views, precision_weight, recall_weight, trim_fraction
            )
            scored.append((cost, transform))
    scored.sort(key=lambda item: item[0])
    if not scored or not np.isfinite(scored[0][0]):
        return None
    coarse_best_T = scored[0][1]
    seed_count = max(1, int(getattr(args, "registration_silhouette_refine_seeds", 2)))
    seeds = [transform for _cost, transform in scored[:seed_count]]

    refine_cameras = int(getattr(args, "registration_silhouette_refine_cameras", 12))
    refine_serials = mask_serials if refine_cameras <= 0 else mask_serials[:refine_cameras]
    refine_downscale = max(1, int(getattr(args, "registration_image_mask_downscale", 2)))
    refine_views = [
        _build_chamfer_view(contexts[s]["object_mask"], contexts[s]["cam"], refine_downscale) for s in refine_serials
    ]
    max_rounds = max(0, int(getattr(args, "registration_silhouette_refine_rounds", 12)))
    # The chamfer cost is smooth and cheap, so a finer step tail can be walked to
    # close the last few degrees / few millimetres without a rasterization budget.
    rotation_steps_deg = (10.0, 4.0, 1.5)
    translation_steps = (0.04, 0.015, 0.005)
    world_axes = np.eye(3, dtype=np.float64)

    def _refine(seed_T: "np.ndarray") -> Tuple["np.ndarray", float]:
        current = np.asarray(seed_T, dtype=np.float64).copy()
        best_cost, _n = _multiview_chamfer_cost(
            scoring_points, current, refine_views, precision_weight, recall_weight, trim_fraction
        )
        for _round in range(max_rounds):
            center = current[:3, 3].copy()
            best_delta = None
            best_delta_cost = best_cost
            for axis_index in range(3):
                for step in rotation_steps_deg:
                    for sign in (1.0, -1.0):
                        rot = _rotation_about_axis(world_axes[axis_index], np.deg2rad(sign * step))
                        delta = np.eye(4, dtype=np.float64)
                        delta[:3, :3] = rot
                        delta[:3, 3] = center - rot @ center
                        candidate = delta @ current
                        cost, _n = _multiview_chamfer_cost(
                            scoring_points, candidate, refine_views, precision_weight, recall_weight, trim_fraction
                        )
                        if cost < best_delta_cost:
                            best_delta_cost = cost
                            best_delta = candidate
            for axis_index in range(3):
                for step in translation_steps:
                    for sign in (1.0, -1.0):
                        delta = np.eye(4, dtype=np.float64)
                        delta[:3, 3] = world_axes[axis_index] * (sign * step * mesh_diag)
                        candidate = delta @ current
                        cost, _n = _multiview_chamfer_cost(
                            scoring_points, candidate, refine_views, precision_weight, recall_weight, trim_fraction
                        )
                        if cost < best_delta_cost:
                            best_delta_cost = cost
                            best_delta = candidate
            if best_delta is None or best_delta_cost >= best_cost - 1.0e-9:
                break
            current = best_delta
            best_cost = best_delta_cost
        return current, best_cost

    best_T = None
    best_cost = float("inf")
    for seed in seeds:
        refined_T, refined_cost = _refine(seed)
        if refined_cost < best_cost:
            best_cost = refined_cost
            best_T = refined_T
    if best_T is None:
        best_T = coarse_best_T
        best_cost = float(scored[0][0])

    out_dir = os.path.join(reg_dir, "image_refine")
    os.makedirs(out_dir, exist_ok=True)
    overlay_serials = refine_serials[: min(len(refine_serials), int(getattr(args, "registration_image_overlay_cameras", 6)))]
    overlay_rows = _write_registration_image_refine_overlays(
        out_dir, vertices, faces, coarse_best_T, best_T, contexts, overlay_serials
    )

    inlier_count = int(np.count_nonzero(ray_inliers))
    diagnostics = {
        "method": "multiview_mask_silhouette",
        "translation_world": translation.tolist(),
        "translation_ray_count": int(len(origins)),
        "translation_ray_inliers": inlier_count,
        "mask_cameras": int(len(mask_serials)),
        "coarse_cameras": int(len(coarse_serials)),
        "refine_cameras": int(len(refine_serials)),
        "orientation_candidates": int(len(scored)),
        "coarse_best_cost": float(scored[0][0]),
        "refined_cost": float(best_cost),
        "refine_seeds": int(len(seeds)),
        "score_method": "chamfer",
        "score_points": int(scoring_points.shape[0]),
    }
    print(
        f"[MASKREG] {state_id}: masks={len(mask_serials)} rays={len(origins)}(in={inlier_count}) "
        f"cand={len(scored)} coarse={scored[0][0]:.4f} refined={best_cost:.4f}"
    )
    return {
        "T_world_object": best_T,
        "coarse_T_world_object": coarse_best_T,
        "diagnostics": diagnostics,
        "overlays": overlay_rows,
        "warnings": warnings,
        "best_cost": float(best_cost),
    }


def _register_state_mesh(report: dict, output_dir: str, mesh_info: dict, args: argparse.Namespace) -> dict:
    state_id = report["state_id"]
    reg_dir = os.path.join(output_dir, "registration", state_id)
    os.makedirs(reg_dir, exist_ok=True)
    summary = {
        "state_id": state_id,
        "status": "skipped",
        "method": args.registration_method,
        "refine_method": args.registration_refine_method,
        "output_dir": reg_dir,
        "input_mesh_path": mesh_info.get("mesh_path"),
        "pointcloud_path": None,
        "coarse_T_world_object": None,
        "T_world_object": None,
        "coarse_aligned_mesh_sample_path": None,
        "aligned_mesh_sample_path": None,
        "sparse_vs_aligned_mesh_overlay_path": None,
        "coarse_quality": {},
        "quality": {},
        "refinement": None,
        "warnings": [],
    }

    mesh_path = mesh_info.get("mesh_path")
    if not mesh_path or not os.path.exists(mesh_path):
        summary["warnings"].append("input mesh is missing.")
        _write_json(os.path.join(reg_dir, "registration.json"), summary)
        return summary

    pointcloud_path = _find_state_pointcloud(report, output_dir)
    summary["pointcloud_path"] = pointcloud_path
    target_mode = getattr(args, "registration_target", "auto")
    hull_path = _find_state_hull_pointcloud(output_dir, state_id) if target_mode in ("auto", "hull") else None
    summary["hull_pointcloud_path"] = hull_path
    if pointcloud_path is None and hull_path is None:
        summary["warnings"].append("state sparse pointcloud is missing.")
        _write_json(os.path.join(reg_dir, "registration.json"), summary)
        return summary

    try:
        target_source = "sparse"
        target_points = None
        target_colors = None
        if hull_path is not None:
            hull_points, hull_colors = _read_ascii_ply_points(hull_path)
            min_hull_points = max(3, int(getattr(args, "registration_hull_min_points", 100)))
            if hull_points.shape[0] >= min_hull_points:
                target_source = "hull"
                target_points, target_colors = hull_points, hull_colors
        if target_source != "hull":
            if target_mode == "hull":
                summary["warnings"].append(
                    "registration target 'hull' requested but the hull is missing/too small; using sparse."
                )
            if pointcloud_path is None:
                raise ValueError("no usable registration target (sparse missing, hull too small).")
            target_points, target_colors = _read_ascii_ply_points(pointcloud_path)
        summary["registration_target_source"] = target_source
        target_points, target_colors = _subsample_points_and_colors(
            target_points,
            target_colors,
            args.registration_max_points,
        )
        mesh_points = _load_mesh_sample_points(mesh_path, args.registration_sample_count)
        if target_points.shape[0] < 3:
            raise ValueError("target pointcloud has fewer than 3 points.")
        if mesh_points.shape[0] < 3:
            raise ValueError("mesh sample has fewer than 3 points.")

        if args.registration_method != "pca_centroid":
            raise ValueError(f"unsupported registration method: {args.registration_method}")

        import numpy as np

        mesh_diag = float(np.linalg.norm(np.max(mesh_points, axis=0) - np.min(mesh_points, axis=0)))
        object_crop_distance = float(getattr(args, "registration_object_crop_distance", 0.15))
        coarse_aligned_path = os.path.join(reg_dir, "coarse_aligned_mesh_sample.ply")
        quality_target_for_final = target_points

        coarse_method = str(getattr(args, "registration_coarse_method", "hull_pca"))
        mask_result = None
        if coarse_method == "mask_silhouette":
            try:
                ms_vertices, ms_faces = _load_mesh_vertices_faces(mesh_path)
                mask_result = _register_mask_silhouette(
                    report, output_dir, reg_dir, ms_vertices, ms_faces, mesh_points, args
                )
            except Exception as exc:
                summary["warnings"].append(f"mask_silhouette registration failed: {exc}")
                mask_result = None
            if mask_result is None:
                summary["warnings"].append(
                    "mask_silhouette registration unavailable (too few masks); using hull/pca fallback."
                )

        if mask_result is not None:
            T = mask_result["T_world_object"]
            coarse_T = mask_result["coarse_T_world_object"]
            transformed = _apply_transform(mesh_points, T)
            quality = _registration_quality(
                transformed, target_points, max_eval_points=int(args.registration_eval_points)
            )
            coarse_quality = _registration_quality(
                _apply_transform(mesh_points, coarse_T),
                target_points,
                max_eval_points=int(args.registration_eval_points),
            )
            diagnostics = mask_result["diagnostics"]
            status = "mask_silhouette"
            refinement = {
                "method": "multiview_mask_silhouette",
                "accepted": True,
                "refined_cost": mask_result["best_cost"],
                "diagnostics": mask_result["diagnostics"],
            }
            summary["coarse_T_world_object"] = coarse_T.tolist()
            _write_points_ply(
                coarse_aligned_path,
                _subsample_points(_apply_transform(mesh_points, coarse_T), args.registration_export_points),
                (255, 180, 80),
            )
        elif object_crop_distance > 0.0:
            roi = _object_roi_register(mesh_points, target_points, mesh_diag, args)
            T = roi["T_world_object"]
            transformed = _apply_transform(mesh_points, T)
            object_points = roi["object_points"]
            quality_target = object_points if object_points.shape[0] >= 3 else target_points
            quality_target_for_final = quality_target
            quality = _registration_quality(
                transformed,
                quality_target,
                max_eval_points=int(args.registration_eval_points),
            )
            coarse_quality = quality
            diagnostics = roi["coarse_candidates"]
            status = "object_roi_refined"
            refinement = {
                "method": "object_roi_crop_icp",
                "accepted": True,
                "selected_candidate_index": roi["selected_candidate_index"],
                "candidate_reports": roi["candidate_reports"],
                "object_quality": roi["object_quality"],
            }
            summary["object_registration"] = roi["object_quality"]
            summary["coarse_T_world_object"] = T.tolist()
            _write_points_ply(
                coarse_aligned_path,
                _subsample_points(transformed, args.registration_export_points),
                (255, 180, 80),
            )
            if object_points.shape[0] > 0:
                object_ply_path = os.path.join(reg_dir, "object_sparse_points.ply")
                _write_points_ply(
                    object_ply_path,
                    _subsample_points(object_points, args.registration_export_points),
                    (120, 255, 120),
                )
                summary["object_sparse_points_path"] = object_ply_path
        else:
            coarse_T, diagnostics = _estimate_pca_registration(mesh_points, target_points, args.registration_scale_mode)
            coarse_transformed = _apply_transform(mesh_points, coarse_T)
            coarse_quality = _registration_quality(
                coarse_transformed,
                target_points,
                max_eval_points=int(args.registration_eval_points),
            )
            _write_points_ply(
                coarse_aligned_path,
                _subsample_points(coarse_transformed, args.registration_export_points),
                (255, 180, 80),
            )

            T = coarse_T
            transformed = coarse_transformed
            quality = coarse_quality
            status = "coarse"
            refinement = None

            if args.registration_refine_method == "trimmed_icp":
                try:
                    refined_T, refinement = _refine_registration_trimmed_icp(mesh_points, target_points, coarse_T, args)
                    refined_transformed = _apply_transform(mesh_points, refined_T)
                    refined_quality = _registration_quality(
                        refined_transformed,
                        target_points,
                        max_eval_points=int(args.registration_eval_points),
                    )

                    coarse_median = coarse_quality.get("median_nn_distance")
                    refined_median = refined_quality.get("median_nn_distance")
                    accepted = True
                    if coarse_median is not None and refined_median is not None:
                        accepted = refined_median <= coarse_median * float(args.icp_acceptance_ratio)
                    refinement["accepted"] = accepted
                    refinement["quality"] = refined_quality

                    if accepted:
                        T = refined_T
                        transformed = refined_transformed
                        quality = refined_quality
                        status = "refined"
                    else:
                        refinement["status"] = "rejected"
                        summary["warnings"].append("ICP refinement was rejected because nearest-neighbor quality worsened.")
                except Exception as exc:
                    refinement = {
                        "method": "trimmed_icp",
                        "status": "failed",
                        "accepted": False,
                        "warning": str(exc),
                    }
                    summary["warnings"].append(f"ICP refinement failed; kept coarse transform: {exc}")
            elif args.registration_refine_method != "none":
                raise ValueError(f"unsupported registration refine method: {args.registration_refine_method}")

            summary["coarse_T_world_object"] = coarse_T.tolist()

        image_refinement = None
        if mask_result is not None:
            # mask_silhouette already fit the mesh to the masks and wrote overlays;
            # the edge/mask polish below is its fallback-path replacement.
            image_refinement = {
                "method": "multiview_mask_silhouette",
                "enabled": True,
                "accepted": True,
                "best_score": mask_result["best_cost"],
                "overlays": mask_result.get("overlays", []),
            }
        elif bool(getattr(args, "registration_image_refine", False)):
            try:
                mesh_vertices, mesh_faces = _load_mesh_vertices_faces(mesh_path)
                image_refinement = _refine_registration_with_image_edges(
                    report,
                    reg_dir,
                    mesh_vertices,
                    mesh_faces,
                    T,
                    quality_target_for_final,
                    args,
                    output_dir=output_dir,
                )
                if image_refinement.get("accepted") and image_refinement.get("T_world_object"):
                    T = _to_4x4_transform(image_refinement["T_world_object"], "image_refinement.T_world_object")
                    transformed = _apply_transform(mesh_points, T)
                    quality = _registration_quality(
                        transformed,
                        quality_target_for_final,
                        max_eval_points=int(args.registration_eval_points),
                    )
                    status = f"{status}_image_refined"
            except Exception as exc:
                image_refinement = {
                    "method": "single_frame_silhouette_edge_search",
                    "enabled": True,
                    "accepted": False,
                    "warning": str(exc),
                }
                summary["warnings"].append(f"image registration refinement failed: {exc}")

        aligned_path = os.path.join(reg_dir, "aligned_mesh_sample.ply")
        aligned_export_points = _subsample_points(transformed, args.registration_export_points)
        _write_points_ply(aligned_path, aligned_export_points, (80, 180, 255))
        sparse_export_points = _subsample_points(target_points, args.registration_export_points)
        overlay_path = os.path.join(reg_dir, "sparse_vs_aligned_mesh_overlay.ply")
        overlay_info = _write_colored_point_sets_ply(
            overlay_path,
            [
                (sparse_export_points, (255, 255, 255)),
                (aligned_export_points, (80, 180, 255)),
            ],
        )

        summary["status"] = status
        summary["T_world_object"] = T.tolist()
        summary["coarse_aligned_mesh_sample_path"] = coarse_aligned_path
        summary["aligned_mesh_sample_path"] = aligned_path
        summary["sparse_vs_aligned_mesh_overlay_path"] = overlay_info.get("path")
        summary["diagnostics"] = diagnostics
        summary["coarse_quality"] = coarse_quality
        summary["quality"] = quality
        summary["refinement"] = refinement
        summary["image_refinement"] = image_refinement
    except Exception as exc:
        summary["status"] = "failed"
        summary["warnings"].append(str(exc))

    _write_json(os.path.join(reg_dir, "registration.json"), summary)
    return summary


def _translation_refine_pose(
    points_obj: "np.ndarray",
    rotation: "np.ndarray",
    translation0: "np.ndarray",
    chamfer_views: List[dict],
    precision_weight: float,
    recall_weight: float,
    trim_fraction: float,
    mesh_diag: float,
    rounds: int,
    steps: Tuple[float, ...],
) -> Tuple["np.ndarray", float]:
    """Greedy translation-only refine with the orientation held fixed."""
    import numpy as np

    translation = np.asarray(translation0, dtype=np.float64).copy()
    world_axes = np.eye(3, dtype=np.float64)

    def _pose(t: "np.ndarray") -> "np.ndarray":
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = t
        return transform

    best_cost, _n = _multiview_chamfer_cost(
        points_obj, _pose(translation), chamfer_views, precision_weight, recall_weight, trim_fraction
    )
    for _round in range(max(0, int(rounds))):
        best_delta = None
        best_delta_cost = best_cost
        for axis_index in range(3):
            for step in steps:
                for sign in (1.0, -1.0):
                    candidate_t = translation + world_axes[axis_index] * (sign * step * mesh_diag)
                    cost, _n = _multiview_chamfer_cost(
                        points_obj, _pose(candidate_t), chamfer_views, precision_weight, recall_weight, trim_fraction
                    )
                    if cost < best_delta_cost:
                        best_delta_cost = cost
                        best_delta = candidate_t
        if best_delta is None or best_delta_cost >= best_cost - 1.0e-9:
            break
        translation = best_delta
        best_cost = best_delta_cost
    return translation, best_cost


def _share_placement_group_orientation(
    state_reports: List[dict],
    summaries: List[dict],
    output_dir: str,
    mesh_info: dict,
    args: argparse.Namespace,
) -> List[dict]:
    """Enforce one shared body orientation per placement group (mask_silhouette only).

    Within a placement the object body does not move (only the articulated part
    does), so all states of a group share a single body ``world_T_object``
    rotation. A single flat, near-symmetric state can lock onto a 90-degree
    in-plane flip (the moving articulated part mismatches the rest-pose mesh and
    misleads the per-view cost). This pass takes each group state's independent
    orientation and its +/-90/180-degree in-plane variants as candidates, scores
    every candidate against the pooled masks of the whole group (translation
    re-optimized per state), and adopts the group-best orientation for every
    state. The pooled evidence and the state whose part matches the mesh break the
    ambiguity that no single state could.
    """
    import numpy as np

    if str(getattr(args, "registration_coarse_method", "hull_pca")) != "mask_silhouette":
        return summaries
    if not bool(getattr(args, "registration_group_share_pose", True)):
        return summaries
    if str(getattr(args, "placement_group_mode", "auto")) == "single":
        # A single lumped group mixes different placements; sharing would be wrong.
        return summaries
    mesh_path = mesh_info.get("mesh_path")
    if not mesh_path or not os.path.exists(mesh_path):
        return summaries

    summary_by_state = {s.get("state_id"): s for s in summaries}
    groups: Dict[int, List[dict]] = {}
    for report in state_reports:
        gid = report.get("placement_group_id")
        if gid is None:
            continue
        summary = summary_by_state.get(report.get("state_id"))
        if not summary or summary.get("status") != "mask_silhouette" or summary.get("T_world_object") is None:
            continue
        groups.setdefault(int(gid), []).append(report)
    if not any(len(reports) >= 2 for reports in groups.values()):
        return summaries

    precision_weight = max(0.0, float(getattr(args, "registration_mask_precision_weight", 1.0)))
    recall_weight = max(0.0, float(getattr(args, "registration_mask_recall_weight", 0.5)))
    trim_fraction = float(getattr(args, "registration_silhouette_trim_fraction", 0.3))
    refine_cameras = int(getattr(args, "registration_silhouette_refine_cameras", 12))
    refine_downscale = max(1, int(getattr(args, "registration_image_mask_downscale", 2)))
    rounds = max(0, int(getattr(args, "registration_silhouette_refine_rounds", 12)))
    translation_steps = (0.04, 0.015, 0.005)

    vertices, faces = _load_mesh_vertices_faces(mesh_path)
    mesh_points = _load_mesh_sample_points(mesh_path, args.registration_sample_count)
    score_budget = max(64, int(getattr(args, "registration_silhouette_score_points", 2000)))
    scoring_points = _subsample_points(mesh_points, score_budget)
    mesh_diag = float(np.linalg.norm(np.max(mesh_points, axis=0) - np.min(mesh_points, axis=0)))
    _centroid, mesh_axes, _extents = _pca_axes(mesh_points)
    normal_axis = mesh_axes[:, 2]

    for gid in sorted(groups.keys()):
        reports = groups[gid]
        if len(reports) < 2:
            continue
        state_data = []
        for report in reports:
            bundle, _warnings = _load_state_camera_bundle(report["path"], report.get("undistorted_cameras", []))
            if not bundle:
                continue
            serials_all = sorted(bundle.keys())
            contexts = _prepare_registration_image_contexts(
                bundle, serials_all, None, args, output_dir=output_dir, state_id=str(report.get("state_id"))
            )
            mask_serials = [s for s in serials_all if contexts.get(s, {}).get("object_mask") is not None]
            use_serials = mask_serials if refine_cameras <= 0 else mask_serials[:refine_cameras]
            if not use_serials:
                continue
            views = [
                _build_chamfer_view(contexts[s]["object_mask"], contexts[s]["cam"], refine_downscale) for s in use_serials
            ]
            pose = np.asarray(summary_by_state[report["state_id"]]["T_world_object"], dtype=np.float64)
            state_data.append(
                {
                    "report": report,
                    "contexts": contexts,
                    "serials": use_serials,
                    "views": views,
                    "rotation": pose[:3, :3].copy(),
                    "translation": pose[:3, 3].copy(),
                }
            )
        if len(state_data) < 2:
            continue

        candidates: List["np.ndarray"] = []
        for data in state_data:
            for k in range(4):
                rot = data["rotation"] @ _rotation_about_axis(normal_axis, k * np.pi / 2.0)
                if not any(float(np.trace(kept.T @ rot)) > 3.0 - 1.0e-3 for kept in candidates):
                    candidates.append(rot)

        # Score each shared-orientation candidate against the pooled group masks
        # using each state's existing translation -- cheap (one chamfer evaluation
        # per state, no per-candidate translation refine). The old scorer re-ran a
        # full greedy translation refine for every candidate, which dominated the
        # runtime; the discrete 0/90/180/270-degree choice does not need it.
        best_rotation = None
        best_group_cost = float("inf")
        for rotation in candidates:
            costs = []
            for data in state_data:
                candidate_T = np.eye(4, dtype=np.float64)
                candidate_T[:3, :3] = rotation
                candidate_T[:3, 3] = data["translation"]
                cost, _n = _multiview_chamfer_cost(
                    scoring_points, candidate_T, data["views"], precision_weight, recall_weight, trim_fraction
                )
                costs.append(cost)
            group_cost = float(np.mean(costs))
            if group_cost < best_group_cost:
                best_group_cost = group_cost
                best_rotation = rotation
        if best_rotation is None:
            continue

        # One final translation refine per state under the adopted shared orientation.
        translations = []
        costs = []
        for data in state_data:
            t_refined, cost = _translation_refine_pose(
                scoring_points, best_rotation, data["translation"], data["views"],
                precision_weight, recall_weight, trim_fraction, mesh_diag, rounds, translation_steps,
            )
            translations.append(t_refined)
            costs.append(cost)
        best = {
            "rotation": best_rotation,
            "translations": translations,
            "costs": costs,
            "cost": float(np.mean(costs)),
        }

        for data, t_new, state_cost in zip(state_data, best["translations"], best["costs"]):
            report = data["report"]
            summary = summary_by_state[report["state_id"]]
            base_T = np.eye(4, dtype=np.float64)
            base_T[:3, :3] = data["rotation"]
            base_T[:3, 3] = data["translation"]
            new_T = np.eye(4, dtype=np.float64)
            new_T[:3, :3] = best["rotation"]
            new_T[:3, 3] = t_new
            reg_dir = summary.get("output_dir") or os.path.join(output_dir, "registration", report["state_id"])
            out_dir = os.path.join(reg_dir, "image_refine")
            os.makedirs(out_dir, exist_ok=True)
            overlay_serials = data["serials"][: min(len(data["serials"]), int(getattr(args, "registration_image_overlay_cameras", 6)))]
            _write_registration_image_refine_overlays(
                out_dir, vertices, faces, base_T, new_T, data["contexts"], overlay_serials
            )
            target_points = None
            hull_path = summary.get("hull_pointcloud_path")
            if hull_path and os.path.exists(hull_path):
                hull_points, _colors = _read_ascii_ply_points(hull_path)
                target_points = _subsample_points(hull_points, args.registration_max_points)
            if target_points is not None and target_points.shape[0] >= 3:
                summary["quality"] = _registration_quality(
                    _apply_transform(mesh_points, new_T), target_points, max_eval_points=int(args.registration_eval_points)
                )
            summary["T_world_object"] = new_T.tolist()
            summary["status"] = "mask_silhouette_group_shared"
            summary["group_shared_pose"] = {
                "placement_group_id": int(gid),
                "group_cost": float(best["cost"]),
                "state_cost": float(state_cost),
                "candidates_evaluated": int(len(candidates)),
                "pre_share_T_world_object": base_T.tolist(),
            }
            _write_json(os.path.join(reg_dir, "registration.json"), summary)
        print(
            f"[GROUPREG] group={gid}: states={len(state_data)} candidates={len(candidates)} "
            f"shared group_cost={best['cost']:.4f}"
        )

    return summaries


def _register_states(state_reports: List[dict], output_dir: str, mesh_info: dict, args: argparse.Namespace) -> List[dict]:
    if args.check_only or args.skip_registration:
        return []

    summaries = []
    for report in state_reports:
        summary = _register_state_mesh(report, output_dir, mesh_info, args)
        report["registration"] = summary
        summaries.append(summary)
        print(
            f"[REG] {report['state_id']} {report['state_label']}: "
            f"status={summary['status']}, target={summary.get('registration_target_source')}, "
            f"median_nn={summary.get('quality', {}).get('median_nn_distance')}"
        )
    summaries = _share_placement_group_orientation(state_reports, summaries, output_dir, mesh_info, args)
    return summaries


def _to_4x4_transform(value: Any, name: str) -> "np.ndarray":
    import numpy as np

    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)
    raise ValueError(f"{name} must be a 4x4 transform.")


def _part_registration_candidates(part_plan: dict) -> List[dict]:
    candidates = []
    for part in part_plan.get("parts", []):
        mesh_path = part.get("mesh_path")
        if part.get("mesh_source") == "whole_input_mesh":
            continue
        if mesh_path and os.path.exists(mesh_path):
            candidates.append(part)
    return candidates


def _build_segmented_part_targets(
    candidates: List[dict],
    source_points: "np.ndarray",
    source_colors: Optional["np.ndarray"],
    whole_registration: dict,
    args: argparse.Namespace,
) -> Dict[str, dict]:
    import numpy as np

    if args.skip_segmentation_targets or len(candidates) < 2:
        return {}
    whole_T_value = whole_registration.get("T_world_object")
    if whole_T_value is None:
        return {}

    canonical_points = _apply_transform(
        source_points,
        _invert_transform(_to_4x4_transform(whole_T_value, "T_world_object")),
    )

    part_samples = []
    usable_parts = []
    for part in candidates:
        mesh_path = part.get("segmented_mesh_path") or part.get("mesh_path")
        if not mesh_path or not os.path.exists(mesh_path):
            continue
        try:
            samples = _load_mesh_sample_points(mesh_path, int(args.part_target_sample_count))
        except Exception:
            continue
        if samples.shape[0] < 3:
            continue
        part_samples.append(samples)
        usable_parts.append(part)

    if len(usable_parts) < 2:
        return {}

    distance_columns = []
    for samples in part_samples:
        distance_columns.append(_nearest_distances(canonical_points, samples))
    distances = np.stack(distance_columns, axis=1)
    nearest = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)

    max_distance = float(args.part_target_max_distance)
    if max_distance > 0.0:
        valid = min_distances <= max_distance
    else:
        valid = np.ones((source_points.shape[0],), dtype=bool)

    targets: Dict[str, dict] = {}
    for part_idx, part in enumerate(usable_parts):
        keep = (nearest == part_idx) & valid
        target_points = source_points[keep]
        target_colors = (
            source_colors[keep] if source_colors is not None and source_colors.shape[0] == source_points.shape[0] else None
        )
        targets[part["part_id"]] = {
            "points": target_points,
            "colors": target_colors,
            "info": {
                "scope": "segmented_mesh_sparse_pointcloud",
                "part_id": part["part_id"],
                "mesh_path": part.get("segmented_mesh_path") or part.get("mesh_path"),
                "source_point_count": int(source_points.shape[0]),
                "selected_point_count": int(target_points.shape[0]),
                "mean_assignment_distance": float(np.mean(min_distances[keep])) if np.any(keep) else None,
                "median_assignment_distance": float(np.median(min_distances[keep])) if np.any(keep) else None,
                "max_distance": max_distance if max_distance > 0.0 else None,
            },
        }
    return targets


def _register_part_mesh_to_state(
    part: dict,
    report: dict,
    state_id: str,
    state_part_dir: str,
    source_points: "np.ndarray",
    source_colors: Optional["np.ndarray"],
    bundle: Dict[str, dict],
    segmented_targets: Dict[str, dict],
    whole_registration: dict,
    args: argparse.Namespace,
    session_path: str,
) -> dict:
    import numpy as np

    part_id = part["part_id"]
    part_dir = os.path.join(state_part_dir, part_id)
    os.makedirs(part_dir, exist_ok=True)
    summary = {
        "schema": "paradex.articulated_object.part_pose_observation.v0",
        "state_id": state_id,
        "part_id": part_id,
        "label": part.get("label"),
        "status": "skipped",
        "mesh_path": part.get("mesh_path"),
        "mesh_source": part.get("mesh_source"),
        "target_scope": None,
        "target_pointcloud_path": None,
        "target_point_count": 0,
        "target": None,
        "initial_T_world_part": None,
        "T_world_part": None,
        "initial_aligned_part_mesh_sample_path": None,
        "aligned_part_mesh_sample_path": None,
        "initial_quality": {},
        "quality": {},
        "refinement": None,
        "warnings": [],
    }

    mesh_path = part.get("mesh_path")
    if not mesh_path or not os.path.exists(mesh_path):
        summary["warnings"].append("part mesh is missing.")
        _write_json(os.path.join(part_dir, "part_registration.json"), summary)
        return summary

    whole_T_value = whole_registration.get("T_world_object")
    if whole_T_value is None:
        summary["warnings"].append("whole-object registration transform is missing.")
        _write_json(os.path.join(part_dir, "part_registration.json"), summary)
        return summary

    try:
        whole_T = _to_4x4_transform(whole_T_value, "T_world_object")
        canonical_T_part = _to_4x4_transform(part.get("canonical_T_part") or _identity4(), "canonical_T_part")
        initial_T = whole_T @ canonical_T_part
        target_points, target_colors, target_info = _select_part_target_points(
            report,
            part,
            part_dir,
            source_points,
            source_colors,
            bundle,
            segmented_targets,
            args,
            session_path,
        )
        summary["target_scope"] = target_info["scope"]
        summary["target_pointcloud_path"] = target_info.get("pointcloud_path")
        summary["target_point_count"] = target_info.get("target_point_count", 0)
        summary["target"] = target_info
        summary["warnings"].extend(target_info.get("warnings", []))
        mesh_points = _load_mesh_sample_points(mesh_path, int(args.part_registration_sample_count))
        if target_points.shape[0] < 3:
            raise ValueError("target pointcloud has fewer than 3 points.")
        if mesh_points.shape[0] < 3:
            raise ValueError("part mesh sample has fewer than 3 points.")

        initial_transformed = _apply_transform(mesh_points, initial_T)
        initial_quality = _registration_quality(
            initial_transformed,
            target_points,
            max_eval_points=int(args.registration_eval_points),
        )
        initial_aligned_path = os.path.join(part_dir, "initial_aligned_part_mesh_sample.ply")
        _write_points_ply(
            initial_aligned_path,
            _subsample_points(initial_transformed, args.registration_export_points),
            (255, 180, 120),
        )

        T = initial_T
        transformed = initial_transformed
        quality = initial_quality
        status = "initialized"
        refinement = None

        if args.part_registration_refine_method == "trimmed_icp":
            try:
                refined_T, refinement = _refine_registration_trimmed_icp(mesh_points, target_points, initial_T, args)
                refined_transformed = _apply_transform(mesh_points, refined_T)
                refined_quality = _registration_quality(
                    refined_transformed,
                    target_points,
                    max_eval_points=int(args.registration_eval_points),
                )

                initial_median = initial_quality.get("median_nn_distance")
                refined_median = refined_quality.get("median_nn_distance")
                accepted = True
                if initial_median is not None and refined_median is not None:
                    accepted = refined_median <= initial_median * float(args.part_icp_acceptance_ratio)
                refinement["accepted"] = accepted
                refinement["quality"] = refined_quality

                if accepted:
                    T = refined_T
                    transformed = refined_transformed
                    quality = refined_quality
                    status = "refined_candidate"
                else:
                    refinement["status"] = "rejected"
                    summary["warnings"].append("Part ICP was rejected because nearest-neighbor quality worsened.")
            except Exception as exc:
                refinement = {
                    "method": "trimmed_icp",
                    "status": "failed",
                    "accepted": False,
                    "warning": str(exc),
                }
                summary["warnings"].append(f"Part ICP failed; kept initial transform: {exc}")
        elif args.part_registration_refine_method != "none":
            raise ValueError(f"unsupported part registration refine method: {args.part_registration_refine_method}")

        aligned_path = os.path.join(part_dir, "aligned_part_mesh_sample.ply")
        _write_points_ply(aligned_path, _subsample_points(transformed, args.registration_export_points), (120, 255, 140))

        summary["status"] = status
        summary["initial_T_world_part"] = initial_T.tolist()
        summary["T_world_part"] = T.tolist()
        summary["initial_aligned_part_mesh_sample_path"] = initial_aligned_path
        summary["aligned_part_mesh_sample_path"] = aligned_path
        summary["initial_quality"] = initial_quality
        summary["quality"] = quality
        summary["refinement"] = refinement
    except Exception as exc:
        summary["status"] = "failed"
        summary["warnings"].append(str(exc))

    _write_json(os.path.join(part_dir, "part_registration.json"), summary)
    return summary


def _register_state_parts(
    report: dict,
    output_dir: str,
    part_plan: dict,
    args: argparse.Namespace,
    session_path: str,
) -> dict:
    state_id = report["state_id"]
    state_part_dir = os.path.join(output_dir, "part_registration", state_id)
    os.makedirs(state_part_dir, exist_ok=True)
    candidates = _part_registration_candidates(part_plan)
    summary = {
        "schema": "paradex.articulated_object.state_part_registration.v0",
        "state_id": state_id,
        "status": "skipped",
        "output_dir": state_part_dir,
        "pointcloud_path": None,
        "whole_registration_status": (report.get("registration") or {}).get("status"),
        "target_scope": None,
        "part_count": len(candidates),
        "parts": [],
        "warnings": [],
    }

    if not candidates:
        summary["warnings"].append("no ready part mesh candidates; provide --part-mesh-paths to enable part registration.")
        _write_json(os.path.join(state_part_dir, "part_registration.json"), summary)
        return summary

    whole_registration = report.get("registration") or {}
    if not whole_registration.get("T_world_object"):
        summary["warnings"].append("whole-object registration is missing; part registration needs it as an initial pose.")
        _write_json(os.path.join(state_part_dir, "part_registration.json"), summary)
        return summary

    pointcloud_path = _find_state_pointcloud(report, output_dir)
    summary["pointcloud_path"] = pointcloud_path
    if pointcloud_path is None:
        summary["warnings"].append("state sparse pointcloud is missing.")
        _write_json(os.path.join(state_part_dir, "part_registration.json"), summary)
        return summary

    try:
        source_points, source_colors = _read_ascii_ply_points(pointcloud_path)
        source_points, source_colors = _subsample_points_and_colors(
            source_points,
            source_colors,
            args.part_registration_max_points,
        )
        if source_points.shape[0] < 3:
            raise ValueError("target pointcloud has fewer than 3 points.")

        bundle, camera_warnings = _load_state_camera_bundle(report["path"], report["undistorted_cameras"])
        summary["warnings"].extend(camera_warnings)
        segmented_targets = _build_segmented_part_targets(
            candidates,
            source_points,
            source_colors,
            whole_registration,
            args,
        )

        part_summaries = []
        for part in candidates:
            part_summary = _register_part_mesh_to_state(
                part,
                report,
                state_id,
                state_part_dir,
                source_points,
                source_colors,
                bundle,
                segmented_targets,
                whole_registration,
                args,
                session_path,
            )
            part_summaries.append(part_summary)

        summary["parts"] = part_summaries
        if any(part_summary["status"] in {"refined_candidate", "initialized"} for part_summary in part_summaries):
            summary["status"] = "candidate"
        else:
            summary["status"] = "failed"
        scopes = sorted(set(part_summary.get("target_scope") for part_summary in part_summaries if part_summary.get("target_scope")))
        if len(scopes) == 1:
            summary["target_scope"] = scopes[0]
        elif scopes:
            summary["target_scope"] = "mixed"
        else:
            summary["target_scope"] = "unknown"
        if summary["target_scope"] == "whole_state_sparse_pointcloud":
            summary["warnings"].append(
                "Treat part poses as candidates until masks, segmented pointclouds, or motion consistency validation are added."
            )
        else:
            summary["warnings"].append(
                "Mask-filtered part poses are still candidates until cross-state motion consistency validation is added."
            )
    except Exception as exc:
        summary["status"] = "failed"
        summary["warnings"].append(str(exc))

    _write_json(os.path.join(state_part_dir, "part_registration.json"), summary)
    return summary


def _register_parts_for_states(
    state_reports: List[dict],
    output_dir: str,
    part_plan: dict,
    args: argparse.Namespace,
    session_path: str,
) -> List[dict]:
    if args.check_only or args.skip_registration or args.skip_part_registration:
        return []

    candidates = _part_registration_candidates(part_plan)
    if not candidates:
        return []

    summaries = []
    for report in state_reports:
        summary = _register_state_parts(report, output_dir, part_plan, args, session_path)
        report["part_registration"] = summary
        summaries.append(summary)
        refined = sum(1 for part in summary.get("parts", []) if part.get("status") == "refined_candidate")
        print(
            f"[PART-REG] {report['state_id']} {report['state_label']}: "
            f"status={summary['status']}, parts={len(summary.get('parts', []))}, refined={refined}"
        )
    return summaries


def _invert_transform(transform: "np.ndarray") -> "np.ndarray":
    import numpy as np

    T = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(T)


def _voxel_downsample_points_and_colors(
    points: "np.ndarray",
    colors: Optional["np.ndarray"],
    voxel_size: float,
) -> Tuple["np.ndarray", Optional["np.ndarray"]]:
    import numpy as np

    voxel_size = float(voxel_size)
    if voxel_size <= 0.0 or points.shape[0] == 0:
        return points, colors

    voxels: Dict[Tuple[int, int, int], List[int]] = {}
    keys = np.floor(points / voxel_size).astype(np.int64)
    for idx, key in enumerate(keys):
        key_tuple = (int(key[0]), int(key[1]), int(key[2]))
        voxels.setdefault(key_tuple, []).append(idx)

    sampled_points = []
    sampled_colors = [] if colors is not None and colors.shape[0] == points.shape[0] else None
    for indices in voxels.values():
        idx_arr = np.asarray(indices, dtype=np.int64)
        sampled_points.append(np.mean(points[idx_arr], axis=0))
        if sampled_colors is not None:
            sampled_colors.append(np.mean(colors[idx_arr], axis=0))

    out_points = np.asarray(sampled_points, dtype=np.float64)
    out_colors = None
    if sampled_colors is not None:
        out_colors = np.clip(np.asarray(sampled_colors, dtype=np.float64), 0, 255).astype(np.uint8)
    return out_points, out_colors


def _write_completed_surfel_obj(
    path: str,
    mesh: "trimesh.Trimesh",
    completion_points: "np.ndarray",
    surfel_radius: float,
) -> dict:
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    base_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    base_faces = np.asarray(mesh.faces, dtype=np.int64)
    completion_points = np.asarray(completion_points, dtype=np.float64)
    surfel_radius = float(surfel_radius)

    surfel_vertices = []
    surfel_faces = []
    if completion_points.shape[0] > 0 and surfel_radius > 0.0:
        if base_vertices.shape[0] > 0:
            nearest_indices, _ = _nearest_neighbors(completion_points, base_vertices)
            normals = completion_points - base_vertices[nearest_indices]
        else:
            normals = np.zeros_like(completion_points)
            normals[:, 2] = 1.0

        for point, normal in zip(completion_points, normals):
            norm = float(np.linalg.norm(normal))
            if norm <= 1.0e-12:
                normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                normal = normal / norm

            ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(np.dot(normal, ref))) > 0.9:
                ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            axis_u = np.cross(normal, ref)
            axis_u_norm = float(np.linalg.norm(axis_u))
            if axis_u_norm <= 1.0e-12:
                axis_u = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                axis_u = axis_u / axis_u_norm
            axis_v = np.cross(normal, axis_u)
            axis_v_norm = float(np.linalg.norm(axis_v))
            if axis_v_norm <= 1.0e-12:
                axis_v = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                axis_v = axis_v / axis_v_norm

            start = len(base_vertices) + len(surfel_vertices) + 1
            surfel_vertices.extend(
                [
                    point + surfel_radius * axis_u,
                    point + surfel_radius * (-0.5 * axis_u + 0.8660254038 * axis_v),
                    point + surfel_radius * (-0.5 * axis_u - 0.8660254038 * axis_v),
                ]
            )
            surfel_faces.append((start, start + 1, start + 2))

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Generated by articulated_object hidden-surface completion.\n")
        f.write("o completed_part\n")
        f.write("g base_mesh\n")
        for vertex in base_vertices:
            f.write(f"v {float(vertex[0]):.8f} {float(vertex[1]):.8f} {float(vertex[2]):.8f}\n")
        if completion_points.shape[0] > 0 and surfel_radius <= 0.0:
            f.write("g completion_points\n")
            for point in completion_points:
                f.write(f"v {float(point[0]):.8f} {float(point[1]):.8f} {float(point[2]):.8f}\n")
        for face in base_faces:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")
        if surfel_vertices:
            f.write("g completion_surfels\n")
            for vertex in surfel_vertices:
                f.write(f"v {float(vertex[0]):.8f} {float(vertex[1]):.8f} {float(vertex[2]):.8f}\n")
            for face in surfel_faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    return {
        "path": path,
        "base_vertex_count": int(base_vertices.shape[0]),
        "base_face_count": int(base_faces.shape[0]),
        "completion_point_count": int(completion_points.shape[0]),
        "surfel_face_count": int(len(surfel_faces)),
        "surfel_radius": surfel_radius,
    }


def _completion_uses_target_scope(scope: Optional[str], args: argparse.Namespace) -> bool:
    if scope in {"part_mask_sparse_pointcloud", "segmented_mesh_sparse_pointcloud"}:
        return True
    if scope == "whole_state_sparse_pointcloud" and args.completion_allow_whole_targets:
        return True
    return False


def _load_part_mesh_cache(part: dict, args: argparse.Namespace) -> Optional[dict]:
    import numpy as np

    mesh_path = part.get("mesh_path")
    if not mesh_path or not os.path.exists(mesh_path):
        return None
    try:
        mesh = _load_mesh_geometry(mesh_path)
        samples = _load_mesh_sample_points(mesh_path, int(args.completion_surface_sample_count))
        if samples.shape[0] == 0:
            samples = np.asarray(mesh.vertices, dtype=np.float64)
        return {"mesh": mesh, "samples": samples}
    except Exception:
        return None


def _collect_completion_points_for_part(
    part: dict,
    mesh_cache: dict,
    state_reports: List[dict],
    output_dir: str,
    args: argparse.Namespace,
) -> Tuple["np.ndarray", Optional["np.ndarray"], List[dict], List[str]]:
    import numpy as np

    part_id = part["part_id"]
    samples = mesh_cache["samples"]
    all_points = []
    all_colors = []
    all_have_colors = True
    observations = []
    warnings = []
    min_surface_distance = float(args.completion_min_surface_distance)
    max_surface_distance = float(args.completion_max_surface_distance)

    for report in state_reports:
        state_id = report["state_id"]
        pose = _state_part_pose_map(report).get(part_id)
        if pose is None:
            observations.append({"state_id": state_id, "status": "missing_pose"})
            continue

        scope = pose.get("target_scope")
        target_path = pose.get("target_pointcloud_path")
        if not _completion_uses_target_scope(scope, args):
            observations.append(
                {
                    "state_id": state_id,
                    "status": "skipped_target_scope",
                    "target_scope": scope,
                }
            )
            continue
        if not target_path or not os.path.exists(target_path):
            observations.append(
                {
                    "state_id": state_id,
                    "status": "missing_target_pointcloud",
                    "target_scope": scope,
                    "target_pointcloud_path": target_path,
                }
            )
            continue

        try:
            world_points, colors = _read_ascii_ply_points(target_path)
            world_points, colors = _subsample_points_and_colors(
                world_points,
                colors,
                int(args.completion_max_points_per_state),
            )
            if world_points.shape[0] == 0:
                observations.append(
                    {
                        "state_id": state_id,
                        "status": "empty_target_pointcloud",
                        "target_scope": scope,
                        "target_pointcloud_path": target_path,
                    }
                )
                continue

            T_world_part = _to_4x4_transform(pose["T_world_part"], "T_world_part")
            canonical_points = _apply_transform(world_points, _invert_transform(T_world_part))
            distances = _nearest_distances(canonical_points, samples)
            keep = np.isfinite(distances)
            if min_surface_distance > 0.0:
                keep &= distances >= min_surface_distance
            if max_surface_distance > 0.0:
                keep &= distances <= max_surface_distance

            selected = canonical_points[keep]
            selected_colors = colors[keep] if colors is not None and colors.shape[0] == world_points.shape[0] else None
            if selected.shape[0] > 0:
                all_points.append(selected)
                if selected_colors is None:
                    all_have_colors = False
                else:
                    all_colors.append(selected_colors)

            observations.append(
                {
                    "state_id": state_id,
                    "status": "ok",
                    "target_scope": scope,
                    "target_pointcloud_path": target_path,
                    "source_point_count": int(world_points.shape[0]),
                    "selected_point_count": int(selected.shape[0]),
                    "mean_surface_distance": float(np.mean(distances)) if distances.shape[0] else None,
                    "median_surface_distance": float(np.median(distances)) if distances.shape[0] else None,
                    "min_surface_distance": min_surface_distance,
                    "max_surface_distance": max_surface_distance if max_surface_distance > 0.0 else None,
                }
            )
        except Exception as exc:
            observations.append(
                {
                    "state_id": state_id,
                    "status": "failed",
                    "target_scope": scope,
                    "target_pointcloud_path": target_path,
                    "warning": str(exc),
                }
            )
            warnings.append(f"{state_id}: failed to collect completion points for {part_id}: {exc}")

    if not all_points:
        return np.zeros((0, 3), dtype=np.float64), None, observations, warnings

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = None
    if all_have_colors and all_colors:
        merged_colors = np.concatenate(all_colors, axis=0)

    merged_points, merged_colors = _voxel_downsample_points_and_colors(
        merged_points,
        merged_colors,
        float(args.completion_voxel_size),
    )
    return merged_points, merged_colors, observations, warnings


def _apply_completion_to_part_plan(part_plan: dict, completion_summary: dict) -> None:
    outputs = {part["part_id"]: part for part in completion_summary.get("parts", [])}
    for part in part_plan.get("parts", []):
        output = outputs.get(part["part_id"])
        if not output:
            continue
        part["completion"] = {
            "status": output.get("status"),
            "method": completion_summary.get("method"),
            "pointcloud_path": output.get("pointcloud_path"),
            "completed_mesh_path": output.get("completed_mesh_path"),
            "completion_point_count": output.get("completion_point_count", 0),
        }
        if output.get("completed_mesh_path"):
            part["completed_mesh_path"] = output["completed_mesh_path"]
            part["completed_mesh_source"] = "sparse_hidden_surface_surfels"


def _complete_hidden_surfaces(
    state_reports: List[dict],
    output_dir: str,
    part_plan: dict,
    args: argparse.Namespace,
) -> dict:
    import numpy as np

    completion_dir = os.path.join(output_dir, "completion")
    parts_dir = os.path.join(completion_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    candidates = _part_registration_candidates(part_plan)
    summary = {
        "schema": "paradex.articulated_object.hidden_surface_completion.v0",
        "status": "skipped",
        "method": "sparse_residual_surfels",
        "output_dir": completion_dir,
        "parts_dir": parts_dir,
        "part_count": len(candidates),
        "parameters": {
            "min_surface_distance": float(args.completion_min_surface_distance),
            "max_surface_distance": (
                float(args.completion_max_surface_distance)
                if float(args.completion_max_surface_distance) > 0.0
                else None
            ),
            "voxel_size": float(args.completion_voxel_size),
            "min_points": int(args.completion_min_points),
            "surfel_radius": float(args.completion_surfel_radius),
            "allow_whole_targets": bool(args.completion_allow_whole_targets),
        },
        "parts": [],
        "limitations": [
            "Completion uses sparse multiview residual points only; this is not dense MVS, TSDF fusion, or watertight surface reconstruction."
        ],
        "warnings": [],
    }

    if args.check_only or args.skip_completion:
        _write_json(os.path.join(completion_dir, "completion.json"), summary)
        return summary
    if args.skip_registration or args.skip_part_registration:
        summary["warnings"].append("completion needs per-part registration results.")
        _write_json(os.path.join(completion_dir, "completion.json"), summary)
        return summary
    if not candidates:
        summary["warnings"].append("no ready part mesh candidates were available for completion.")
        _write_json(os.path.join(completion_dir, "completion.json"), summary)
        return summary

    ready_count = 0
    partial_count = 0
    for part in candidates:
        part_id = part["part_id"]
        part_dir = os.path.join(parts_dir, part_id)
        os.makedirs(part_dir, exist_ok=True)
        part_summary = {
            "part_id": part_id,
            "label": part.get("label"),
            "status": "skipped",
            "input_mesh_path": part.get("mesh_path"),
            "pointcloud_path": None,
            "completed_mesh_path": None,
            "completion_point_count": 0,
            "observation_count": 0,
            "observations": [],
            "warnings": [],
        }

        mesh_cache = _load_part_mesh_cache(part, args)
        if mesh_cache is None:
            part_summary["status"] = "failed"
            part_summary["warnings"].append("part mesh could not be loaded.")
            _write_json(os.path.join(part_dir, "completion.json"), part_summary)
            summary["parts"].append(part_summary)
            continue

        points, colors, observations, warnings = _collect_completion_points_for_part(
            part,
            mesh_cache,
            state_reports,
            output_dir,
            args,
        )
        part_summary["observations"] = observations
        part_summary["observation_count"] = sum(1 for obs in observations if obs.get("status") == "ok")
        part_summary["warnings"].extend(warnings)
        part_summary["completion_point_count"] = int(points.shape[0])

        if points.shape[0] < int(args.completion_min_points):
            part_summary["status"] = "insufficient_points"
            part_summary["warnings"].append(
                f"only {points.shape[0]} completion points after filtering; {int(args.completion_min_points)}+ required."
            )
            _write_json(os.path.join(part_dir, "completion.json"), part_summary)
            summary["parts"].append(part_summary)
            continue

        export_points, export_colors = _subsample_points_and_colors(
            points,
            colors,
            int(args.completion_export_points),
        )
        pointcloud_path = os.path.join(part_dir, "completion_points.ply")
        _write_target_points(pointcloud_path, export_points, export_colors)
        obj_path = os.path.join(part_dir, "completed_surfel_mesh.obj")
        obj_info = _write_completed_surfel_obj(
            obj_path,
            mesh_cache["mesh"],
            export_points,
            float(args.completion_surfel_radius),
        )

        part_summary["status"] = "candidate"
        part_summary["pointcloud_path"] = pointcloud_path
        part_summary["completed_mesh_path"] = obj_path
        part_summary["export"] = obj_info
        if export_points.shape[0] < points.shape[0]:
            part_summary["warnings"].append(
                f"exported {export_points.shape[0]} of {points.shape[0]} completion points; raise --completion-export-points for more."
            )
        ready_count += 1
        _write_json(os.path.join(part_dir, "completion.json"), part_summary)
        summary["parts"].append(part_summary)

    partial_count = sum(1 for part in summary["parts"] if part.get("status") == "insufficient_points")
    if ready_count == len(candidates):
        summary["status"] = "ok"
    elif ready_count > 0:
        summary["status"] = "partial"
        summary["warnings"].append(f"{ready_count}/{len(candidates)} parts produced completion candidates.")
    elif partial_count > 0:
        summary["status"] = "insufficient_points"
    else:
        summary["status"] = "failed"

    _apply_completion_to_part_plan(part_plan, summary)
    path = os.path.join(completion_dir, "completion.json")
    _write_json(path, summary)
    print(f"[COMPLETE] status={summary['status']}, parts={ready_count}/{len(candidates)}")
    return summary


def _transform_delta_summary(delta: "np.ndarray") -> dict:
    import numpy as np

    T = np.asarray(delta, dtype=np.float64).reshape(4, 4)
    return {
        "translation_norm": float(np.linalg.norm(T[:3, 3])),
        "rotation_deg": _rotation_angle_deg(T[:3, :3]),
        "translation": T[:3, 3].tolist(),
    }


def _pose_status_is_usable(status: Optional[str]) -> bool:
    return status in {"initialized", "refined_candidate"}


def _state_part_pose_map(report: dict) -> Dict[str, dict]:
    part_registration = report.get("part_registration") or {}
    pose_map = {}
    for part_summary in part_registration.get("parts", []):
        if not _pose_status_is_usable(part_summary.get("status")):
            continue
        if part_summary.get("T_world_part") is None:
            continue
        pose_map[part_summary["part_id"]] = part_summary
    return pose_map


def _build_part_pose_tracks(state_reports: List[dict], part_plan: dict) -> dict:
    tracks = {}
    for part in part_plan.get("parts", []):
        part_id = part["part_id"]
        observations = []
        for report in state_reports:
            pose = _state_part_pose_map(report).get(part_id)
            if pose is None:
                continue
            observations.append(
                {
                    "state_id": report["state_id"],
                    "state_label": report.get("state_label"),
                    "state_kind": report.get("state_kind"),
                    "status": pose.get("status"),
                    "target_scope": pose.get("target_scope"),
                    "target_point_count": pose.get("target_point_count"),
                    "T_world_part": pose.get("T_world_part"),
                    "quality": pose.get("quality"),
                    "warnings": pose.get("warnings", []),
                }
            )
        tracks[part_id] = {
            "part_id": part_id,
            "label": part.get("label"),
            "mesh_path": part.get("mesh_path"),
            "observation_count": len(observations),
            "observations": observations,
        }
    return tracks


def _select_motion_reference_state(state_reports: List[dict], args: argparse.Namespace) -> Optional[str]:
    if args.motion_reference_state:
        return args.motion_reference_state
    for report in state_reports:
        if _state_part_pose_map(report):
            return report["state_id"]
    return None


def _pair_motion_summary(state_observations: List[dict]) -> dict:
    import numpy as np

    translations = [obs["delta_from_reference"]["translation_norm"] for obs in state_observations]
    rotations = [obs["delta_from_reference"]["rotation_deg"] for obs in state_observations]
    if not translations:
        return {"state_count": 0}
    return {
        "state_count": len(state_observations),
        "max_translation_delta": float(np.max(translations)),
        "mean_translation_delta": float(np.mean(translations)),
        "max_rotation_delta_deg": float(np.max(rotations)),
        "mean_rotation_delta_deg": float(np.mean(rotations)),
    }


def _build_motion_observations(
    state_reports: List[dict],
    output_dir: str,
    part_plan: dict,
    args: argparse.Namespace,
) -> dict:
    import itertools
    import numpy as np

    motion_dir = os.path.join(output_dir, "motion_observations")
    os.makedirs(motion_dir, exist_ok=True)
    part_tracks = _build_part_pose_tracks(state_reports, part_plan)
    part_tracks_path = os.path.join(motion_dir, "part_pose_tracks.json")
    _write_json(
        part_tracks_path,
        {
            "schema": "paradex.articulated_object.part_pose_tracks.v0",
            "tracks": part_tracks,
        },
    )

    summary = {
        "schema": "paradex.articulated_object.motion_observations.v0",
        "status": "skipped",
        "output_dir": motion_dir,
        "part_pose_tracks_path": part_tracks_path,
        "reference_state_id": None,
        "part_ids": [part["part_id"] for part in part_plan.get("parts", [])],
        "pairs": [],
        "warnings": [],
    }

    if args.check_only or args.skip_registration or args.skip_part_registration or args.skip_motion_observations:
        _write_json(os.path.join(motion_dir, "motion_observations.json"), summary)
        return summary

    part_ids = [part["part_id"] for part in part_plan.get("parts", [])]
    if len(part_ids) < 2:
        summary["warnings"].append("at least two parts are required for relative motion observations.")
        _write_json(os.path.join(motion_dir, "motion_observations.json"), summary)
        return summary

    reference_state_id = _select_motion_reference_state(state_reports, args)
    summary["reference_state_id"] = reference_state_id
    if reference_state_id is None:
        summary["warnings"].append("no usable part poses were found.")
        _write_json(os.path.join(motion_dir, "motion_observations.json"), summary)
        return summary

    pose_maps = {report["state_id"]: _state_part_pose_map(report) for report in state_reports}
    reference_pose_map = pose_maps.get(reference_state_id, {})
    if not reference_pose_map:
        summary["warnings"].append(f"reference state has no usable part poses: {reference_state_id}")
        _write_json(os.path.join(motion_dir, "motion_observations.json"), summary)
        return summary

    pair_summaries = []
    min_states = max(2, int(args.motion_min_states))
    for parent_id, child_id in itertools.combinations(part_ids, 2):
        if parent_id not in reference_pose_map or child_id not in reference_pose_map:
            continue

        ref_parent_T = _to_4x4_transform(reference_pose_map[parent_id]["T_world_part"], "reference parent T_world_part")
        ref_child_T = _to_4x4_transform(reference_pose_map[child_id]["T_world_part"], "reference child T_world_part")
        ref_T_parent_child = _invert_transform(ref_parent_T) @ ref_child_T

        observations = []
        for report in state_reports:
            state_id = report["state_id"]
            pose_map = pose_maps.get(state_id, {})
            if parent_id not in pose_map or child_id not in pose_map:
                continue
            parent_T = _to_4x4_transform(pose_map[parent_id]["T_world_part"], "parent T_world_part")
            child_T = _to_4x4_transform(pose_map[child_id]["T_world_part"], "child T_world_part")
            T_parent_child = _invert_transform(parent_T) @ child_T
            delta = _invert_transform(ref_T_parent_child) @ T_parent_child
            observations.append(
                {
                    "state_id": state_id,
                    "state_label": report.get("state_label"),
                    "state_kind": report.get("state_kind"),
                    "T_parent_child": T_parent_child.tolist(),
                    "delta_from_reference_T": delta.tolist(),
                    "delta_from_reference": _transform_delta_summary(delta),
                    "parent_status": pose_map[parent_id].get("status"),
                    "child_status": pose_map[child_id].get("status"),
                    "parent_target_scope": pose_map[parent_id].get("target_scope"),
                    "child_target_scope": pose_map[child_id].get("target_scope"),
                }
            )

        pair_summary = {
            "pair_id": f"{parent_id}__{child_id}",
            "parent_part_id": parent_id,
            "child_part_id": child_id,
            "reference_state_id": reference_state_id,
            "reference_T_parent_child": ref_T_parent_child.tolist(),
            "observation_count": len(observations),
            "status": "ok" if len(observations) >= min_states else "insufficient_states",
            "motion_summary": _pair_motion_summary(observations),
            "observations": observations,
        }
        pair_path = os.path.join(motion_dir, f"{pair_summary['pair_id']}.json")
        _write_json(pair_path, pair_summary)
        pair_summary["path"] = pair_path
        pair_summaries.append(pair_summary)

    summary["pairs"] = pair_summaries
    ok_pairs = [pair for pair in pair_summaries if pair["status"] == "ok"]
    if ok_pairs:
        summary["status"] = "ok"
    elif pair_summaries:
        summary["status"] = "insufficient_states"
        summary["warnings"].append(f"{min_states}+ states with both part poses are required per pair.")
    else:
        summary["status"] = "no_pairs"
        summary["warnings"].append("no part pair had usable poses in the reference state.")

    summary_path = os.path.join(motion_dir, "motion_observations.json")
    _write_json(summary_path, summary)
    print(
        f"[MOTION] status={summary['status']}, pairs={len(pair_summaries)}, "
        f"reference={summary['reference_state_id']}"
    )
    return summary


def _load_mesh_geometry(mesh_path: str) -> "trimesh.Trimesh":
    import trimesh

    geom = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(geom, trimesh.Scene):
        geoms = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"no mesh geometry in scene: {mesh_path}")
        geom = trimesh.util.concatenate(geoms)
    if not isinstance(geom, trimesh.Trimesh):
        raise ValueError(f"unsupported mesh type: {type(geom)}")
    return geom


def _kmeans_labels(features: "np.ndarray", k: int, iterations: int) -> Tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("k-means features must be a 2D array.")
    n = features.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, features.shape[1]), dtype=np.float64)
    k = int(k)
    if k <= 1:
        return np.zeros((n,), dtype=np.int64), np.mean(features, axis=0, keepdims=True)
    if k > n:
        raise ValueError("cluster count cannot exceed feature count.")

    centers = []
    first = int(np.argmin(np.mean(features, axis=1)))
    centers.append(features[first])
    while len(centers) < k:
        center_arr = np.asarray(centers, dtype=np.float64)
        d2 = np.min(np.sum((features[:, None, :] - center_arr[None, :, :]) ** 2, axis=2), axis=1)
        centers.append(features[int(np.argmax(d2))])
    centers = np.asarray(centers, dtype=np.float64)

    labels = np.zeros((n,), dtype=np.int64)
    for _ in range(max(1, int(iterations))):
        d2 = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1).astype(np.int64)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            members = features[labels == cluster_id]
            if members.shape[0] > 0:
                centers[cluster_id] = np.mean(members, axis=0)
            else:
                errors = np.min(d2, axis=1)
                centers[cluster_id] = features[int(np.argmax(errors))]
    return labels, centers


def _renumber_clusters_by_motion_score(
    cluster_labels: "np.ndarray",
    centers: "np.ndarray",
    part_count: int,
) -> Tuple["np.ndarray", dict]:
    import numpy as np

    scores = np.mean(centers, axis=1)
    ordered_clusters = sorted(range(part_count), key=lambda idx: (float(scores[idx]), idx))
    cluster_to_part = {int(cluster_id): int(part_idx) for part_idx, cluster_id in enumerate(ordered_clusters)}
    part_labels = np.asarray([cluster_to_part[int(cluster)] for cluster in cluster_labels], dtype=np.int64)
    return part_labels, {
        "cluster_to_part_index": cluster_to_part,
        "cluster_motion_scores": [float(score) for score in scores],
        "ordered_clusters": [int(cluster_id) for cluster_id in ordered_clusters],
    }


def _mesh_vertex_edges(mesh: "trimesh.Trimesh") -> "np.ndarray":
    import numpy as np

    try:
        edges = np.asarray(mesh.edges_unique, dtype=np.int64)
        if edges.size > 0:
            return edges.reshape(-1, 2)
    except Exception:
        pass

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _mesh_diagonal_from_vertices(vertices: "np.ndarray") -> float:
    import numpy as np

    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape[0] == 0:
        return 1.0
    bounds = np.stack([np.min(vertices, axis=0), np.max(vertices, axis=0)], axis=0)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    return diagonal if diagonal > 1.0e-12 else 1.0


def _dedupe_edges(edges: "np.ndarray") -> "np.ndarray":
    import numpy as np

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if edges.shape[0] == 0:
        return edges
    edges = np.sort(edges, axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(edges, axis=0)


def _exact_duplicate_vertex_edges(vertices: "np.ndarray") -> Tuple["np.ndarray", dict]:
    import numpy as np

    groups: Dict[Tuple[float, float, float], List[int]] = {}
    for index, vertex in enumerate(np.asarray(vertices, dtype=np.float64)):
        key = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
        groups.setdefault(key, []).append(int(index))

    edges = []
    duplicate_groups = 0
    largest_group = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        duplicate_groups += 1
        largest_group = max(largest_group, len(group))
        anchor = group[0]
        for member in group[1:]:
            edges.append((anchor, member))

    return _dedupe_edges(np.asarray(edges, dtype=np.int64)), {
        "duplicate_group_count": int(duplicate_groups),
        "largest_duplicate_group": int(largest_group),
    }


def _near_duplicate_vertex_edges(vertices: "np.ndarray", absolute_tolerance: float) -> Tuple["np.ndarray", dict]:
    import numpy as np

    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape[0] == 0 or absolute_tolerance <= 0.0:
        return np.zeros((0, 2), dtype=np.int64), {
            "duplicate_group_count": 0,
            "largest_duplicate_group": 0,
        }

    cell_size = max(float(absolute_tolerance), 1.0e-12)
    cells: Dict[Tuple[int, int, int], List[int]] = {}
    for index, vertex in enumerate(vertices):
        cell = tuple(np.floor(vertex / cell_size).astype(np.int64).tolist())
        cells.setdefault(cell, []).append(int(index))

    parent = list(range(vertices.shape[0]))
    rank = [0 for _ in range(vertices.shape[0])]

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    tolerance2 = float(absolute_tolerance * absolute_tolerance)
    for cell, indices in cells.items():
        for offset in neighbor_offsets:
            neighbor_cell = (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
            neighbor_indices = cells.get(neighbor_cell)
            if not neighbor_indices:
                continue
            for left in indices:
                for right in neighbor_indices:
                    if right <= left:
                        continue
                    diff = vertices[left] - vertices[right]
                    if float(diff @ diff) <= tolerance2:
                        union(left, right)

    groups: Dict[int, List[int]] = {}
    for index in range(vertices.shape[0]):
        groups.setdefault(find(index), []).append(index)

    edges = []
    duplicate_groups = 0
    largest_group = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        duplicate_groups += 1
        largest_group = max(largest_group, len(group))
        anchor = group[0]
        for member in group[1:]:
            edges.append((anchor, member))

    return _dedupe_edges(np.asarray(edges, dtype=np.int64)), {
        "duplicate_group_count": int(duplicate_groups),
        "largest_duplicate_group": int(largest_group),
    }


def _segmentation_adjacency_edges(mesh: "trimesh.Trimesh", args: argparse.Namespace) -> Tuple["np.ndarray", dict]:
    import numpy as np

    face_edges = _mesh_vertex_edges(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    diagonal = _mesh_diagonal_from_vertices(vertices)
    tolerance_ratio = float(getattr(args, "segmentation_adjacency_weld_tolerance", 1.0e-6))

    virtual_edges = np.zeros((0, 2), dtype=np.int64)
    weld_info = {
        "enabled": tolerance_ratio >= 0.0,
        "tolerance_ratio": tolerance_ratio,
        "absolute_tolerance": None,
        "virtual_edge_count": 0,
        "duplicate_group_count": 0,
        "largest_duplicate_group": 0,
    }
    if tolerance_ratio >= 0.0:
        absolute_tolerance = diagonal * tolerance_ratio
        weld_info["absolute_tolerance"] = float(absolute_tolerance)
        if absolute_tolerance > 0.0:
            virtual_edges, duplicate_info = _near_duplicate_vertex_edges(vertices, absolute_tolerance)
        else:
            virtual_edges, duplicate_info = _exact_duplicate_vertex_edges(vertices)
        weld_info.update(duplicate_info)
        weld_info["virtual_edge_count"] = int(virtual_edges.shape[0])

    if virtual_edges.shape[0] > 0:
        edges = _dedupe_edges(np.concatenate([face_edges, virtual_edges], axis=0))
    else:
        edges = face_edges

    return edges, {
        "face_edge_count": int(face_edges.shape[0]),
        "edge_count": int(edges.shape[0]),
        "mesh_diagonal": float(diagonal),
        "weld": weld_info,
    }


def _mesh_vertex_adjacency(vertex_count: int, edges: "np.ndarray") -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(vertex_count)]
    for a_value, b_value in edges:
        a = int(a_value)
        b = int(b_value)
        if 0 <= a < vertex_count and 0 <= b < vertex_count and a != b:
            adjacency[a].append(b)
            adjacency[b].append(a)
    return adjacency


def _connected_vertex_components(
    allowed_mask: "np.ndarray",
    adjacency: List[List[int]],
) -> List["np.ndarray"]:
    import numpy as np

    allowed = np.asarray(allowed_mask, dtype=bool)
    visited = np.zeros((allowed.shape[0],), dtype=bool)
    components = []
    for seed in np.flatnonzero(allowed):
        seed = int(seed)
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        members = []
        while stack:
            vertex = stack.pop()
            members.append(vertex)
            for neighbor in adjacency[vertex]:
                if allowed[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(np.asarray(members, dtype=np.int64))
    return components


def _candidate_boundary_fraction(mask: "np.ndarray", edges: "np.ndarray") -> float:
    import numpy as np

    if edges.shape[0] == 0:
        return 0.0
    mask = np.asarray(mask, dtype=bool)
    inside_edges = mask[edges[:, 0]]
    outside_edges = mask[edges[:, 1]]
    boundary_edges = edges[inside_edges != outside_edges]
    candidate_vertices = int(np.count_nonzero(mask))
    if candidate_vertices <= 0 or boundary_edges.shape[0] == 0:
        return 0.0
    boundary_vertices = np.unique(boundary_edges[mask[boundary_edges]].reshape(-1))
    return float(boundary_vertices.shape[0] / max(candidate_vertices, 1))


def _face_labels_from_vertex_labels(faces: "np.ndarray", vertex_labels: "np.ndarray", part_count: int) -> "np.ndarray":
    import numpy as np

    face_labels = np.zeros((faces.shape[0],), dtype=np.int64)
    for idx, face in enumerate(faces):
        labels = vertex_labels[np.asarray(face, dtype=np.int64)]
        counts = np.bincount(labels, minlength=part_count)
        face_labels[idx] = int(np.argmax(counts))
    return face_labels


def _best_neighbor_label_for_component(
    component: "np.ndarray",
    vertex_labels: "np.ndarray",
    adjacency: List[List[int]],
    part_count: int,
) -> Optional[int]:
    import numpy as np

    in_component = np.zeros((vertex_labels.shape[0],), dtype=bool)
    in_component[component] = True
    counts = np.zeros((part_count,), dtype=np.int64)
    current_label = int(vertex_labels[int(component[0])]) if component.shape[0] > 0 else -1
    for vertex_value in component:
        vertex = int(vertex_value)
        for neighbor in adjacency[vertex]:
            if in_component[neighbor]:
                continue
            label = int(vertex_labels[neighbor])
            if 0 <= label < part_count and label != current_label:
                counts[label] += 1
    if int(np.sum(counts)) == 0:
        return None
    return int(np.argmax(counts))


def _cleanup_segmentation_vertex_labels(
    mesh: "trimesh.Trimesh",
    vertex_labels: "np.ndarray",
    part_count: int,
    args: argparse.Namespace,
) -> Tuple["np.ndarray", dict]:
    import numpy as np

    labels = np.asarray(vertex_labels, dtype=np.int64).copy()
    if bool(args.skip_segmentation_cleanup):
        return labels, {"enabled": False, "reason": "skipped_by_argument"}

    vertex_count = int(labels.shape[0])
    if vertex_count == 0 or part_count < 2:
        return labels, {"enabled": False, "reason": "not_enough_vertices_or_parts"}

    edges, adjacency_info = _segmentation_adjacency_edges(mesh, args)
    adjacency = _mesh_vertex_adjacency(vertex_count, edges)
    min_fraction = max(0.0, float(args.segmentation_cleanup_min_component_fraction))
    min_vertices = max(
        1,
        int(args.segmentation_cleanup_min_component_vertices),
        int(np.ceil(vertex_count * min_fraction)),
    )
    max_changes = max(0, int(args.segmentation_cleanup_report_limit))

    changes = []
    reassigned_vertices = 0
    reassigned_components = 0
    for part_index in range(part_count):
        part_mask = labels == part_index
        components = _connected_vertex_components(part_mask, adjacency)
        if len(components) <= 1:
            continue
        components = sorted(components, key=lambda item: item.shape[0], reverse=True)

        # Keep the largest component for each label even if it is small; this protects true
        # small parts such as handles while removing stray islands assigned to the same label.
        for component_rank, component in enumerate(components[1:], start=1):
            if component.shape[0] >= min_vertices:
                continue
            target_label = _best_neighbor_label_for_component(component, labels, adjacency, part_count)
            if target_label is None or target_label == part_index:
                continue
            labels[component] = target_label
            reassigned_vertices += int(component.shape[0])
            reassigned_components += 1
            if len(changes) < max_changes:
                changes.append(
                    {
                        "from_part_index": int(part_index),
                        "to_part_index": int(target_label),
                        "component_rank": int(component_rank),
                        "vertex_count": int(component.shape[0]),
                    }
                )

    return labels, {
        "enabled": True,
        "min_component_vertices": int(min_vertices),
        "min_component_fraction": min_fraction,
        "adjacency": adjacency_info,
        "reassigned_vertex_count": int(reassigned_vertices),
        "reassigned_component_count": int(reassigned_components),
        "changes": changes,
        "changes_truncated": reassigned_components > len(changes),
    }


def _segmentation_component_stats(
    mesh: "trimesh.Trimesh",
    vertex_labels: "np.ndarray",
    part_count: int,
    args: argparse.Namespace,
) -> List[dict]:
    vertex_count = int(vertex_labels.shape[0])
    edges, _ = _segmentation_adjacency_edges(mesh, args)
    adjacency = _mesh_vertex_adjacency(vertex_count, edges)
    stats = []
    for part_index in range(part_count):
        components = _connected_vertex_components(vertex_labels == part_index, adjacency)
        sizes = sorted((int(component.shape[0]) for component in components), reverse=True)
        stats.append(
            {
                "part_index": int(part_index),
                "component_count": int(len(sizes)),
                "largest_component_vertices": int(sizes[0]) if sizes else 0,
                "largest_component_fraction": float(sizes[0] / max(vertex_count, 1)) if sizes else 0.0,
                "component_vertices": sizes[:10],
                "component_vertices_truncated": len(sizes) > 10,
            }
        )
    return stats


def _export_mesh_faces(mesh: "trimesh.Trimesh", face_indices: "np.ndarray", path: str) -> dict:
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if face_indices.shape[0] == 0:
        return {"path": None, "vertex_count": 0, "face_count": 0, "warnings": ["no faces assigned to this part."]}
    submesh = mesh.submesh([face_indices], append=True, repair=False)
    submesh.export(path)
    return {
        "path": path,
        "vertex_count": int(len(submesh.vertices)),
        "face_count": int(len(submesh.faces)),
        "warnings": [],
    }


def _segmentation_part_ids(part_plan: dict) -> List[str]:
    return [part["part_id"] for part in part_plan.get("parts", [])]


def _usable_state_for_segmentation(report: dict, output_dir: str) -> bool:
    registration = report.get("registration") or {}
    return bool(registration.get("T_world_object")) and _find_state_pointcloud(report, output_dir) is not None


def _load_canonical_state_points(
    report: dict,
    output_dir: str,
    max_points: int,
    mesh_vertices: Optional["np.ndarray"] = None,
    object_crop_fraction: float = 0.0,
) -> Tuple["np.ndarray", Optional["np.ndarray"], dict]:
    import numpy as np

    pointcloud_path = _find_state_pointcloud(report, output_dir)
    if pointcloud_path is None:
        raise ValueError(f"{report['state_id']}: state sparse pointcloud is missing.")
    registration = report.get("registration") or {}
    if not registration.get("T_world_object"):
        raise ValueError(f"{report['state_id']}: whole-object registration is missing.")

    points, colors = _read_ascii_ply_points(pointcloud_path)
    points, colors = _subsample_points_and_colors(points, colors, max_points)
    world_T_object = _to_4x4_transform(registration["T_world_object"], "T_world_object")
    canonical_points = _apply_transform(points, _invert_transform(world_T_object))

    info = {
        "state_id": report["state_id"],
        "pointcloud_path": pointcloud_path,
        "point_count": int(canonical_points.shape[0]),
    }
    # Optional object-ROI crop: keep only sparse points near the registered mesh
    # surface (in canonical frame). This removes scene background/outliers so
    # motion-residual and moving-evidence operate on object points only. Because
    # the moved part's displaced observations fall outside this shell, the crop
    # also sharpens motion signals: a vertex whose only nearby observation is its
    # rest-pose sparse point shows a large residual once that part moves away.
    if (
        mesh_vertices is not None
        and float(object_crop_fraction) > 0.0
        and canonical_points.shape[0] > 0
        and np.asarray(mesh_vertices).shape[0] > 0
    ):
        mesh_ref = _subsample_points(np.asarray(mesh_vertices, dtype=np.float64), 4000)
        diag = float(np.linalg.norm(np.max(mesh_ref, axis=0) - np.min(mesh_ref, axis=0)))
        threshold = float(object_crop_fraction) * max(diag, 1.0e-9)
        distances = _nearest_distances(canonical_points, mesh_ref)
        keep = distances <= threshold
        info["object_crop"] = {
            "applied": True,
            "crop_fraction": float(object_crop_fraction),
            "crop_distance": threshold,
            "before": int(canonical_points.shape[0]),
            "after": int(np.count_nonzero(keep)),
        }
        canonical_points = canonical_points[keep]
        if colors is not None and colors.shape[0] == keep.shape[0]:
            colors = colors[keep]
        info["point_count"] = int(canonical_points.shape[0])
    else:
        info["object_crop"] = {"applied": False}

    return canonical_points, colors, info


def _collect_motion_residual_features(
    mesh: "trimesh.Trimesh",
    state_reports: List[dict],
    output_dir: str,
    args: argparse.Namespace,
) -> Tuple["np.ndarray", List[dict], float]:
    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    scale = float(np.linalg.norm(bounds[1] - bounds[0])) if bounds.shape == (2, 3) else 1.0
    if scale <= 1.0e-12:
        scale = 1.0

    state_features = []
    state_infos = []
    for report in state_reports:
        if not _usable_state_for_segmentation(report, output_dir):
            continue
        canonical_points, _, info = _load_canonical_state_points(
            report,
            output_dir,
            int(args.segmentation_max_points_per_state),
            mesh_vertices=vertices,
            object_crop_fraction=float(getattr(args, "segmentation_object_crop_distance", 0.0)),
        )
        if canonical_points.shape[0] < 3:
            continue
        distances = _nearest_distances(vertices, canonical_points)
        normalized = np.clip(distances / scale, 0.0, float(args.segmentation_residual_clip))
        state_features.append(normalized)
        info["mean_normalized_residual"] = float(np.mean(normalized))
        info["median_normalized_residual"] = float(np.median(normalized))
        state_infos.append(info)

    if not state_features:
        return np.zeros((vertices.shape[0], 0), dtype=np.float64), state_infos, scale
    return np.stack(state_features, axis=1), state_infos, scale


def _segment_vertices_by_motion_residual(
    mesh: "trimesh.Trimesh",
    state_reports: List[dict],
    output_dir: str,
    part_count: int,
    args: argparse.Namespace,
) -> Tuple["np.ndarray", dict]:
    import numpy as np

    features, state_infos, scale = _collect_motion_residual_features(mesh, state_reports, output_dir, args)
    if features.shape[1] < max(2, int(args.segmentation_min_states)):
        raise ValueError(
            f"motion residual segmentation needs {max(2, int(args.segmentation_min_states))}+ usable states; "
            f"found {features.shape[1]}."
        )

    cluster_labels, centers = _kmeans_labels(features, part_count, int(args.segmentation_kmeans_iterations))
    vertex_labels, cluster_info = _renumber_clusters_by_motion_score(cluster_labels, centers, part_count)
    diagnostics = {
        "method": "motion_residual",
        "state_count": len(state_infos),
        "states": state_infos,
        "mesh_scale": scale,
        "feature_shape": [int(features.shape[0]), int(features.shape[1])],
        "cluster_info": cluster_info,
    }
    return vertex_labels, diagnostics


def _add_proposal_candidate(
    proposals: List[dict],
    seen: set,
    indices: "np.ndarray",
    vertex_count: int,
    edges: "np.ndarray",
    source: str,
    detail: dict,
    args: argparse.Namespace,
) -> None:
    import numpy as np

    indices = np.unique(np.asarray(indices, dtype=np.int64))
    if indices.shape[0] == 0:
        return

    min_fraction = max(0.0, float(args.segmentation_proposal_min_vertex_fraction))
    max_fraction = min(1.0, max(min_fraction, float(args.segmentation_proposal_max_vertex_fraction)))
    min_vertices = max(3, int(np.ceil(vertex_count * min_fraction)))
    max_vertices = max(min_vertices, int(np.floor(vertex_count * max_fraction)))
    if indices.shape[0] < min_vertices or indices.shape[0] > max_vertices:
        return

    signature = indices.tobytes()
    if signature in seen:
        return
    seen.add(signature)

    mask = np.zeros((vertex_count,), dtype=bool)
    mask[indices] = True
    boundary_fraction = _candidate_boundary_fraction(mask, edges)
    max_boundary = float(args.segmentation_proposal_max_boundary_fraction)
    if source != "disconnected_component" and max_boundary >= 0.0 and boundary_fraction > max_boundary:
        return

    proposals.append(
        {
            "mask": mask,
            "source": source,
            "detail": detail,
            "vertex_count": int(indices.shape[0]),
            "vertex_fraction": float(indices.shape[0] / max(vertex_count, 1)),
            "boundary_fraction": boundary_fraction,
        }
    )


def _add_moving_proposal_candidate(
    proposals: List[dict],
    seen: set,
    indices: "np.ndarray",
    vertex_count: int,
    edges: "np.ndarray",
    source: str,
    detail: dict,
    args: argparse.Namespace,
    seed_mask: Optional["np.ndarray"] = None,
) -> None:
    import numpy as np

    before_count = len(proposals)
    _add_proposal_candidate(proposals, seen, indices, vertex_count, edges, source, detail, args)
    if len(proposals) == before_count:
        return

    proposal = proposals[-1]
    if seed_mask is None:
        return

    candidate_mask = proposal["mask"]
    seed_mask = np.asarray(seed_mask, dtype=bool)
    seed_in_candidate = int(np.count_nonzero(candidate_mask & seed_mask))
    proposal["seed_mask"] = seed_mask.copy()
    proposal["seed_vertex_count"] = seed_in_candidate
    proposal["seed_fraction_of_candidate"] = float(seed_in_candidate / max(int(np.count_nonzero(candidate_mask)), 1))


def _dilate_vertex_mask(seed_mask: "np.ndarray", adjacency: List[List[int]], steps: int) -> "np.ndarray":
    import numpy as np

    mask = np.asarray(seed_mask, dtype=bool).copy()
    frontier = set(int(value) for value in np.flatnonzero(mask))
    for _ in range(max(0, int(steps))):
        if not frontier:
            break
        next_frontier = set()
        for vertex in frontier:
            for neighbor in adjacency[vertex]:
                if not mask[neighbor]:
                    mask[neighbor] = True
                    next_frontier.add(int(neighbor))
        frontier = next_frontier
    return mask


def _add_moving_seed_expansion_proposals(
    proposals: List[dict],
    seen: set,
    seed_indices: "np.ndarray",
    vertex_count: int,
    edges: "np.ndarray",
    adjacency: List[List[int]],
    source: str,
    detail: dict,
    motion_score: "np.ndarray",
    near_moving_points: "np.ndarray",
    min_motion_score: float,
    args: argparse.Namespace,
) -> None:
    import numpy as np

    seed_indices = np.unique(np.asarray(seed_indices, dtype=np.int64))
    if seed_indices.shape[0] == 0:
        return

    seed_mask = np.zeros((vertex_count,), dtype=bool)
    seed_mask[seed_indices] = True
    seed_detail = dict(detail)
    seed_detail["seed_vertex_count"] = int(seed_indices.shape[0])
    seed_detail["seed_motion_score_mean"] = float(np.mean(motion_score[seed_indices]))
    _add_moving_proposal_candidate(
        proposals,
        seen,
        seed_indices,
        vertex_count,
        edges,
        source,
        seed_detail,
        args,
        seed_mask=seed_mask,
    )

    grow_rings = max(0, int(getattr(args, "segmentation_moving_seed_grow_rings", 4)))
    for ring_count in range(1, grow_rings + 1):
        grown_mask = _dilate_vertex_mask(seed_mask, adjacency, ring_count)
        _add_moving_proposal_candidate(
            proposals,
            seen,
            np.flatnonzero(grown_mask),
            vertex_count,
            edges,
            f"{source}_grown_rings",
            {
                **seed_detail,
                "grow_rings": int(ring_count),
                "growth_mode": "mesh_rings",
            },
            args,
            seed_mask=seed_mask,
        )

    relaxed_steps = max(0, int(getattr(args, "segmentation_moving_seed_relax_steps", 4)))
    relaxed_min_fraction = max(
        0.0,
        min(1.0, float(getattr(args, "segmentation_moving_seed_relaxed_score_fraction", 0.25))),
    )
    if relaxed_steps > 0 and min_motion_score > 0.0:
        relax_fractions = np.linspace(1.0, relaxed_min_fraction, relaxed_steps + 1)[1:]
        for relax_fraction in relax_fractions:
            threshold = float(min_motion_score * relax_fraction)
            allowed = motion_score >= threshold
            if near_moving_points.shape[0] == vertex_count:
                allowed = allowed | (near_moving_points & (motion_score >= threshold * 0.5))
            for component in _connected_vertex_components(allowed, adjacency):
                if not np.any(seed_mask[component]):
                    continue
                _add_moving_proposal_candidate(
                    proposals,
                    seen,
                    component,
                    vertex_count,
                    edges,
                    f"{source}_relaxed_component",
                    {
                        **seed_detail,
                        "growth_mode": "relaxed_motion_component",
                        "relaxed_score_fraction": float(relax_fraction),
                        "threshold": threshold,
                    },
                    args,
                    seed_mask=seed_mask,
                )


def _build_geometry_part_proposals(
    mesh: "trimesh.Trimesh",
    residual_features: "np.ndarray",
    args: argparse.Namespace,
) -> Tuple[List[dict], dict]:
    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertex_count = int(vertices.shape[0])
    edges, adjacency_info = _segmentation_adjacency_edges(mesh, args)
    adjacency = _mesh_vertex_adjacency(vertex_count, edges)
    proposals: List[dict] = []
    seen = set()

    all_mask = np.ones((vertex_count,), dtype=bool)
    components = _connected_vertex_components(all_mask, adjacency)
    components_sorted = sorted(components, key=lambda item: item.shape[0], reverse=True)
    for component_index, component in enumerate(components_sorted[1:], start=1):
        _add_proposal_candidate(
            proposals,
            seen,
            component,
            vertex_count,
            edges,
            "disconnected_component",
            {"component_rank": component_index},
            args,
        )

    centroid, axes, extents = _pca_axes(vertices)
    projected = (vertices - centroid.reshape(1, 3)) @ axes
    tail_steps = max(1, int(args.segmentation_proposal_tail_steps))
    tail_min = max(0.001, float(args.segmentation_proposal_tail_min_fraction))
    tail_max = min(0.499, max(tail_min, float(args.segmentation_proposal_tail_max_fraction)))
    tail_fractions = np.linspace(tail_min, tail_max, tail_steps)

    for axis_index in range(min(3, projected.shape[1])):
        if extents[axis_index] <= 1.0e-12:
            continue
        values = projected[:, axis_index]
        for tail_fraction in tail_fractions:
            low_threshold = float(np.quantile(values, tail_fraction))
            high_threshold = float(np.quantile(values, 1.0 - tail_fraction))
            for side, allowed in (
                ("low", values <= low_threshold),
                ("high", values >= high_threshold),
            ):
                for component in _connected_vertex_components(allowed, adjacency):
                    _add_proposal_candidate(
                        proposals,
                        seen,
                        component,
                        vertex_count,
                        edges,
                        "pca_tail_component",
                        {
                            "axis_index": axis_index,
                            "axis_extent": float(extents[axis_index]),
                            "side": side,
                            "tail_fraction": float(tail_fraction),
                        },
                        args,
                    )

    if residual_features.shape[1] > 0:
        residual_mean = np.mean(residual_features, axis=1)
        residual_std = np.std(residual_features, axis=1) if residual_features.shape[1] > 1 else residual_mean
        for score_name, score_values in (("residual_mean", residual_mean), ("residual_std", residual_std)):
            for tail_fraction in tail_fractions:
                threshold = float(np.quantile(score_values, 1.0 - tail_fraction))
                allowed = score_values >= threshold
                for component in _connected_vertex_components(allowed, adjacency):
                    _add_proposal_candidate(
                        proposals,
                        seen,
                        component,
                        vertex_count,
                        edges,
                        "residual_tail_component",
                        {
                            "score": score_name,
                            "tail_fraction": float(tail_fraction),
                            "threshold": threshold,
                        },
                        args,
                    )

    diagnostics = {
        "vertex_count": vertex_count,
        "adjacency": adjacency_info,
        "edge_count": int(edges.shape[0]),
        "connected_component_count": len(components_sorted),
        "proposal_count": len(proposals),
        "pca_extents": [float(value) for value in extents],
    }
    return proposals, diagnostics


def _score_geometry_part_proposals(
    proposals: List[dict],
    residual_features: "np.ndarray",
    args: argparse.Namespace,
) -> List[dict]:
    import numpy as np

    if not proposals:
        return []

    has_residual = residual_features.shape[1] > 0
    residual_mean = np.mean(residual_features, axis=1) if has_residual else None
    residual_std = np.std(residual_features, axis=1) if residual_features.shape[1] > 1 else residual_mean
    observed_threshold = float(args.segmentation_proposal_observed_distance)
    observed_min = float(args.segmentation_proposal_min_observed_fraction)

    scored = []
    for proposal in proposals:
        mask = proposal["mask"]
        inverse = ~mask
        if not has_residual and proposal["source"] != "disconnected_component":
            proposal = dict(proposal)
            proposal.update(
                {
                    "score": -1.0e9,
                    "residual_excess": 0.0,
                    "residual_std_excess": 0.0,
                    "observed_fraction": None,
                    "rejected_reason": "no_residual_evidence_for_connected_proposal",
                }
            )
            scored.append(proposal)
            continue
        boundary_fraction = float(proposal["boundary_fraction"])
        neck_score = 1.0 - min(1.0, boundary_fraction / max(float(args.segmentation_proposal_max_boundary_fraction), 1.0e-9))
        size_penalty = 0.15 * float(proposal["vertex_fraction"])
        source_bonus = {
            "disconnected_component": 0.20,
            "pca_tail_component": 0.05,
            "residual_tail_component": 0.08,
        }.get(proposal["source"], 0.0)

        residual_excess = 0.0
        residual_std_excess = 0.0
        observed_fraction = None
        if has_residual:
            body_mean = float(np.mean(residual_mean[inverse])) if np.any(inverse) else 0.0
            body_std = float(np.mean(residual_std[inverse])) if np.any(inverse) else 0.0
            part_mean = float(np.mean(residual_mean[mask]))
            part_std = float(np.mean(residual_std[mask]))
            residual_excess = part_mean - body_mean
            residual_std_excess = part_std - body_std
            observed = np.min(residual_features[mask], axis=1) <= observed_threshold
            observed_fraction = float(np.mean(observed)) if observed.shape[0] > 0 else 0.0
            if observed_fraction < observed_min:
                proposal = dict(proposal)
                proposal.update(
                    {
                        "score": -1.0e9,
                        "residual_excess": residual_excess,
                        "residual_std_excess": residual_std_excess,
                        "observed_fraction": observed_fraction,
                        "rejected_reason": "below_min_observed_fraction",
                    }
                )
                scored.append(proposal)
                continue

        score = (
            source_bonus
            + 0.20 * neck_score
            - size_penalty
            + 3.0 * max(0.0, residual_excess)
            + 1.0 * max(0.0, residual_std_excess)
            + (0.20 * observed_fraction if observed_fraction is not None else 0.0)
        )
        proposal = dict(proposal)
        proposal.update(
            {
                "score": float(score),
                "neck_score": float(neck_score),
                "residual_excess": float(residual_excess),
                "residual_std_excess": float(residual_std_excess),
                "observed_fraction": observed_fraction,
            }
        )
        scored.append(proposal)

    return sorted(scored, key=lambda item: (float(item["score"]), -float(item["vertex_fraction"])), reverse=True)


def _select_geometry_part_proposals(
    scored: List[dict],
    required_count: int,
    args: argparse.Namespace,
) -> List[dict]:
    import numpy as np

    selected = []
    min_score = float(args.segmentation_proposal_min_score)
    overlap_threshold = float(args.segmentation_proposal_overlap_threshold)
    for proposal in scored:
        if float(proposal.get("score", -1.0e9)) < min_score:
            continue
        mask = proposal["mask"]
        too_much_overlap = False
        for prior in selected:
            prior_mask = prior["mask"]
            intersection = int(np.count_nonzero(mask & prior_mask))
            union = int(np.count_nonzero(mask | prior_mask))
            jaccard = float(intersection / max(union, 1))
            containment = float(intersection / max(int(np.count_nonzero(mask)), 1))
            if max(jaccard, containment) > overlap_threshold:
                too_much_overlap = True
                break
        if too_much_overlap:
            continue
        selected.append(proposal)
        if len(selected) >= required_count:
            break
    return selected


def _segment_vertices_by_geometry_proposals(
    mesh: "trimesh.Trimesh",
    state_reports: List[dict],
    output_dir: str,
    part_count: int,
    args: argparse.Namespace,
) -> Tuple["np.ndarray", dict]:
    import numpy as np

    if part_count < 2:
        raise ValueError("geometry proposal segmentation needs at least two planned part slots.")

    residual_features, state_infos, scale = _collect_motion_residual_features(mesh, state_reports, output_dir, args)
    proposals, proposal_diagnostics = _build_geometry_part_proposals(mesh, residual_features, args)
    scored = _score_geometry_part_proposals(proposals, residual_features, args)
    selected = _select_geometry_part_proposals(scored, part_count - 1, args)
    if len(selected) < part_count - 1:
        raise ValueError(
            f"geometry proposal segmentation needs {part_count - 1} accepted non-overlapping proposals; "
            f"found {len(selected)}."
        )

    vertex_count = int(len(mesh.vertices))
    vertex_labels = np.zeros((vertex_count,), dtype=np.int64)
    for part_index, proposal in enumerate(selected, start=1):
        vertex_labels[proposal["mask"]] = part_index

    def _proposal_for_json(proposal: dict) -> dict:
        return {
            "source": proposal.get("source"),
            "detail": proposal.get("detail"),
            "vertex_count": proposal.get("vertex_count"),
            "vertex_fraction": proposal.get("vertex_fraction"),
            "boundary_fraction": proposal.get("boundary_fraction"),
            "score": proposal.get("score"),
            "neck_score": proposal.get("neck_score"),
            "residual_excess": proposal.get("residual_excess"),
            "residual_std_excess": proposal.get("residual_std_excess"),
            "observed_fraction": proposal.get("observed_fraction"),
            "rejected_reason": proposal.get("rejected_reason"),
        }

    diagnostics = {
        "method": "geometry_proposals",
        "state_count": len(state_infos),
        "states": state_infos,
        "mesh_scale": scale,
        "feature_shape": [int(residual_features.shape[0]), int(residual_features.shape[1])],
        "proposal_generation": proposal_diagnostics,
        "selection": {
            "required_proposals": int(part_count - 1),
            "accepted_proposals": [_proposal_for_json(proposal) for proposal in selected],
            "top_scored_proposals": [_proposal_for_json(proposal) for proposal in scored[:20]],
        },
    }
    return vertex_labels, diagnostics


def _proposal_for_json(proposal: dict) -> dict:
    return {
        "source": proposal.get("source"),
        "detail": proposal.get("detail"),
        "vertex_count": proposal.get("vertex_count"),
        "vertex_fraction": proposal.get("vertex_fraction"),
        "boundary_fraction": proposal.get("boundary_fraction"),
        "score": proposal.get("score"),
        "neck_score": proposal.get("neck_score"),
        "motion_score_mean": proposal.get("motion_score_mean"),
        "motion_score_median": proposal.get("motion_score_median"),
        "sparse_motion_score_mean": proposal.get("sparse_motion_score_mean"),
        "image_motion_score_mean": proposal.get("image_motion_score_mean"),
        "image_motion_evidence_fraction": proposal.get("image_motion_evidence_fraction"),
        "motion_evidence_fraction": proposal.get("motion_evidence_fraction"),
        "observed_fraction": proposal.get("observed_fraction"),
        "near_moving_point_fraction": proposal.get("near_moving_point_fraction"),
        "seed_vertex_count": proposal.get("seed_vertex_count"),
        "seed_fraction_of_candidate": proposal.get("seed_fraction_of_candidate"),
        "attached_to_body": proposal.get("attached_to_body"),
        "base_score": proposal.get("base_score"),
        "two_body_fit": proposal.get("two_body_fit"),
        "raw_seed_penalty": proposal.get("raw_seed_penalty"),
        "geometry_penalty": proposal.get("geometry_penalty"),
        "diagnostic_mesh_path": proposal.get("diagnostic_mesh_path"),
        "diagnostic_face_count": proposal.get("diagnostic_face_count"),
        "residual_excess": proposal.get("residual_excess"),
        "residual_std_excess": proposal.get("residual_std_excess"),
        "rejected_reason": proposal.get("rejected_reason"),
    }


def _collect_moving_evidence_points(
    state_reports: List[dict],
    output_dir: str,
    args: argparse.Namespace,
    scale: float,
    mesh_vertices: Optional["np.ndarray"] = None,
) -> Tuple["np.ndarray", Optional[str], dict]:
    import numpy as np

    object_crop_fraction = float(getattr(args, "segmentation_object_crop_distance", 0.0))

    evidence_path = None
    diagnostics = {
        "reference_state_id": None,
        "distance_threshold": float(scale * float(getattr(args, "segmentation_moving_point_distance", 0.03))),
        "states": [],
        "point_count": 0,
        "path": None,
    }
    reference = _select_segmentation_reference_state(state_reports, output_dir, args)
    if reference is None:
        return np.zeros((0, 3), dtype=np.float64), evidence_path, diagnostics

    ref_points, _, ref_info = _load_canonical_state_points(
        reference,
        output_dir,
        int(args.segmentation_max_points_per_state),
        mesh_vertices=mesh_vertices,
        object_crop_fraction=object_crop_fraction,
    )
    diagnostics["reference_state_id"] = reference["state_id"]
    diagnostics["reference"] = ref_info
    if ref_points.shape[0] < 3:
        return np.zeros((0, 3), dtype=np.float64), evidence_path, diagnostics

    threshold = float(diagnostics["distance_threshold"])
    evidence_parts = []
    for report in state_reports:
        if report.get("state_id") == reference.get("state_id"):
            continue
        if not _usable_state_for_segmentation(report, output_dir):
            continue
        points, _, info = _load_canonical_state_points(
            report,
            output_dir,
            int(args.segmentation_max_points_per_state),
            mesh_vertices=mesh_vertices,
            object_crop_fraction=object_crop_fraction,
        )
        if points.shape[0] < 3:
            continue

        current_to_ref = _nearest_distances(points, ref_points)
        ref_to_current = _nearest_distances(ref_points, points)
        moved = points[current_to_ref > threshold]
        disappeared = ref_points[ref_to_current > threshold]
        if moved.shape[0] > 0:
            evidence_parts.append(moved)
        if disappeared.shape[0] > 0:
            # Disappeared reference points are often closest to the canonical moving-part mesh.
            evidence_parts.append(disappeared)
        diagnostics["states"].append(
            {
                "state_id": report["state_id"],
                "point_count": info.get("point_count"),
                "moved_point_count": int(moved.shape[0]),
                "disappeared_reference_point_count": int(disappeared.shape[0]),
                "mean_current_to_reference_distance": float(np.mean(current_to_ref)),
                "median_current_to_reference_distance": float(np.median(current_to_ref)),
                "mean_reference_to_current_distance": float(np.mean(ref_to_current)),
                "median_reference_to_current_distance": float(np.median(ref_to_current)),
            }
        )

    if not evidence_parts:
        return np.zeros((0, 3), dtype=np.float64), evidence_path, diagnostics

    evidence = np.concatenate(evidence_parts, axis=0)
    evidence = _subsample_points(evidence, int(getattr(args, "segmentation_moving_max_evidence_points", 20000)))
    diagnostics["point_count"] = int(evidence.shape[0])
    if evidence.shape[0] > 0 and not bool(getattr(args, "check_only", False)):
        evidence_path = os.path.join(output_dir, "segmentation", "moving_evidence_points.ply")
        _write_points_ply(evidence_path, evidence, (255, 120, 80))
        diagnostics["path"] = evidence_path
    return evidence, evidence_path, diagnostics


def _moving_evidence_score_threshold(args: argparse.Namespace) -> float:
    source = getattr(args, "segmentation_moving_evidence_source", "sparse")
    if source == "image_diff":
        return float(getattr(args, "segmentation_image_motion_min_score", 0.05))
    return float(getattr(args, "segmentation_moving_min_score", 0.02))


def _image_motion_diff_mask(
    reference_image: "np.ndarray",
    current_image: "np.ndarray",
    args: argparse.Namespace,
) -> Tuple[Optional["np.ndarray"], dict]:
    import cv2
    import numpy as np

    if reference_image is None or current_image is None:
        return None, {"status": "missing_image"}
    if reference_image.shape[:2] != current_image.shape[:2]:
        return None, {
            "status": "shape_mismatch",
            "reference_shape": list(reference_image.shape[:2]),
            "current_shape": list(current_image.shape[:2]),
        }

    if reference_image.ndim == 3:
        reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        reference_gray = reference_image
    if current_image.ndim == 3:
        current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    else:
        current_gray = current_image

    blur = max(0, int(getattr(args, "segmentation_image_motion_blur", 5)))
    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        reference_gray = cv2.GaussianBlur(reference_gray, (blur, blur), 0)
        current_gray = cv2.GaussianBlur(current_gray, (blur, blur), 0)

    diff = cv2.absdiff(reference_gray, current_gray)
    threshold = float(getattr(args, "segmentation_image_motion_threshold", 30.0))
    raw_mask = diff >= threshold
    raw_changed_fraction = float(np.mean(raw_mask)) if raw_mask.size else 0.0

    mask_u8 = (raw_mask.astype(np.uint8) * 255)
    dilate_iterations = max(0, int(getattr(args, "segmentation_image_motion_dilate_iterations", 2)))
    if dilate_iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=dilate_iterations)

    mask = mask_u8 > 0
    changed_fraction = float(np.mean(mask)) if mask.size else 0.0
    min_fraction = max(0.0, float(getattr(args, "segmentation_image_motion_min_changed_fraction", 0.0005)))
    max_fraction = float(getattr(args, "segmentation_image_motion_max_changed_fraction", 0.50))
    if changed_fraction < min_fraction:
        return None, {
            "status": "skipped_too_little_change",
            "threshold": threshold,
            "raw_changed_fraction": raw_changed_fraction,
            "changed_fraction": changed_fraction,
            "min_changed_fraction": min_fraction,
        }
    if max_fraction > 0.0 and changed_fraction > max_fraction:
        return None, {
            "status": "skipped_too_much_change",
            "threshold": threshold,
            "raw_changed_fraction": raw_changed_fraction,
            "changed_fraction": changed_fraction,
            "max_changed_fraction": max_fraction,
        }

    return mask, {
        "status": "ok",
        "threshold": threshold,
        "blur": blur,
        "dilate_iterations": dilate_iterations,
        "raw_changed_fraction": raw_changed_fraction,
        "changed_fraction": changed_fraction,
    }


def _vote_projected_vertices_from_mask(
    vertices: "np.ndarray",
    world_T_object: "np.ndarray",
    cam: dict,
    mask: "np.ndarray",
    valid_mask: Optional["np.ndarray"] = None,
) -> Tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    world_points = _apply_transform(vertices, world_T_object)
    pixels = _project_points(cam["projection"], world_points)
    depths = _depths_in_camera(cam["cam_from_world"], world_points)
    xy = np.rint(pixels).astype(np.int64)
    height, width = mask.shape[:2]
    visible = (
        np.isfinite(pixels).all(axis=1)
        & np.isfinite(depths)
        & (depths > 0.0)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )

    hits = np.zeros((vertices.shape[0],), dtype=bool)
    valid_indices = np.flatnonzero(visible)
    if valid_mask is not None and valid_indices.size > 0:
        # Points landing outside valid_mask (untrusted background) carry no evidence
        # either way: this camera neither supports nor vetoes them.
        trusted = valid_mask[xy[valid_indices, 1], xy[valid_indices, 0]]
        visible[valid_indices[~trusted]] = False
        valid_indices = valid_indices[trusted]
    if valid_indices.size > 0:
        hits[valid_indices] = mask[xy[valid_indices, 1], xy[valid_indices, 0]]
    return visible, hits


def _collect_image_motion_vertex_features(
    mesh: "trimesh.Trimesh",
    state_reports: List[dict],
    output_dir: str,
    args: argparse.Namespace,
) -> Tuple[Optional[dict], dict]:
    import cv2
    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertex_count = int(vertices.shape[0])
    diagnostics = {
        "enabled": True,
        "reference_state_id": None,
        "projection_mode": getattr(args, "segmentation_image_motion_projection_mode", "both"),
        "vertex_count": vertex_count,
        "state_count": 0,
        "camera_comparison_count": 0,
        "used_camera_comparison_count": 0,
        "warnings": [],
        "states": [],
        "paths": {},
    }

    reference = _select_segmentation_reference_state(state_reports, output_dir, args)
    if reference is None:
        diagnostics["warnings"].append("no usable reference state for image motion evidence.")
        return None, diagnostics
    reference_registration = reference.get("registration") or {}
    if not reference_registration.get("T_world_object"):
        diagnostics["warnings"].append("reference state is missing whole-object registration.")
        return None, diagnostics

    reference_world_T_object = _to_4x4_transform(reference_registration["T_world_object"], "T_world_object")
    reference_bundle, reference_camera_warnings = _load_state_camera_bundle(
        reference["path"],
        reference["undistorted_cameras"],
    )
    diagnostics["reference_state_id"] = reference["state_id"]
    if reference_camera_warnings:
        diagnostics["warnings"].extend(f"reference {warning}" for warning in reference_camera_warnings)
    if not reference_bundle:
        diagnostics["warnings"].append("reference state has no usable camera bundle.")
        return None, diagnostics

    vote_count = np.zeros((vertex_count,), dtype=np.float64)
    opportunity_count = np.zeros((vertex_count,), dtype=np.float64)
    per_camera_rows = []
    projection_mode = getattr(args, "segmentation_image_motion_projection_mode", "both")
    max_cameras = int(getattr(args, "segmentation_image_motion_max_cameras_per_state", 0))

    for report in state_reports:
        if report.get("state_id") == reference.get("state_id"):
            continue
        if not _same_placement_group(reference, report):
            diagnostics["states"].append(
                {"state_id": report.get("state_id"), "status": "skipped_cross_placement", "cameras": []}
            )
            continue
        registration = report.get("registration") or {}
        if not registration.get("T_world_object"):
            diagnostics["states"].append(
                {"state_id": report.get("state_id"), "status": "missing_registration", "cameras": []}
            )
            continue
        current_world_T_object = _to_4x4_transform(registration["T_world_object"], "T_world_object")
        current_bundle, camera_warnings = _load_state_camera_bundle(report["path"], report["undistorted_cameras"])
        common_serials = [serial for serial in sorted(reference_bundle.keys()) if serial in current_bundle]
        if max_cameras > 0:
            common_serials = common_serials[:max_cameras]

        state_info = {
            "state_id": report.get("state_id"),
            "status": "ok",
            "camera_count": len(common_serials),
            "used_camera_count": 0,
            "warnings": camera_warnings,
            "cameras": [],
        }
        diagnostics["state_count"] += 1

        for serial in common_serials:
            ref_cam = reference_bundle[serial]
            cur_cam = current_bundle[serial]
            ref_image = _cv2_imread(ref_cam["image_path"], cv2.IMREAD_COLOR)
            cur_image = _cv2_imread(cur_cam["image_path"], cv2.IMREAD_COLOR)
            mask, mask_info = _image_motion_diff_mask(ref_image, cur_image, args)
            camera_info = {
                "state_id": report.get("state_id"),
                "serial": serial,
                "reference_image": ref_cam["image_path"],
                "current_image": cur_cam["image_path"],
                "status": mask_info.get("status"),
                "raw_changed_fraction": mask_info.get("raw_changed_fraction"),
                "changed_fraction": mask_info.get("changed_fraction"),
                "visible_vertices": 0,
                "hit_vertices": 0,
            }
            diagnostics["camera_comparison_count"] += 1
            if mask is None:
                camera_info.update(mask_info)
                state_info["cameras"].append(camera_info)
                per_camera_rows.append(camera_info)
                continue

            visible_total = np.zeros((vertex_count,), dtype=bool)
            hit_total = np.zeros((vertex_count,), dtype=bool)
            if projection_mode in {"reference", "both"}:
                visible, hits = _vote_projected_vertices_from_mask(vertices, reference_world_T_object, ref_cam, mask)
                visible_total |= visible
                hit_total |= hits
            if projection_mode in {"current", "both"}:
                visible, hits = _vote_projected_vertices_from_mask(vertices, current_world_T_object, cur_cam, mask)
                visible_total |= visible
                hit_total |= hits

            opportunity_count[visible_total] += 1.0
            vote_count[hit_total] += 1.0
            camera_info["visible_vertices"] = int(np.count_nonzero(visible_total))
            camera_info["hit_vertices"] = int(np.count_nonzero(hit_total))
            camera_info["hit_fraction_of_visible_vertices"] = float(
                np.count_nonzero(hit_total) / max(np.count_nonzero(visible_total), 1)
            )
            camera_info.update(mask_info)
            state_info["used_camera_count"] += 1
            diagnostics["used_camera_comparison_count"] += 1
            state_info["cameras"].append(camera_info)
            per_camera_rows.append(camera_info)

        diagnostics["states"].append(state_info)

    score = np.zeros((vertex_count,), dtype=np.float64)
    valid = opportunity_count > 0.0
    score[valid] = vote_count[valid] / opportunity_count[valid]
    min_score = float(getattr(args, "segmentation_image_motion_min_score", 0.05))
    seed_mask = score >= min_score
    diagnostics["score"] = {
        "min": float(np.min(score)) if score.shape[0] else None,
        "mean": float(np.mean(score)) if score.shape[0] else None,
        "median": float(np.median(score)) if score.shape[0] else None,
        "p90": float(np.percentile(score, 90.0)) if score.shape[0] else None,
        "max": float(np.max(score)) if score.shape[0] else None,
        "threshold": min_score,
        "seed_vertex_count": int(np.count_nonzero(seed_mask)),
        "visible_vertex_count": int(np.count_nonzero(valid)),
    }

    if not bool(getattr(args, "check_only", False)):
        diag_dir = os.path.join(output_dir, "segmentation", "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        rows = [
            {
                "vertex_index": int(index),
                "image_motion_score": float(score[index]),
                "vote_count": int(vote_count[index]),
                "opportunity_count": int(opportunity_count[index]),
            }
            for index in range(vertex_count)
        ]
        vertex_score_path = os.path.join(diag_dir, "image_motion_vertex_scores.csv")
        _write_csv_rows(
            vertex_score_path,
            ["vertex_index", "image_motion_score", "vote_count", "opportunity_count"],
            rows,
        )
        diagnostics["paths"]["vertex_scores_csv"] = vertex_score_path

        camera_path = os.path.join(diag_dir, "image_motion_camera_comparisons.csv")
        _write_csv_rows(
            camera_path,
            [
                "state_id",
                "serial",
                "status",
                "raw_changed_fraction",
                "changed_fraction",
                "visible_vertices",
                "hit_vertices",
                "hit_fraction_of_visible_vertices",
                "reference_image",
                "current_image",
            ],
            per_camera_rows,
        )
        diagnostics["paths"]["camera_comparisons_csv"] = camera_path

        seed_path = os.path.join(diag_dir, "image_motion_seed_vertices.ply")
        _write_points_ply(seed_path, vertices[seed_mask], (255, 220, 40))
        diagnostics["paths"]["seed_vertices_ply"] = seed_path

    return {
        "score": score,
        "vote_count": vote_count,
        "opportunity_count": opportunity_count,
    }, diagnostics


def _load_mesh_vertices_faces(mesh_path: str) -> Tuple["np.ndarray", "np.ndarray"]:
    import numpy as np
    import trimesh

    geom = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(geom, trimesh.Scene):
        geoms = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        geom = trimesh.util.concatenate(geoms) if geoms else None
    if geom is None or not isinstance(geom, trimesh.Trimesh):
        raise ValueError(f"unsupported mesh for silhouette rendering: {mesh_path}")
    return np.asarray(geom.vertices, dtype=np.float64), np.asarray(geom.faces, dtype=np.int64)


def _render_mesh_silhouette_mask(
    vertices: "np.ndarray",
    faces: "np.ndarray",
    world_T_object: "np.ndarray",
    cam: dict,
    image_shape: Tuple[int, int],
    dilate_iters: int = 0,
) -> "np.ndarray":
    """Texture-free body silhouette: rasterize the registered mesh into the view.

    The mask is a pure coverage silhouette (filled triangles), so no per-pixel
    depth is required. Used to gate out the rigid body so that image evidence
    *beyond* the registered body isolates the moving/separated part.
    """
    import cv2
    import numpy as np

    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    if height <= 0 or width <= 0:
        return mask.astype(bool)
    world_points = _apply_transform(np.asarray(vertices, dtype=np.float64), world_T_object)
    pixels = _project_points(cam["projection"], world_points)
    depths = _depths_in_camera(cam["cam_from_world"], world_points)
    valid = np.isfinite(pixels).all(axis=1) & np.isfinite(depths) & (depths > 0.0)
    xy = np.clip(np.rint(pixels), -100000, 100000).astype(np.int32)
    faces_arr = np.asarray(faces, dtype=np.int64) if faces is not None else None
    if faces_arr is not None and faces_arr.shape[0] > 0:
        face_valid = valid[faces_arr].all(axis=1)
        tris = xy[faces_arr[face_valid]]
        if tris.shape[0] > 0:
            cv2.fillPoly(mask, [t for t in tris], 1)
    else:
        ok = valid & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        mask[xy[ok, 1], xy[ok, 0]] = 1
    if dilate_iters > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=int(dilate_iters))
    return mask.astype(bool)


def _render_silhouette_evidence(
    state_reports: List[dict],
    output_dir: str,
    mesh_info: dict,
    args: argparse.Namespace,
) -> dict:
    """Road S / stage 1: rendered-body-mask + beyond-body change evidence overlays.

    For each moved state and camera: (a) image change vs the reference state,
    (b) the registered mesh body silhouette, (c) beyond-body change = change and
    not body = the moving/separated part in image space. Writes side-by-side
    overlays so the body mask and the isolated part can be validated by eye before
    the multiview carve (stage 2) turns them into a 3D moving-part cloud.
    """
    import cv2
    import numpy as np

    summary = {"enabled": False, "states": [], "warnings": []}
    if not bool(getattr(args, "enable_silhouette_evidence", False)):
        return summary
    if bool(getattr(args, "check_only", False)):
        return summary
    mesh_path = mesh_info.get("mesh_path")
    if not mesh_path or not os.path.exists(mesh_path):
        summary["warnings"].append("mesh missing for silhouette evidence.")
        return summary
    try:
        vertices, faces = _load_mesh_vertices_faces(mesh_path)
    except Exception as exc:
        summary["warnings"].append(str(exc))
        return summary
    summary["enabled"] = True

    reference = _select_segmentation_reference_state(state_reports, output_dir, args)
    if reference is None:
        summary["warnings"].append("no usable reference state.")
        return summary
    ref_reg = reference.get("registration") or {}
    if not ref_reg.get("T_world_object"):
        summary["warnings"].append("reference state missing registration.")
        return summary
    ref_bundle, _ = _load_state_camera_bundle(reference["path"], reference.get("undistorted_cameras", []))
    if not ref_bundle:
        summary["warnings"].append("reference state has no usable cameras.")
        return summary
    summary["reference_state_id"] = reference.get("state_id")

    dilate_iters = int(getattr(args, "silhouette_body_dilate_iters", 3))
    max_cameras = int(getattr(args, "silhouette_max_cameras", 8))
    out_root = os.path.join(output_dir, "silhouette_evidence")

    for report in state_reports:
        if report.get("state_id") == reference.get("state_id"):
            continue
        if not _same_placement_group(reference, report):
            summary["states"].append(
                {"state_id": report.get("state_id"), "status": "skipped_cross_placement", "cameras": []}
            )
            continue
        reg = report.get("registration") or {}
        if not reg.get("T_world_object"):
            continue
        cur_T = _to_4x4_transform(reg["T_world_object"], "T_world_object")
        cur_bundle, _ = _load_state_camera_bundle(report["path"], report.get("undistorted_cameras", []))
        common = [s for s in sorted(ref_bundle.keys()) if s in cur_bundle]
        if max_cameras > 0:
            common = common[:max_cameras]
        state_dir = os.path.join(out_root, str(report.get("state_id")))
        state_info = {"state_id": report.get("state_id"), "cameras": []}
        for serial in common:
            cur_cam = cur_bundle[serial]
            ref_cam = ref_bundle[serial]
            ref_img = _cv2_imread(ref_cam["image_path"], cv2.IMREAD_COLOR)
            cur_img = _cv2_imread(cur_cam["image_path"], cv2.IMREAD_COLOR)
            if ref_img is None or cur_img is None or ref_img.shape[:2] != cur_img.shape[:2]:
                continue
            change, _change_info = _image_motion_diff_mask(ref_img, cur_img, args)
            if change is None:
                continue
            body = _render_mesh_silhouette_mask(vertices, faces, cur_T, cur_cam, cur_img.shape[:2], dilate_iters)
            beyond = change & (~body)
            change_px = int(np.count_nonzero(change))
            body_px = int(np.count_nonzero(body))
            beyond_px = int(np.count_nonzero(beyond))
            os.makedirs(state_dir, exist_ok=True)
            overlay = cur_img.copy()
            if body_px > 0:
                overlay[body] = (0.55 * overlay[body] + 0.45 * np.array([150, 150, 150], dtype=np.float64)).astype(np.uint8)
            overlay[change] = np.array([0, 0, 255], dtype=np.uint8)
            overlay[beyond] = np.array([0, 255, 0], dtype=np.uint8)
            panel = np.hstack([cur_img, overlay])
            out_path = os.path.join(state_dir, f"silhouette_{serial}.png")
            cv2.imwrite(out_path, panel)
            state_info["cameras"].append(
                {
                    "serial": serial,
                    "change_px": change_px,
                    "body_px": body_px,
                    "beyond_px": beyond_px,
                    "beyond_fraction_of_change": float(beyond_px / max(change_px, 1)),
                    "overlay_path": out_path,
                }
            )
        summary["states"].append(state_info)
        print(
            f"[SILH] {report.get('state_id')}: cameras={len(state_info['cameras'])} "
            f"overlays -> {state_dir}"
        )

    return summary


def _rotation_about_axis(axis: "np.ndarray", angle_rad: float) -> "np.ndarray":
    import numpy as np

    a = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(a))
    if norm < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = a / norm
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _silhouette_orientation_sweep(
    state_reports: List[dict],
    output_dir: str,
    mesh_info: dict,
    args: argparse.Namespace,
) -> dict:
    """Diagnose registration orientation error for flat/weakly-constrained objects.

    Rotates the mesh about its own thin axis (the slab normal) through a full turn
    and, for each angle, renders the body silhouette and scores how well its outline
    sits on image edges (distance-transform of Canny edges; lower is better). If a
    non-zero best angle aligns the silhouette, the registration orientation was
    wrong only in-plane (recoverable). If no angle aligns, the slab plane itself is
    tilted (needs a fuller pose search). Writes per-angle CSV plus current(0-deg)
    and best-angle overlays for visual confirmation.
    """
    import cv2
    import numpy as np

    summary = {"enabled": False, "states": [], "warnings": []}
    if not bool(getattr(args, "silhouette_orientation_sweep", False)):
        return summary
    if bool(getattr(args, "check_only", False)):
        return summary
    mesh_path = mesh_info.get("mesh_path")
    if not mesh_path or not os.path.exists(mesh_path):
        summary["warnings"].append("mesh missing for orientation sweep.")
        return summary
    try:
        vertices, faces = _load_mesh_vertices_faces(mesh_path)
    except Exception as exc:
        summary["warnings"].append(str(exc))
        return summary
    summary["enabled"] = True

    centroid = vertices.mean(axis=0)
    cov = np.cov((vertices - centroid).T)
    _eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    summary["object_normal"] = normal.tolist()

    step = max(1, int(getattr(args, "silhouette_sweep_step_deg", 15)))
    angles = list(range(0, 360, step))
    max_cameras = int(getattr(args, "silhouette_max_cameras", 8))
    out_root = os.path.join(output_dir, "silhouette_orientation")

    def _body_at(theta_deg, T, cam, shape):
        rot = _rotation_about_axis(normal, np.deg2rad(theta_deg))
        verts = (vertices - centroid) @ rot.T + centroid
        return _render_mesh_silhouette_mask(verts, faces, T, cam, shape, 0)

    for report in state_reports:
        reg = report.get("registration") or {}
        if not reg.get("T_world_object"):
            continue
        T = _to_4x4_transform(reg["T_world_object"], "T_world_object")
        bundle, _ = _load_state_camera_bundle(report["path"], report.get("undistorted_cameras", []))
        serials = sorted(bundle.keys())
        if max_cameras > 0:
            serials = serials[:max_cameras]
        state_dir = os.path.join(out_root, str(report.get("state_id")))
        state_info = {"state_id": report.get("state_id"), "cameras": []}
        for serial in serials:
            cam = bundle[serial]
            img = _cv2_imread(cam["image_path"], cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            inv = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
            rows = []
            best_theta = None
            best_score = None
            for theta in angles:
                body = _body_at(theta, T, cam, img.shape[:2])
                body_px = int(np.count_nonzero(body))
                if body_px > 0:
                    eroded = cv2.erode(body.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
                    boundary = body & (~eroded)
                    score = float(inv[boundary].mean()) if bool(boundary.any()) else float("inf")
                else:
                    score = float("inf")
                rows.append((theta, score, body_px))
                if best_score is None or score < best_score:
                    best_score = score
                    best_theta = theta
            os.makedirs(state_dir, exist_ok=True)
            csv_path = os.path.join(state_dir, f"sweep_{serial}.csv")
            with open(csv_path, "w", encoding="ascii") as handle:
                handle.write("theta_deg,edge_score,body_px\n")
                for theta, score, body_px in rows:
                    handle.write(f"{theta},{score},{body_px}\n")
            for theta, name in ((0.0, f"sweep_current_{serial}.png"), (float(best_theta), f"sweep_best_{serial}.png")):
                body = _body_at(theta, T, cam, img.shape[:2])
                overlay = img.copy()
                if bool(body.any()):
                    overlay[body] = (0.5 * overlay[body] + 0.5 * np.array([0, 220, 0], dtype=np.float64)).astype(np.uint8)
                cv2.imwrite(os.path.join(state_dir, name), np.hstack([img, overlay]))
            state_info["cameras"].append(
                {
                    "serial": serial,
                    "best_theta_deg": best_theta,
                    "best_score": best_score,
                    "score_at_0_deg": rows[0][1],
                }
            )
        summary["states"].append(state_info)
        print(
            f"[SWEEP] {report.get('state_id')}: best_theta per cam = "
            f"{[c['best_theta_deg'] for c in state_info['cameras']]}"
        )

    return summary


def _carve_moving_part_image_change(
    state_reports: List[dict],
    output_dir: str,
    object_roi_summary: dict,
    args: argparse.Namespace,
) -> dict:
    """Pose-free multiview carve of the moving part from inter-state image change.

    For a fixed-body / moving-articulation capture this recovers the moving part in
    3D world coordinates without any mesh registration or object masks: fill the
    camera-derived working volume (ROI sphere, above the ground plane) with voxels
    and keep the voxels that project into the state-to-reference image-change mask in
    a majority of views. Shadows fall on the floor (dropped by the above-ground
    constraint); background and photographer fall outside the ROI. The carved cloud
    contains the handle at both its rest and displaced positions and lives in the
    same world frame as ``object_points_roi.ply`` for registration-free validation.
    """
    import numpy as np

    summary = {"enabled": False, "states": [], "warnings": []}
    if not bool(getattr(args, "enable_moving_part_carve", False)):
        return summary
    if bool(getattr(args, "check_only", False)):
        return summary
    summary["enabled"] = True

    import cv2

    roi_states = {s.get("state_id"): s for s in (object_roi_summary or {}).get("states", [])}
    res = max(16, int(getattr(args, "carve_grid_resolution", 96)))
    min_view_fraction = float(getattr(args, "carve_min_view_fraction", 0.5))
    min_opportunities = int(getattr(args, "carve_min_opportunities", 3))
    max_cameras = int(getattr(args, "silhouette_max_cameras", 8))
    max_cameras = 0 if max_cameras <= 0 else max(max_cameras, 0)

    group_order: List[int] = []
    group_members: Dict[int, List[dict]] = {}
    for report in state_reports:
        group_id = int(report.get("placement_group_id", 0) or 0)
        if group_id not in group_members:
            group_members[group_id] = []
            group_order.append(group_id)
        group_members[group_id].append(report)

    out_root = os.path.join(output_dir, "moving_part_carve")
    summary["groups"] = []
    all_carved = []
    identity = np.eye(4, dtype=np.float64)

    for group_id in group_order:
        members = group_members[group_id]
        group_info = {"group_id": group_id, "state_ids": [str(m.get("state_id")) for m in members]}
        if len(members) < 2:
            group_info["status"] = "skipped_single_state"
            summary["groups"].append(group_info)
            continue

        reference = None
        preferred = getattr(args, "segmentation_reference_state", None)
        if preferred:
            for member in members:
                if member.get("state_id") == preferred:
                    reference = member
                    break
        if reference is None:
            for member in members:
                if _find_state_pointcloud(member, output_dir) is not None:
                    reference = member
                    break
        if reference is None:
            reference = members[0]
        ref_bundle, _ = _load_state_camera_bundle(reference["path"], reference.get("undistorted_cameras", []))
        if not ref_bundle:
            group_info["status"] = "no_reference_cameras"
            summary["groups"].append(group_info)
            continue
        group_info["reference_state_id"] = reference.get("state_id")
        if "reference_state_id" not in summary:
            summary["reference_state_id"] = reference.get("state_id")

        ref_roi = roi_states.get(reference.get("state_id")) or {}
        center = np.asarray(ref_roi.get("roi_center", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        radius = float(ref_roi.get("roi_radius", 0.0))
        if radius <= 0.0:
            group_info["status"] = "no_roi_radius"
            summary["groups"].append(group_info)
            summary["warnings"].append(
                f"group {group_id}: ROI radius unavailable (enable --roi-crop-mode auto with a mesh)."
            )
            continue
        ground = ref_roi.get("ground_plane")

        lin = np.linspace(-radius, radius, res)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1) + center.reshape(1, 3)
        inside = np.linalg.norm(grid - center.reshape(1, 3), axis=1) <= radius
        if ground is not None and ground.get("normal") is not None:
            normal = np.asarray(ground["normal"], dtype=np.float64)
            offset = float(ground.get("offset", 0.0))
            margin = 0.01 * radius
            inside &= (grid @ normal + offset) > margin
        voxels = grid[inside]
        group_info["voxel_count"] = int(voxels.shape[0])
        group_info["roi_center"] = center.tolist()
        group_info["roi_radius"] = radius
        if voxels.shape[0] == 0:
            group_info["status"] = "no_voxels"
            summary["groups"].append(group_info)
            continue
        group_info["status"] = "ok"
        summary["groups"].append(group_info)

        for report in members:
            if report.get("state_id") == reference.get("state_id"):
                continue
            cur_bundle, _ = _load_state_camera_bundle(report["path"], report.get("undistorted_cameras", []))
            common = [s for s in sorted(ref_bundle.keys()) if s in cur_bundle]
            if max_cameras > 0:
                common = common[:max_cameras]
            votes = np.zeros((voxels.shape[0],), dtype=np.float64)
            opportunities = np.zeros((voxels.shape[0],), dtype=np.float64)
            used_cameras = 0

            for serial in common:
                ref_img = _cv2_imread(ref_bundle[serial]["image_path"], cv2.IMREAD_COLOR)
                cur_img = _cv2_imread(cur_bundle[serial]["image_path"], cv2.IMREAD_COLOR)
                if ref_img is None or cur_img is None or ref_img.shape[:2] != cur_img.shape[:2]:
                    continue
                change, _info = _image_motion_diff_mask(ref_img, cur_img, args)
                if change is None:
                    continue
                visible, hits = _vote_projected_vertices_from_mask(voxels, identity, cur_bundle[serial], change)
                opportunities[visible] += 1.0
                votes[hits] += 1.0
                used_cameras += 1

            occupied = (opportunities >= float(min_opportunities)) & (votes >= min_view_fraction * np.maximum(opportunities, 1.0))
            carved = voxels[occupied]
            state_info = {
                "state_id": report.get("state_id"),
                "group_id": group_id,
                "reference_state_id": reference.get("state_id"),
                "used_cameras": used_cameras,
                "carved_point_count": int(carved.shape[0]),
            }
            if carved.shape[0] > 0:
                os.makedirs(out_root, exist_ok=True)
                state_path = os.path.join(out_root, f"carved_{report.get('state_id')}.ply")
                _write_points_ply(state_path, carved, (255, 60, 60))
                state_info["path"] = state_path
                all_carved.append(carved)
            summary["states"].append(state_info)
            print(
                f"[CARVE] group {group_id} {report.get('state_id')}: "
                f"cameras={used_cameras} carved_voxels={int(carved.shape[0])}"
            )

    if all_carved:
        combined = np.concatenate(all_carved, axis=0)
        os.makedirs(out_root, exist_ok=True)
        combined_path = os.path.join(out_root, "carved_all.ply")
        _write_points_ply(combined_path, combined, (255, 60, 60))
        summary["combined_path"] = combined_path
        summary["combined_point_count"] = int(combined.shape[0])

    return summary


def _build_moving_evidence_proposals(
    mesh: "trimesh.Trimesh",
    residual_features: "np.ndarray",
    moving_points: "np.ndarray",
    args: argparse.Namespace,
    image_motion_features: Optional[dict] = None,
) -> Tuple[List[dict], dict]:
    import numpy as np

    if residual_features.shape[1] < max(2, int(args.segmentation_min_states)):
        raise ValueError(
            f"moving evidence segmentation needs {max(2, int(args.segmentation_min_states))}+ usable states; "
            f"found {residual_features.shape[1]}."
        )

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertex_count = int(vertices.shape[0])
    edges, adjacency_info = _segmentation_adjacency_edges(mesh, args)
    adjacency = _mesh_vertex_adjacency(vertex_count, edges)
    sparse_motion_score = np.max(residual_features, axis=1) - np.min(residual_features, axis=1)
    observed_score = np.min(residual_features, axis=1)
    image_motion_score = None
    if image_motion_features is not None:
        candidate_score = image_motion_features.get("score")
        if candidate_score is not None and candidate_score.shape[0] == vertex_count:
            image_motion_score = np.asarray(candidate_score, dtype=np.float64)

    evidence_source = getattr(args, "segmentation_moving_evidence_source", "sparse")
    image_weight = float(getattr(args, "segmentation_image_motion_score_weight", 1.0))
    if evidence_source == "image_diff":
        motion_score = image_motion_score if image_motion_score is not None else np.zeros_like(sparse_motion_score)
    elif evidence_source == "combined":
        if image_motion_score is None:
            motion_score = sparse_motion_score
        else:
            motion_score = np.maximum(sparse_motion_score, image_motion_score * image_weight)
    else:
        motion_score = sparse_motion_score
    min_motion_score = _moving_evidence_score_threshold(args)
    observed_distance = float(getattr(args, "segmentation_moving_observed_distance", 0.03))
    mesh_diagonal = _mesh_diagonal_from_vertices(vertices)
    moving_point_distance = mesh_diagonal * float(getattr(args, "segmentation_moving_vertex_distance", 0.03))

    near_moving_points = np.zeros((vertex_count,), dtype=bool)
    moving_point_distances = np.zeros((vertex_count,), dtype=np.float64)
    if moving_points.shape[0] > 0:
        moving_point_distances = _nearest_distances(vertices, moving_points)
        near_moving_points = moving_point_distances <= moving_point_distance

    proposals: List[dict] = []
    seen = set()
    tail_steps = max(1, int(getattr(args, "segmentation_moving_tail_steps", 6)))
    tail_min = max(0.001, float(getattr(args, "segmentation_moving_tail_min_fraction", 0.03)))
    tail_max = min(0.499, max(tail_min, float(getattr(args, "segmentation_moving_tail_max_fraction", 0.30))))
    tail_fractions = np.linspace(tail_min, tail_max, tail_steps)

    for tail_fraction in tail_fractions:
        threshold = max(min_motion_score, float(np.quantile(motion_score, 1.0 - tail_fraction)))
        allowed = motion_score >= threshold
        for component in _connected_vertex_components(allowed, adjacency):
            _add_moving_seed_expansion_proposals(
                proposals,
                seen,
                component,
                vertex_count,
                edges,
                adjacency,
                "moving_score_component",
                {"tail_fraction": float(tail_fraction), "threshold": float(threshold)},
                motion_score,
                near_moving_points,
                min_motion_score,
                args,
            )
        if moving_points.shape[0] > 0:
            allowed_with_points = allowed & near_moving_points
            for component in _connected_vertex_components(allowed_with_points, adjacency):
                _add_moving_seed_expansion_proposals(
                    proposals,
                    seen,
                    component,
                    vertex_count,
                    edges,
                    adjacency,
                    "moving_point_supported_component",
                    {"tail_fraction": float(tail_fraction), "threshold": float(threshold)},
                    motion_score,
                    near_moving_points,
                    min_motion_score,
                    args,
                )

    geometry_proposals, geometry_diagnostics = _build_geometry_part_proposals(mesh, residual_features, args)
    geometry_overlap_min = float(getattr(args, "segmentation_moving_geometry_overlap_fraction", 0.25))
    motion_evidence_mask = motion_score >= min_motion_score
    seed_union = np.zeros((vertex_count,), dtype=bool)
    for proposal in proposals:
        seed_mask = proposal.get("seed_mask")
        if seed_mask is not None:
            seed_union |= np.asarray(seed_mask, dtype=bool)
    geometry_auxiliary_enabled = bool(getattr(args, "segmentation_moving_geometry_auxiliary", True))
    geometry_support_source = getattr(args, "segmentation_moving_geometry_support_source", "primary")
    geometry_auxiliary_added = 0
    geometry_auxiliary_rejected = 0
    if geometry_auxiliary_enabled:
        for proposal in geometry_proposals:
            mask = proposal["mask"]
            if int(np.count_nonzero(mask)) == 0:
                continue
            motion_overlap = float(np.mean(motion_evidence_mask[mask]))
            point_overlap = float(np.mean(near_moving_points[mask])) if moving_points.shape[0] > 0 else 0.0
            seed_coverage = (
                float(np.count_nonzero(mask & seed_union) / max(int(np.count_nonzero(seed_union)), 1))
                if np.any(seed_union)
                else 0.0
            )
            if geometry_support_source == "primary_or_sparse" or evidence_source == "sparse":
                support_value = max(motion_overlap, point_overlap, seed_coverage)
            else:
                support_value = max(motion_overlap, seed_coverage)
            if support_value < geometry_overlap_min:
                geometry_auxiliary_rejected += 1
                continue
            before_count = len(proposals)
            _add_moving_proposal_candidate(
                proposals,
                seen,
                np.flatnonzero(mask),
                vertex_count,
                edges,
                "geometry_supported_by_moving_evidence",
                {
                    "geometry_source": proposal.get("source"),
                    "geometry_detail": proposal.get("detail"),
                    "motion_overlap": motion_overlap,
                    "moving_point_overlap": point_overlap,
                    "seed_coverage": seed_coverage,
                    "support_value": support_value,
                    "support_source": geometry_support_source,
                },
                args,
                seed_mask=seed_union if np.any(seed_union) else None,
            )
            if len(proposals) > before_count:
                geometry_auxiliary_added += 1

    diagnostics = {
        "adjacency": adjacency_info,
        "evidence_source": evidence_source,
        "motion_score": {
            "min": float(np.min(motion_score)) if motion_score.shape[0] else None,
            "mean": float(np.mean(motion_score)) if motion_score.shape[0] else None,
            "median": float(np.median(motion_score)) if motion_score.shape[0] else None,
            "p90": float(np.percentile(motion_score, 90.0)) if motion_score.shape[0] else None,
            "max": float(np.max(motion_score)) if motion_score.shape[0] else None,
            "threshold": min_motion_score,
        },
        "sparse_motion_score": {
            "min": float(np.min(sparse_motion_score)) if sparse_motion_score.shape[0] else None,
            "mean": float(np.mean(sparse_motion_score)) if sparse_motion_score.shape[0] else None,
            "median": float(np.median(sparse_motion_score)) if sparse_motion_score.shape[0] else None,
            "p90": float(np.percentile(sparse_motion_score, 90.0)) if sparse_motion_score.shape[0] else None,
            "max": float(np.max(sparse_motion_score)) if sparse_motion_score.shape[0] else None,
        },
        "image_motion_score": {
            "available": image_motion_score is not None,
            "weight": image_weight,
            "min": float(np.min(image_motion_score)) if image_motion_score is not None and image_motion_score.shape[0] else None,
            "mean": float(np.mean(image_motion_score)) if image_motion_score is not None and image_motion_score.shape[0] else None,
            "median": float(np.median(image_motion_score)) if image_motion_score is not None and image_motion_score.shape[0] else None,
            "p90": float(np.percentile(image_motion_score, 90.0)) if image_motion_score is not None and image_motion_score.shape[0] else None,
            "max": float(np.max(image_motion_score)) if image_motion_score is not None and image_motion_score.shape[0] else None,
        },
        "observed_score_threshold": observed_distance,
        "moving_point_vertex_distance": moving_point_distance,
        "moving_point_vertex_distance_ratio": float(getattr(args, "segmentation_moving_vertex_distance", 0.03)),
        "moving_point_count": int(moving_points.shape[0]),
        "near_moving_vertex_count": int(np.count_nonzero(near_moving_points)),
        "proposal_count": int(len(proposals)),
        "seed_union_vertex_count": int(np.count_nonzero(seed_union)),
        "seed_growth": {
            "grow_rings": int(getattr(args, "segmentation_moving_seed_grow_rings", 4)),
            "relax_steps": int(getattr(args, "segmentation_moving_seed_relax_steps", 4)),
            "relaxed_score_fraction": float(
                getattr(args, "segmentation_moving_seed_relaxed_score_fraction", 0.25)
            ),
        },
        "geometry_auxiliary": geometry_diagnostics,
        "geometry_auxiliary_selection": {
            "enabled": geometry_auxiliary_enabled,
            "support_source": geometry_support_source,
            "overlap_min": geometry_overlap_min,
            "added_count": int(geometry_auxiliary_added),
            "rejected_by_support_count": int(geometry_auxiliary_rejected),
        },
    }
    feature_bundle = {
        "motion_score": motion_score,
        "sparse_motion_score": sparse_motion_score,
        "image_motion_score": image_motion_score,
        "observed_score": observed_score,
        "near_moving_points": near_moving_points,
        "moving_point_distances": moving_point_distances,
    }
    for proposal in proposals:
        proposal["feature_bundle"] = feature_bundle
    return proposals, diagnostics


def _score_moving_evidence_proposals(
    proposals: List[dict],
    args: argparse.Namespace,
) -> List[dict]:
    import numpy as np

    scored = []
    min_motion_score = _moving_evidence_score_threshold(args)
    min_evidence_fraction = float(getattr(args, "segmentation_moving_min_evidence_fraction", 0.25))
    min_observed_fraction = float(getattr(args, "segmentation_moving_min_observed_fraction", 0.01))
    min_seed_vertices = max(1, int(getattr(args, "segmentation_moving_min_seed_vertices", 3)))
    observed_distance = float(getattr(args, "segmentation_moving_observed_distance", 0.03))
    max_boundary = max(float(args.segmentation_proposal_max_boundary_fraction), 1.0e-9)
    relation_prior = getattr(args, "segmentation_relation_prior", "articulated")
    raw_seed_penalty_value = max(0.0, float(getattr(args, "segmentation_moving_raw_seed_penalty", 0.80)))
    evidence_source = getattr(args, "segmentation_moving_evidence_source", "sparse")
    min_image_geometry_fraction = float(getattr(args, "segmentation_image_motion_min_geometry_fraction", 0.0))
    image_geometry_penalty = max(0.0, float(getattr(args, "segmentation_image_motion_geometry_penalty", 0.0)))
    for proposal in proposals:
        bundle = proposal.get("feature_bundle") or {}
        motion_score = bundle.get("motion_score")
        sparse_motion_score = bundle.get("sparse_motion_score")
        image_motion_score = bundle.get("image_motion_score")
        observed_score = bundle.get("observed_score")
        near_moving_points = bundle.get("near_moving_points")
        if motion_score is None or observed_score is None:
            continue
        mask = proposal["mask"]
        if int(np.count_nonzero(mask)) == 0:
            continue
        component_motion = motion_score[mask]
        component_sparse_motion = (
            sparse_motion_score[mask]
            if sparse_motion_score is not None and sparse_motion_score.shape[0] == mask.shape[0]
            else None
        )
        component_image_motion = (
            image_motion_score[mask]
            if image_motion_score is not None and image_motion_score.shape[0] == mask.shape[0]
            else None
        )
        motion_evidence_fraction = float(np.mean(component_motion >= min_motion_score))
        observed_fraction = float(np.mean(observed_score[mask] <= observed_distance))
        near_moving_fraction = (
            float(np.mean(near_moving_points[mask]))
            if near_moving_points is not None and near_moving_points.shape[0] == mask.shape[0]
            else 0.0
        )
        seed_vertex_count = int(proposal.get("seed_vertex_count", 0))
        seed_fraction = float(proposal.get("seed_fraction_of_candidate", 0.0))
        has_seed = seed_vertex_count >= min_seed_vertices
        attached_to_body = float(proposal.get("boundary_fraction", 0.0)) > 0.0
        proposal = dict(proposal)
        proposal.pop("feature_bundle", None)
        proposal.pop("seed_mask", None)
        proposal.update(
            {
                "motion_score_mean": float(np.mean(component_motion)),
                "motion_score_median": float(np.median(component_motion)),
                "sparse_motion_score_mean": (
                    float(np.mean(component_sparse_motion)) if component_sparse_motion is not None else None
                ),
                "image_motion_score_mean": (
                    float(np.mean(component_image_motion)) if component_image_motion is not None else None
                ),
                "image_motion_evidence_fraction": (
                    float(np.mean(component_image_motion >= float(getattr(args, "segmentation_image_motion_min_score", 0.05))))
                    if component_image_motion is not None
                    else None
                ),
                "motion_evidence_fraction": motion_evidence_fraction,
                "observed_fraction": observed_fraction,
                "near_moving_point_fraction": near_moving_fraction,
                "seed_vertex_count": seed_vertex_count,
                "seed_fraction_of_candidate": seed_fraction,
                "attached_to_body": attached_to_body,
            }
        )
        image_motion_evidence_fraction = proposal.get("image_motion_evidence_fraction")
        if (
            evidence_source == "image_diff"
            and proposal.get("source") == "geometry_supported_by_moving_evidence"
            and image_motion_evidence_fraction is not None
            and image_motion_evidence_fraction < min_image_geometry_fraction
        ):
            proposal["score"] = -1.0e9
            proposal["rejected_reason"] = "below_min_image_motion_geometry_fraction"
            scored.append(proposal)
            continue
        if motion_evidence_fraction < min_evidence_fraction and not has_seed:
            proposal["score"] = -1.0e9
            proposal["rejected_reason"] = "below_min_motion_evidence_fraction"
            scored.append(proposal)
            continue
        if observed_fraction < min_observed_fraction:
            proposal["score"] = -1.0e9
            proposal["rejected_reason"] = "below_min_observed_fraction"
            scored.append(proposal)
            continue

        neck_score = 1.0 - min(1.0, float(proposal["boundary_fraction"]) / max_boundary)
        size_penalty = 0.10 * float(proposal["vertex_fraction"])
        attachment_penalty = 0.0
        if relation_prior == "articulated" and not attached_to_body:
            attachment_penalty = 0.30
            if not has_seed and near_moving_fraction < min_evidence_fraction:
                proposal["score"] = -1.0e9
                proposal["rejected_reason"] = "articulated_prior_rejects_unsupported_disconnected_candidate"
                scored.append(proposal)
                continue
        source_bonus = {
            "moving_score_component": 0.05,
            "moving_score_component_grown_rings": 0.18,
            "moving_score_component_relaxed_component": 0.22,
            "moving_point_supported_component": 0.08,
            "moving_point_supported_component_grown_rings": 0.22,
            "moving_point_supported_component_relaxed_component": 0.26,
            "geometry_supported_by_moving_evidence": 0.05,
        }.get(proposal["source"], 0.0)
        raw_seed_penalty = (
            raw_seed_penalty_value
            if proposal["source"] in {"moving_score_component", "moving_point_supported_component"}
            else 0.0
        )
        geometry_penalty = (
            image_geometry_penalty
            if evidence_source == "image_diff" and proposal.get("source") == "geometry_supported_by_moving_evidence"
            else 0.0
        )
        seed_support = min(1.0, seed_vertex_count / max(1.0, int(np.count_nonzero(mask)) * min_evidence_fraction))
        score = (
            source_bonus
            + 4.0 * float(np.mean(component_motion))
            + 0.80 * motion_evidence_fraction
            + 0.20 * observed_fraction
            + 0.25 * near_moving_fraction
            + 0.35 * seed_support
            + 0.10 * min(1.0, seed_fraction * 10.0)
            + 0.10 * neck_score
            - size_penalty
            - attachment_penalty
            - raw_seed_penalty
            - geometry_penalty
        )
        proposal["score"] = float(score)
        proposal["base_score"] = float(score)
        proposal["neck_score"] = float(neck_score)
        proposal["attachment_penalty"] = float(attachment_penalty)
        proposal["raw_seed_penalty"] = float(raw_seed_penalty)
        proposal["geometry_penalty"] = float(geometry_penalty)
        scored.append(proposal)

    return sorted(scored, key=lambda item: (float(item.get("score", -1.0e9)), -float(item.get("vertex_fraction", 1.0))), reverse=True)


def _rigid_fit_improvement(
    source_points: "np.ndarray",
    target_points: "np.ndarray",
    scale: float,
    args: argparse.Namespace,
) -> Optional[dict]:
    import numpy as np

    max_source = int(getattr(args, "segmentation_moving_two_body_max_source_points", 1200))
    max_target = int(getattr(args, "segmentation_moving_two_body_max_target_points", 8000))
    min_correspondences = max(3, int(getattr(args, "segmentation_moving_two_body_min_correspondences", 20)))
    trim_fraction = float(getattr(args, "segmentation_moving_two_body_trim_fraction", 0.35))
    if not 0.0 < trim_fraction <= 1.0:
        trim_fraction = 0.35

    source = _subsample_points(source_points, max_source)
    target = _subsample_points(target_points, max_target)
    if source.shape[0] < min_correspondences or target.shape[0] < min_correspondences:
        return None

    nn_indices, before_distances = _nearest_neighbors(source, target)
    candidate_indices = np.flatnonzero(np.isfinite(before_distances))
    if candidate_indices.shape[0] < min_correspondences:
        return None

    ordered = candidate_indices[np.argsort(before_distances[candidate_indices])]
    keep_count = int(np.ceil(ordered.shape[0] * trim_fraction))
    keep_count = min(ordered.shape[0], max(min_correspondences, keep_count))
    kept = ordered[:keep_count]
    if kept.shape[0] < min_correspondences:
        return None

    transform = _estimate_rigid_transform(source[kept], target[nn_indices[kept]])
    transformed_source = _apply_transform(source, transform)
    after_distances = _nearest_distances(transformed_source, target)
    safe_scale = max(float(scale), 1.0e-12)
    before_median = float(np.median(before_distances) / safe_scale)
    after_median = float(np.median(after_distances) / safe_scale)
    before_trimmed = float(np.mean(before_distances[kept]) / safe_scale)
    after_kept = _nearest_distances(transformed_source[kept], target)
    after_trimmed = float(np.mean(after_kept) / safe_scale)
    return {
        "source_points": int(source.shape[0]),
        "target_points": int(target.shape[0]),
        "correspondences": int(kept.shape[0]),
        "trim_fraction": trim_fraction,
        "before_median": before_median,
        "after_median": after_median,
        "median_improvement": float(before_median - after_median),
        "before_trimmed_mean": before_trimmed,
        "after_trimmed_mean": after_trimmed,
        "trimmed_mean_improvement": float(before_trimmed - after_trimmed),
        "translation_norm": float(np.linalg.norm(transform[:3, 3])),
        "rotation_deg": _rotation_angle_deg(transform[:3, :3]),
    }


def _rescore_moving_evidence_proposals_with_two_body_fit(
    scored: List[dict],
    mesh: "trimesh.Trimesh",
    state_reports: List[dict],
    output_dir: str,
    state_infos: List[dict],
    scale: float,
    moving_points: "np.ndarray",
    args: argparse.Namespace,
) -> Tuple[List[dict], dict]:
    import numpy as np

    enabled = bool(getattr(args, "segmentation_moving_two_body_scoring", True))
    diagnostics = {
        "enabled": enabled,
        "evaluated_candidate_count": 0,
        "max_candidates": int(getattr(args, "segmentation_moving_two_body_max_candidates", 30)),
        "weight": float(getattr(args, "segmentation_moving_two_body_weight", 1.5)),
        "min_improvement": float(getattr(args, "segmentation_moving_two_body_min_improvement", 0.0)),
    }
    if not enabled or not scored:
        return scored, diagnostics

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    report_by_state = {report.get("state_id"): report for report in state_reports}
    usable_infos = [info for info in state_infos if info.get("state_id") in report_by_state]
    max_candidates = max(0, int(diagnostics["max_candidates"]))
    weight = float(diagnostics["weight"])
    min_improvement = float(diagnostics["min_improvement"])
    evaluated = 0
    output = []

    for proposal_index, proposal in enumerate(scored):
        proposal = dict(proposal)
        if proposal_index >= max_candidates or float(proposal.get("score", -1.0e9)) <= -1.0e8:
            output.append(proposal)
            continue

        mask = np.asarray(proposal.get("mask"), dtype=bool)
        source_points = vertices[mask]
        if source_points.shape[0] < max(3, int(getattr(args, "segmentation_moving_two_body_min_correspondences", 20))):
            proposal["two_body_fit"] = {"status": "skipped", "reason": "candidate_too_small"}
            output.append(proposal)
            continue

        state_results = []
        improvements = []
        for info in usable_infos:
            report = report_by_state.get(info.get("state_id"))
            if report is None:
                continue
            try:
                canonical_points, _, _ = _load_canonical_state_points(
                    report,
                    output_dir,
                    int(args.segmentation_max_points_per_state),
                )
            except Exception as exc:
                state_results.append({"state_id": info.get("state_id"), "status": "failed", "reason": str(exc)})
                continue
            fit = _rigid_fit_improvement(source_points, canonical_points, scale, args)
            if fit is None:
                state_results.append({"state_id": info.get("state_id"), "status": "skipped", "reason": "not_enough_points"})
                continue
            fit["state_id"] = info.get("state_id")
            fit["status"] = "ok"
            state_results.append(fit)
            improvements.append(float(fit["median_improvement"]))

        moving_fit = None
        if moving_points.shape[0] > 0:
            moving_fit = _rigid_fit_improvement(source_points, moving_points, scale, args)

        if improvements:
            mean_improvement = float(np.mean(improvements))
            median_improvement = float(np.median(improvements))
            positive_fraction = float(np.mean(np.asarray(improvements, dtype=np.float64) > 0.0))
        else:
            mean_improvement = 0.0
            median_improvement = 0.0
            positive_fraction = 0.0

        moving_improvement = float(moving_fit["median_improvement"]) if moving_fit is not None else 0.0
        combined_improvement = max(mean_improvement, 0.0) + 0.5 * max(moving_improvement, 0.0)
        proposal["two_body_fit"] = {
            "status": "ok" if improvements or moving_fit is not None else "skipped",
            "state_count": int(len(improvements)),
            "mean_median_improvement": mean_improvement,
            "median_median_improvement": median_improvement,
            "positive_state_fraction": positive_fraction,
            "moving_points_fit": moving_fit,
            "states": state_results[: int(getattr(args, "segmentation_moving_two_body_report_states", 12))],
            "states_truncated": len(state_results)
            > int(getattr(args, "segmentation_moving_two_body_report_states", 12)),
        }
        proposal["score"] = float(proposal.get("score", -1.0e9)) + weight * combined_improvement
        if combined_improvement < min_improvement:
            proposal["score"] = -1.0e9
            proposal["rejected_reason"] = "below_min_two_body_improvement"
        evaluated += 1
        output.append(proposal)

    diagnostics["evaluated_candidate_count"] = int(evaluated)
    return (
        sorted(output, key=lambda item: (float(item.get("score", -1.0e9)), -float(item.get("vertex_fraction", 1.0))), reverse=True),
        diagnostics,
    )


def _export_moving_evidence_mesh_distance_diagnostics(
    mesh: "trimesh.Trimesh",
    moving_points: "np.ndarray",
    output_dir: str,
    scale: float,
    args: argparse.Namespace,
) -> dict:
    import numpy as np

    diagnostics = {
        "enabled": True,
        "point_count": int(moving_points.shape[0]),
        "distance_threshold_ratio": float(getattr(args, "segmentation_diagnostic_mesh_near_distance", 0.05)),
        "distance_threshold": None,
        "near_point_count": 0,
        "far_point_count": 0,
        "near_point_fraction": 0.0,
        "min_distance": None,
        "median_distance": None,
        "p90_distance": None,
        "max_distance": None,
        "near_points_path": None,
        "far_points_path": None,
    }
    if moving_points.shape[0] == 0:
        diagnostics["status"] = "no_moving_points"
        return diagnostics

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.shape[0] == 0:
        diagnostics["status"] = "empty_mesh"
        return diagnostics

    threshold = max(float(scale), 1.0e-12) * float(diagnostics["distance_threshold_ratio"])
    distances = _nearest_distances(moving_points, vertices)
    near_mask = distances <= threshold
    near_points = moving_points[near_mask]
    far_points = moving_points[~near_mask]
    diagnostics.update(
        {
            "status": "ok",
            "distance_threshold": threshold,
            "near_point_count": int(near_points.shape[0]),
            "far_point_count": int(far_points.shape[0]),
            "near_point_fraction": float(near_points.shape[0] / max(moving_points.shape[0], 1)),
            "min_distance": float(np.min(distances)) if distances.shape[0] else None,
            "median_distance": float(np.median(distances)) if distances.shape[0] else None,
            "p90_distance": float(np.percentile(distances, 90.0)) if distances.shape[0] else None,
            "max_distance": float(np.max(distances)) if distances.shape[0] else None,
        }
    )

    if not bool(getattr(args, "check_only", False)):
        diag_dir = os.path.join(output_dir, "segmentation", "diagnostics")
        if near_points.shape[0] > 0:
            near_path = os.path.join(diag_dir, "moving_evidence_mesh_near_points.ply")
            _write_points_ply(near_path, near_points, (120, 255, 120))
            diagnostics["near_points_path"] = near_path
        if far_points.shape[0] > 0:
            far_path = os.path.join(diag_dir, "moving_evidence_mesh_far_points.ply")
            _write_points_ply(far_path, far_points, (255, 80, 80))
            diagnostics["far_points_path"] = far_path

    return diagnostics


def _export_scored_segmentation_proposals(
    mesh: "trimesh.Trimesh",
    scored: List[dict],
    output_dir: str,
    args: argparse.Namespace,
) -> dict:
    import numpy as np

    max_exports = max(0, int(getattr(args, "segmentation_diagnostic_top_proposals", 20)))
    diagnostics = {
        "enabled": max_exports > 0,
        "max_exports": int(max_exports),
        "output_dir": None,
        "exports": [],
    }
    if max_exports <= 0 or bool(getattr(args, "check_only", False)):
        return diagnostics

    proposal_dir = os.path.join(output_dir, "segmentation", "proposals")
    os.makedirs(proposal_dir, exist_ok=True)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    for rank, proposal in enumerate(scored[:max_exports], start=1):
        mask = np.asarray(proposal.get("mask"), dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != len(mesh.vertices) or faces.shape[0] == 0:
            continue
        face_indices = np.flatnonzero(np.any(mask[faces], axis=1))
        source = _safe_filename_token(proposal.get("source"))
        score = float(proposal.get("score", -1.0e9))
        path = os.path.join(proposal_dir, f"rank_{rank:02d}_{source}.obj")
        export_info = _export_mesh_faces(mesh, face_indices, path)
        proposal["diagnostic_mesh_path"] = export_info.get("path")
        proposal["diagnostic_face_count"] = export_info.get("face_count")
        export_summary = _proposal_for_json(proposal)
        export_summary.update(
            {
                "rank": int(rank),
                "score": score,
                "mesh_path": export_info.get("path"),
                "face_count": export_info.get("face_count"),
                "warnings": export_info.get("warnings", []),
            }
        )
        diagnostics["exports"].append(export_summary)

    diagnostics["output_dir"] = proposal_dir
    _write_json(os.path.join(proposal_dir, "proposals.json"), diagnostics)
    return diagnostics


def _segment_vertices_by_moving_evidence(
    mesh: "trimesh.Trimesh",
    state_reports: List[dict],
    output_dir: str,
    part_count: int,
    args: argparse.Namespace,
) -> Tuple["np.ndarray", dict]:
    import numpy as np

    if part_count < 2:
        raise ValueError("moving evidence segmentation needs at least two planned part slots.")

    residual_features, state_infos, scale = _collect_motion_residual_features(mesh, state_reports, output_dir, args)
    if residual_features.shape[1] < max(2, int(args.segmentation_min_states)):
        raise ValueError(
            f"moving evidence segmentation needs {max(2, int(args.segmentation_min_states))}+ usable states; "
            f"found {residual_features.shape[1]}."
        )

    moving_points, moving_points_path, moving_point_diagnostics = _collect_moving_evidence_points(
        state_reports,
        output_dir,
        args,
        scale,
        mesh_vertices=np.asarray(mesh.vertices, dtype=np.float64),
    )
    if scale > 1.0e-12 and moving_points.shape[0] > 0:
        moving_points_for_scoring = moving_points
    else:
        moving_points_for_scoring = np.zeros((0, 3), dtype=np.float64)
    moving_mesh_diagnostics = _export_moving_evidence_mesh_distance_diagnostics(
        mesh,
        moving_points_for_scoring,
        output_dir,
        scale,
        args,
    )

    image_motion_features = None
    image_motion_diagnostics = {"enabled": False}
    evidence_source = getattr(args, "segmentation_moving_evidence_source", "sparse")
    if evidence_source in {"image_diff", "combined"}:
        image_motion_features, image_motion_diagnostics = _collect_image_motion_vertex_features(
            mesh,
            state_reports,
            output_dir,
            args,
        )

    proposals, proposal_diagnostics = _build_moving_evidence_proposals(
        mesh,
        residual_features,
        moving_points_for_scoring,
        args,
        image_motion_features=image_motion_features,
    )
    scored = _score_moving_evidence_proposals(proposals, args)
    scored, two_body_diagnostics = _rescore_moving_evidence_proposals_with_two_body_fit(
        scored,
        mesh,
        state_reports,
        output_dir,
        state_infos,
        scale,
        moving_points_for_scoring,
        args,
    )
    proposal_export_diagnostics = _export_scored_segmentation_proposals(mesh, scored, output_dir, args)
    selected = _select_geometry_part_proposals(scored, part_count - 1, args)
    if len(selected) < part_count - 1:
        raise ValueError(
            f"moving evidence segmentation needs {part_count - 1} accepted motion-supported proposals; "
            f"found {len(selected)}."
        )

    vertex_count = int(len(mesh.vertices))
    vertex_labels = np.zeros((vertex_count,), dtype=np.int64)
    for part_index, proposal in enumerate(selected, start=1):
        vertex_labels[proposal["mask"]] = part_index

    diagnostics = {
        "method": "moving_evidence",
        "state_count": len(state_infos),
        "states": state_infos,
        "mesh_scale": scale,
        "feature_shape": [int(residual_features.shape[0]), int(residual_features.shape[1])],
        "moving_points": moving_point_diagnostics,
        "moving_points_path": moving_points_path,
        "moving_points_mesh_distance": moving_mesh_diagnostics,
        "image_motion": image_motion_diagnostics,
        "proposal_generation": proposal_diagnostics,
        "proposal_exports": proposal_export_diagnostics,
        "two_body_scoring": two_body_diagnostics,
        "selection": {
            "required_proposals": int(part_count - 1),
            "accepted_proposals": [_proposal_for_json(proposal) for proposal in selected],
            "top_scored_proposals": [_proposal_for_json(proposal) for proposal in scored[:20]],
        },
    }
    return vertex_labels, diagnostics


def _select_segmentation_reference_state(state_reports: List[dict], output_dir: str, args: argparse.Namespace) -> Optional[dict]:
    if args.segmentation_reference_state:
        for report in state_reports:
            if report["state_id"] == args.segmentation_reference_state:
                return report
        return None
    for report in state_reports:
        if _usable_state_for_segmentation(report, output_dir):
            return report
    return None


def _segment_vertices_by_part_masks(
    mesh: "trimesh.Trimesh",
    state_reports: List[dict],
    output_dir: str,
    part_plan: dict,
    args: argparse.Namespace,
    session_path: str,
) -> Tuple["np.ndarray", dict]:
    import numpy as np

    reference = _select_segmentation_reference_state(state_reports, output_dir, args)
    if reference is None:
        raise ValueError("no usable reference state for mask-based mesh segmentation.")

    source_points, source_colors, source_info = _load_canonical_state_points(
        reference,
        output_dir,
        int(args.segmentation_max_points_per_state),
    )
    world_T_object = _to_4x4_transform((reference.get("registration") or {})["T_world_object"], "T_world_object")
    raw_points, raw_colors = _read_ascii_ply_points(_find_state_pointcloud(reference, output_dir))
    raw_points, raw_colors = _subsample_points_and_colors(raw_points, raw_colors, int(args.segmentation_max_points_per_state))
    bundle, camera_warnings = _load_state_camera_bundle(reference["path"], reference["undistorted_cameras"])

    evidence = []
    evidence_infos = []
    for part in part_plan.get("parts", []):
        mask_paths, search_dirs = _find_part_mask_paths(args, reference, part, bundle, session_path)
        if not mask_paths:
            evidence_infos.append(
                {
                    "part_id": part["part_id"],
                    "status": "missing_masks",
                    "mask_paths": {},
                    "mask_search_dirs": search_dirs,
                }
            )
            continue
        masked_points, _, mask_info = _filter_points_by_masks(raw_points, raw_colors, bundle, mask_paths, args)
        if masked_points.shape[0] < max(3, int(args.part_mask_min_points)):
            evidence_infos.append(
                {
                    "part_id": part["part_id"],
                    "status": "too_few_points",
                    "mask_paths": mask_paths,
                    "mask_filter": mask_info,
                }
            )
            continue
        canonical_points = _apply_transform(masked_points, _invert_transform(world_T_object))
        evidence.append((part["part_id"], canonical_points))
        evidence_infos.append(
            {
                "part_id": part["part_id"],
                "status": "ok",
                "mask_paths": mask_paths,
                "canonical_point_count": int(canonical_points.shape[0]),
                "mask_filter": mask_info,
            }
        )

    part_count = len(part_plan.get("parts", []))
    if len(evidence) < part_count:
        raise ValueError(f"mask-based segmentation needs evidence for all {part_count} parts; found {len(evidence)}.")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    distance_columns = []
    evidence_part_indices = []
    part_index_by_id = {part["part_id"]: idx for idx, part in enumerate(part_plan.get("parts", []))}
    for part_id, canonical_points in evidence:
        distance_columns.append(_nearest_distances(vertices, canonical_points))
        evidence_part_indices.append(part_index_by_id[part_id])
    distances = np.stack(distance_columns, axis=1)
    nearest_evidence = np.argmin(distances, axis=1)
    vertex_labels = np.asarray([evidence_part_indices[int(idx)] for idx in nearest_evidence], dtype=np.int64)
    min_distances = np.min(distances, axis=1)

    diagnostics = {
        "method": "mask_points",
        "reference_state_id": reference["state_id"],
        "source": source_info,
        "camera_warnings": camera_warnings,
        "part_evidence": evidence_infos,
        "mean_nearest_evidence_distance": float(np.mean(min_distances)),
        "median_nearest_evidence_distance": float(np.median(min_distances)),
    }
    return vertex_labels, diagnostics


def _write_vertex_labels_csv(path: str, vertex_labels: "np.ndarray", part_plan: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part_ids = _segmentation_part_ids(part_plan)
    with open(path, "w", encoding="utf-8") as f:
        f.write("vertex_index,part_index,part_id\n")
        for idx, part_index in enumerate(vertex_labels):
            f.write(f"{idx},{int(part_index)},{part_ids[int(part_index)]}\n")


def _apply_segmentation_to_part_plan(part_plan: dict, segmentation_summary: dict) -> None:
    part_outputs = {part["part_id"]: part for part in segmentation_summary.get("parts", [])}
    for part in part_plan.get("parts", []):
        output = part_outputs.get(part["part_id"])
        if not output or not output.get("mesh_path"):
            continue
        part["segmented_mesh_path"] = output["mesh_path"]
        part["segmentation"] = {
            "status": output.get("status"),
            "vertex_count": output.get("vertex_count"),
            "face_count": output.get("face_count"),
        }
        if not part.get("mesh_path") or part.get("mesh_source") == "pending_segmentation":
            part["mesh_path"] = output["mesh_path"]
            part["mesh_source"] = "segmented_input_mesh"
            part["status"] = "ready"


def _part_plan_has_all_provided_meshes(part_plan: dict) -> bool:
    parts = part_plan.get("parts", [])
    if not parts:
        return False
    for part in parts:
        mesh_path = part.get("mesh_path")
        if part.get("mesh_source") != "provided_part_mesh":
            return False
        if not mesh_path or not os.path.exists(mesh_path):
            return False
    return True


def _provided_part_outputs(part_plan: dict) -> List[dict]:
    outputs = []
    for part in part_plan.get("parts", []):
        mesh_path = part.get("mesh_path")
        outputs.append(
            {
                "part_id": part["part_id"],
                "label": part.get("label"),
                "status": "ready" if mesh_path and os.path.exists(mesh_path) else "missing",
                "mesh_path": mesh_path,
                "mesh_source": part.get("mesh_source"),
                "vertex_count": None,
                "vertex_fraction": None,
                "face_count": None,
                "warnings": [],
            }
        )
    return outputs


def _debug_part_palette(part_count: int) -> "np.ndarray":
    import numpy as np

    base = np.asarray(
        [
            [80, 150, 255],
            [255, 120, 80],
            [80, 210, 130],
            [220, 130, 255],
            [255, 200, 70],
            [80, 210, 220],
            [230, 90, 150],
            [150, 190, 80],
            [170, 150, 255],
            [255, 150, 140],
        ],
        dtype=np.uint8,
    )
    if part_count <= base.shape[0]:
        return base[:part_count].copy()
    repeats = int(np.ceil(part_count / base.shape[0]))
    return np.tile(base, (repeats, 1))[:part_count].copy()


def _segmentation_part_display_names(part_plan: dict) -> List[str]:
    names = []
    for part in part_plan.get("parts", []):
        label = part.get("label") or part.get("part_id")
        names.append(str(label))
    return names


def _write_segmentation_side_by_side_mesh(
    mesh: "trimesh.Trimesh",
    face_labels: "np.ndarray",
    part_count: int,
    path: str,
) -> dict:
    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return {"path": None, "warnings": ["empty mesh."]}

    colors = _debug_part_palette(part_count)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    width = float(bounds[1, 0] - bounds[0, 0]) if bounds.shape == (2, 3) else 1.0
    diagonal = _mesh_diagonal_from_vertices(vertices)
    spacing = max(width, diagonal * 0.65, 1.0e-6) * 1.35
    center = np.mean(bounds, axis=0) if bounds.shape == (2, 3) else np.zeros(3, dtype=np.float64)

    out_vertices = []
    out_faces = []
    out_colors = []
    for part_index in range(part_count):
        part_faces = np.flatnonzero(face_labels == part_index)
        if part_faces.shape[0] == 0:
            continue
        used = np.unique(faces[part_faces].reshape(-1))
        remap = {int(old): idx for idx, old in enumerate(used.tolist())}
        part_vertices = vertices[used].copy()
        offset = np.asarray([(part_index - (part_count - 1) * 0.5) * spacing, 0.0, 0.0], dtype=np.float64)
        part_vertices = part_vertices - center.reshape(1, 3) + offset.reshape(1, 3)
        start = sum(item.shape[0] for item in out_vertices)
        remapped_faces = np.asarray(
            [[start + remap[int(value)] for value in face] for face in faces[part_faces]],
            dtype=np.int64,
        )
        out_vertices.append(part_vertices)
        out_faces.append(remapped_faces)
        out_colors.append(np.tile(colors[part_index].reshape(1, 3), (part_vertices.shape[0], 1)))

    if not out_vertices:
        return {"path": None, "warnings": ["no faces assigned to segmentation parts."]}
    all_vertices = np.concatenate(out_vertices, axis=0)
    all_faces = np.concatenate(out_faces, axis=0)
    all_colors = np.concatenate(out_colors, axis=0)
    return _write_colored_mesh_ply(path, all_vertices, all_faces, all_colors)


def _render_segmented_parts_overview_png(
    mesh: "trimesh.Trimesh",
    vertex_labels: "np.ndarray",
    face_labels: "np.ndarray",
    part_plan: dict,
    part_outputs: List[dict],
    view_name: str,
    path: str,
) -> Optional[str]:
    import cv2
    import numpy as np

    views = {
        "xy": ((0, 1), 2, "XY"),
        "xz": ((0, 2), 1, "XZ"),
        "yz": ((1, 2), 0, "YZ"),
    }
    if view_name not in views:
        return None
    axes, depth_axis, title = views[view_name]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return None

    part_count = len(part_plan.get("parts", []))
    colors = _debug_part_palette(part_count)
    names = _segmentation_part_display_names(part_plan)
    projected = vertices[:, list(axes)]
    lo = np.min(projected, axis=0)
    hi = np.max(projected, axis=0)
    span = np.maximum(hi - lo, 1.0e-9)

    panel_w = 360
    panel_h = 360
    label_h = 70
    margin = 22
    panels = part_count + 1
    image = np.full((panel_h + label_h, panel_w * panels, 3), 248, dtype=np.uint8)

    def project_panel(points: "np.ndarray", panel_index: int) -> "np.ndarray":
        uv = points[:, list(axes)]
        scale = min((panel_w - 2 * margin) / span[0], (panel_h - 2 * margin) / span[1])
        xy = np.zeros((uv.shape[0], 2), dtype=np.int32)
        xy[:, 0] = np.rint((uv[:, 0] - lo[0]) * scale + panel_index * panel_w + margin).astype(np.int32)
        xy[:, 1] = np.rint(panel_h - ((uv[:, 1] - lo[1]) * scale + margin)).astype(np.int32)
        return xy

    def draw_panel(panel_index: int, part_index: Optional[int]) -> None:
        x0 = panel_index * panel_w
        x1 = x0 + panel_w - 1
        cv2.rectangle(image, (x0, 0), (x1, panel_h - 1), (225, 225, 225), 1)
        if part_index is None:
            face_indices = np.arange(faces.shape[0], dtype=np.int64)
            label = "all parts"
            color_lookup = face_labels
            vertex_count = int(vertex_labels.shape[0])
            face_count = int(faces.shape[0])
        else:
            face_indices = np.flatnonzero(face_labels == part_index)
            label = names[part_index] if part_index < len(names) else f"part_{part_index}"
            color_lookup = np.full((faces.shape[0],), part_index, dtype=np.int64)
            vertex_count = int(np.count_nonzero(vertex_labels == part_index))
            face_count = int(face_indices.shape[0])
        if face_indices.shape[0] > 0:
            depths = np.mean(vertices[faces[face_indices], depth_axis], axis=1)
            ordered = face_indices[np.argsort(depths)]
            xy = project_panel(vertices, panel_index)
            for face_idx in ordered:
                label_index = int(color_lookup[face_idx])
                if label_index < 0 or label_index >= colors.shape[0]:
                    color = np.asarray([160, 160, 160], dtype=np.uint8)
                else:
                    color = colors[label_index]
                triangle = xy[faces[face_idx]].reshape((-1, 1, 2))
                bgr = tuple(int(value) for value in color[::-1])
                cv2.fillPoly(image, [triangle], bgr)
                cv2.polylines(image, [triangle], True, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(
            image,
            label[:32],
            (x0 + 12, panel_h + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"v={vertex_count} f={face_count}",
            (x0 + 12, panel_h + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (75, 75, 75),
            1,
            cv2.LINE_AA,
        )
        if part_index is not None and part_index < len(part_outputs):
            fraction = part_outputs[part_index].get("vertex_fraction")
            if fraction is not None:
                cv2.putText(
                    image,
                    f"{100.0 * float(fraction):.1f}%",
                    (x0 + panel_w - 72, panel_h + 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (75, 75, 75),
                    1,
                    cv2.LINE_AA,
                )

    draw_panel(0, None)
    for part_index in range(part_count):
        draw_panel(part_index + 1, part_index)
    cv2.putText(image, f"segmentation overview {title}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if cv2.imwrite(path, image):
        return path
    return None


def _write_segmentation_visual_diagnostics(
    mesh: "trimesh.Trimesh",
    vertex_labels: "np.ndarray",
    face_labels: "np.ndarray",
    part_plan: dict,
    part_outputs: List[dict],
    seg_dir: str,
) -> dict:
    import numpy as np

    diagnostics = {"output_dir": os.path.join(seg_dir, "diagnostics"), "paths": {}, "warnings": []}
    diag_dir = diagnostics["output_dir"]
    os.makedirs(diag_dir, exist_ok=True)

    part_count = len(part_plan.get("parts", []))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors = _debug_part_palette(part_count)
    vertex_colors = colors[np.clip(np.asarray(vertex_labels, dtype=np.int64), 0, max(part_count - 1, 0))]

    colored_path = os.path.join(diag_dir, "segmented_mesh_colored.ply")
    diagnostics["paths"]["segmented_mesh_colored_ply"] = _write_colored_mesh_ply(
        colored_path,
        vertices,
        faces,
        vertex_colors,
    ).get("path")

    side_by_side_path = os.path.join(diag_dir, "segmented_parts_side_by_side.ply")
    side_info = _write_segmentation_side_by_side_mesh(mesh, face_labels, part_count, side_by_side_path)
    diagnostics["paths"]["segmented_parts_side_by_side_ply"] = side_info.get("path")
    diagnostics["warnings"].extend(side_info.get("warnings", []))

    overview_paths = {}
    for view_name in ("xy", "xz", "yz"):
        out_path = os.path.join(diag_dir, f"segmented_parts_overview_{view_name}.png")
        rendered = _render_segmented_parts_overview_png(
            mesh,
            vertex_labels,
            face_labels,
            part_plan,
            part_outputs,
            view_name,
            out_path,
        )
        overview_paths[view_name] = rendered
    diagnostics["paths"]["overview_pngs"] = overview_paths

    summary_csv = os.path.join(diag_dir, "segmentation_part_summary.csv")
    _write_csv_rows(
        summary_csv,
        [
            "part_id",
            "label",
            "status",
            "vertex_count",
            "vertex_fraction",
            "face_count",
            "component_count",
            "largest_component_vertices",
            "largest_component_fraction",
            "mesh_path",
            "warnings",
        ],
        part_outputs,
    )
    diagnostics["paths"]["part_summary_csv"] = summary_csv
    return diagnostics


def _segment_mesh_parts(
    state_reports: List[dict],
    output_dir: str,
    mesh_info: dict,
    part_plan: dict,
    args: argparse.Namespace,
    session_path: str,
) -> dict:
    import numpy as np

    seg_dir = os.path.join(output_dir, "segmentation")
    parts_dir = os.path.join(seg_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    part_count = len(part_plan.get("parts", []))
    summary = {
        "schema": "paradex.articulated_object.mesh_segmentation.v0",
        "status": "skipped",
        "method": args.segmentation_method,
        "output_dir": seg_dir,
        "parts_dir": parts_dir,
        "input_mesh_path": mesh_info.get("mesh_path"),
        "part_count": part_count,
        "vertex_labels_path": None,
        "parts": [],
        "diagnostics": {},
        "warnings": [],
    }

    if args.check_only or args.skip_segmentation:
        _write_json(os.path.join(seg_dir, "segmentation.json"), summary)
        return summary
    if _part_plan_has_all_provided_meshes(part_plan) and not args.force_segmentation_with_provided_parts:
        summary["status"] = "provided_part_meshes"
        summary["method"] = "provided_part_meshes"
        summary["parts"] = _provided_part_outputs(part_plan)
        summary["warnings"].append(
            "segmentation skipped because every planned part slot already has a provided mesh."
        )
        _write_json(os.path.join(seg_dir, "segmentation.json"), summary)
        print(f"[SEG] status={summary['status']}, method={summary['method']}, parts={len(summary.get('parts', []))}")
        return summary
    if part_count < 2:
        summary["warnings"].append("mesh segmentation needs at least two planned part slots.")
        _write_json(os.path.join(seg_dir, "segmentation.json"), summary)
        return summary

    mesh_path = mesh_info.get("mesh_path")
    if not mesh_path or not os.path.exists(mesh_path):
        summary["status"] = "failed"
        summary["warnings"].append("input mesh is missing.")
        _write_json(os.path.join(seg_dir, "segmentation.json"), summary)
        return summary

    try:
        mesh = _load_mesh_geometry(mesh_path)
        method = args.segmentation_method
        if method == "auto":
            try:
                vertex_labels, diagnostics = _segment_vertices_by_part_masks(
                    mesh,
                    state_reports,
                    output_dir,
                    part_plan,
                    args,
                    session_path,
                )
                method = "mask_points"
            except Exception as exc:
                summary["warnings"].append(f"mask_points segmentation unavailable; falling back to moving_evidence: {exc}")
                try:
                    vertex_labels, diagnostics = _segment_vertices_by_moving_evidence(
                        mesh,
                        state_reports,
                        output_dir,
                        part_count,
                        args,
                    )
                    method = "moving_evidence"
                except Exception as moving_exc:
                    summary["warnings"].append(
                        f"moving_evidence segmentation unavailable; falling back to motion_residual: {moving_exc}"
                    )
                    vertex_labels, diagnostics = _segment_vertices_by_motion_residual(
                        mesh,
                        state_reports,
                        output_dir,
                        part_count,
                        args,
                    )
                    method = "motion_residual"
        elif method == "mask_points":
            vertex_labels, diagnostics = _segment_vertices_by_part_masks(
                mesh,
                state_reports,
                output_dir,
                part_plan,
                args,
                session_path,
            )
        elif method == "moving_evidence":
            vertex_labels, diagnostics = _segment_vertices_by_moving_evidence(
                mesh,
                state_reports,
                output_dir,
                part_count,
                args,
            )
        elif method == "geometry_proposals":
            vertex_labels, diagnostics = _segment_vertices_by_geometry_proposals(
                mesh,
                state_reports,
                output_dir,
                part_count,
                args,
            )
        elif method == "motion_residual":
            vertex_labels, diagnostics = _segment_vertices_by_motion_residual(
                mesh,
                state_reports,
                output_dir,
                part_count,
                args,
            )
        else:
            raise ValueError(f"unsupported segmentation method: {method}")

        vertex_labels, cleanup_diagnostics = _cleanup_segmentation_vertex_labels(mesh, vertex_labels, part_count, args)
        component_stats = _segmentation_component_stats(mesh, vertex_labels, part_count, args)
        if not isinstance(diagnostics, dict):
            diagnostics = {"raw_diagnostics": diagnostics}
        diagnostics["label_cleanup"] = cleanup_diagnostics
        diagnostics["component_stats"] = component_stats

        face_labels = _face_labels_from_vertex_labels(np.asarray(mesh.faces, dtype=np.int64), vertex_labels, part_count)
        vertex_labels_path = os.path.join(seg_dir, "vertex_labels.csv")
        _write_vertex_labels_csv(vertex_labels_path, vertex_labels, part_plan)
        summary["vertex_labels_path"] = vertex_labels_path
        summary["method"] = method
        summary["diagnostics"] = diagnostics

        min_fraction = float(args.segmentation_min_vertex_fraction)
        part_outputs = []
        vertex_total = max(int(vertex_labels.shape[0]), 1)
        component_stats_by_index = {int(item["part_index"]): item for item in component_stats}
        for part_index, part in enumerate(part_plan.get("parts", [])):
            part_id = part["part_id"]
            face_indices = np.flatnonzero(face_labels == part_index)
            out_path = os.path.join(parts_dir, f"{part_id}.obj")
            export_info = _export_mesh_faces(mesh, face_indices, out_path)
            vertex_count = int(np.count_nonzero(vertex_labels == part_index))
            vertex_fraction = float(vertex_count / vertex_total)
            part_component_stats = component_stats_by_index.get(part_index, {})
            warnings = list(export_info.get("warnings", []))
            if vertex_fraction < min_fraction:
                warnings.append(
                    f"assigned vertex fraction {vertex_fraction:.4f} is below --segmentation-min-vertex-fraction."
                )
            part_outputs.append(
                {
                    "part_id": part_id,
                    "label": part.get("label"),
                    "status": "ready" if export_info.get("path") else "empty",
                    "mesh_path": export_info.get("path"),
                    "vertex_count": vertex_count,
                    "vertex_fraction": vertex_fraction,
                    "face_count": export_info.get("face_count", 0),
                    "component_count": part_component_stats.get("component_count"),
                    "largest_component_vertices": part_component_stats.get("largest_component_vertices"),
                    "largest_component_fraction": part_component_stats.get("largest_component_fraction"),
                    "warnings": warnings,
                }
            )

        summary["parts"] = part_outputs
        visual_diagnostics = _write_segmentation_visual_diagnostics(
            mesh,
            vertex_labels,
            face_labels,
            part_plan,
            part_outputs,
            seg_dir,
        )
        summary["visual_diagnostics"] = visual_diagnostics
        summary["diagnostics"]["visualization"] = visual_diagnostics
        ready_parts = [part for part in part_outputs if part["status"] == "ready"]
        summary["status"] = "ok" if len(ready_parts) >= 2 else "insufficient_parts"
        _apply_segmentation_to_part_plan(part_plan, summary)
    except Exception as exc:
        summary["status"] = "failed"
        summary["warnings"].append(str(exc))

    summary_path = os.path.join(seg_dir, "segmentation.json")
    _write_json(summary_path, summary)
    print(f"[SEG] status={summary['status']}, method={summary['method']}, parts={len(summary.get('parts', []))}")
    return summary


def _axis_angle_from_rotation(rotation: "np.ndarray") -> Tuple["np.ndarray", float]:
    import numpy as np

    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    angle = float(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))
    if abs(angle) < 1.0e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64), 0.0
    axis = np.asarray(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(axis)
    if norm <= 1.0e-12:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64), angle
    axis = axis / norm
    if axis[np.argmax(np.abs(axis))] < 0:
        axis *= -1.0
        angle *= -1.0
    return axis, angle


def _principal_axis(vectors: "np.ndarray") -> Tuple["np.ndarray", dict]:
    import numpy as np

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64), {"sample_count": 0}
    norms = np.linalg.norm(vectors, axis=1)
    valid = vectors[norms > 1.0e-9]
    if valid.shape[0] == 0:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64), {"sample_count": 0}
    _, singular_values, vh = np.linalg.svd(valid, full_matrices=False)
    axis = vh[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis *= -1.0
    explained = float(singular_values[0] ** 2 / max(float(np.sum(singular_values ** 2)), 1.0e-12))
    return axis, {
        "sample_count": int(valid.shape[0]),
        "singular_values": singular_values.tolist(),
        "explained_ratio": explained,
    }


def _fit_revolute_joint(observations: List[dict]) -> Tuple[dict, dict]:
    import numpy as np

    axes = []
    angles = []
    pivots_a = []
    pivots_b = []
    residuals = []
    for obs in observations:
        T = _to_4x4_transform(obs["delta_from_reference_T"], "delta_from_reference_T")
        axis, angle = _axis_angle_from_rotation(T[:3, :3])
        if abs(angle) < 1.0e-6:
            continue
        axes.append(axis * np.sign(angle if abs(angle) > 1.0e-12 else 1.0))
        angles.append(float(angle))
        A = np.eye(3, dtype=np.float64) - T[:3, :3]
        b = T[:3, 3]
        pivot, *_ = np.linalg.lstsq(A, b, rcond=None)
        pivots_a.append(pivot)
        pivots_b.append(axis)
        residuals.append(float(np.linalg.norm(A @ pivot - b)))

    axis, axis_quality = _principal_axis(np.asarray(axes, dtype=np.float64))
    pivots = np.asarray(pivots_a, dtype=np.float64) if pivots_a else np.zeros((0, 3), dtype=np.float64)
    pivot = np.mean(pivots, axis=0) if pivots.shape[0] else np.zeros((3,), dtype=np.float64)
    angle_values = np.asarray(angles, dtype=np.float64)
    if angle_values.size == 0:
        limits = [0.0, 0.0]
    else:
        limits = [float(np.min(angle_values)), float(np.max(angle_values))]
    model = {
        "type": "revolute",
        "axis_parent": axis.tolist(),
        "pivot_parent": pivot.tolist(),
        "limit": {"lower": limits[0], "upper": limits[1], "unit": "rad"},
    }
    quality = {
        "axis_quality": axis_quality,
        "pivot_sample_count": int(pivots.shape[0]),
        "pivot_std": np.std(pivots, axis=0).tolist() if pivots.shape[0] else [0.0, 0.0, 0.0],
        "mean_pivot_residual": float(np.mean(residuals)) if residuals else None,
        "angle_abs_max": float(np.max(np.abs(angle_values))) if angle_values.size else 0.0,
    }
    return model, quality


def _fit_prismatic_joint(observations: List[dict]) -> Tuple[dict, dict]:
    import numpy as np

    translations = []
    for obs in observations:
        T = _to_4x4_transform(obs["delta_from_reference_T"], "delta_from_reference_T")
        translations.append(T[:3, 3])
    translations = np.asarray(translations, dtype=np.float64)
    axis, axis_quality = _principal_axis(translations)
    offsets = translations @ axis
    if offsets.size == 0:
        limits = [0.0, 0.0]
        orthogonal_mean = 0.0
    else:
        residual = translations - offsets[:, None] * axis[None, :]
        limits = [float(np.min(offsets)), float(np.max(offsets))]
        orthogonal_mean = float(np.mean(np.linalg.norm(residual, axis=1)))
    model = {
        "type": "prismatic",
        "axis_parent": axis.tolist(),
        "limit": {"lower": limits[0], "upper": limits[1], "unit": "m"},
    }
    quality = {
        "axis_quality": axis_quality,
        "orthogonal_translation_mean": orthogonal_mean,
        "translation_abs_max": float(np.max(np.linalg.norm(translations, axis=1))) if translations.size else 0.0,
    }
    return model, quality


def _classify_joint_observations(observations: List[dict], args: argparse.Namespace) -> Tuple[str, dict]:
    import numpy as np

    rotations = np.asarray([obs["delta_from_reference"]["rotation_deg"] for obs in observations], dtype=np.float64)
    translations = np.asarray([obs["delta_from_reference"]["translation_norm"] for obs in observations], dtype=np.float64)
    max_rot = float(np.max(rotations)) if rotations.size else 0.0
    max_trans = float(np.max(translations)) if translations.size else 0.0

    if max_rot >= float(args.joint_revolute_min_rotation_deg):
        return "revolute", {"max_rotation_deg": max_rot, "max_translation": max_trans}
    if max_trans >= float(args.joint_prismatic_min_translation):
        return "prismatic", {"max_rotation_deg": max_rot, "max_translation": max_trans}
    return "fixed", {"max_rotation_deg": max_rot, "max_translation": max_trans}


def _fit_joint_for_pair(pair: dict, args: argparse.Namespace) -> dict:
    observations = pair.get("observations", [])
    result = {
        "edge_id": pair["pair_id"],
        "parent_part_id": pair["parent_part_id"],
        "child_part_id": pair["child_part_id"],
        "status": "skipped",
        "model": {"type": "unknown"},
        "quality": {},
        "observation_count": len(observations),
        "reference_state_id": pair.get("reference_state_id"),
        "warnings": [],
    }
    min_states = max(2, int(args.joint_min_states))
    if len(observations) < min_states:
        result["status"] = "insufficient_states"
        result["warnings"].append(f"{min_states}+ observations are required for joint fitting.")
        return result

    joint_type, classification = _classify_joint_observations(observations, args)
    result["classification"] = classification
    if joint_type == "revolute":
        model, quality = _fit_revolute_joint(observations)
        result["status"] = "candidate"
        result["model"] = model
        result["quality"] = quality
    elif joint_type == "prismatic":
        model, quality = _fit_prismatic_joint(observations)
        result["status"] = "candidate"
        result["model"] = model
        result["quality"] = quality
    else:
        result["status"] = "candidate"
        result["model"] = {"type": "fixed"}
        result["quality"] = classification

    if classification["max_rotation_deg"] >= float(args.joint_separable_rotation_deg) and classification[
        "max_translation"
    ] >= float(args.joint_separable_translation):
        result["separable_candidate"] = True
        result["warnings"].append("large rotation and translation both observed; treat as separable/free candidate.")
    else:
        result["separable_candidate"] = False
    return result


def _build_kinematic_model(
    output_dir: str,
    part_plan: dict,
    segmentation_summary: dict,
    motion_observations: dict,
    args: argparse.Namespace,
) -> dict:
    kin_dir = os.path.join(output_dir, "kinematic_model")
    os.makedirs(kin_dir, exist_ok=True)
    model = {
        "schema": "paradex.articulated_object.kinematic_model.v0",
        "status": "skipped",
        "output_dir": kin_dir,
        "parts": [
            {
                "part_id": part["part_id"],
                "label": part.get("label"),
                "mesh_path": part.get("mesh_path"),
                "mesh_source": part.get("mesh_source"),
                "completed_mesh_path": part.get("completed_mesh_path"),
                "completed_mesh_source": part.get("completed_mesh_source"),
                "canonical_T_part": part.get("canonical_T_part"),
            }
            for part in part_plan.get("parts", [])
        ],
        "joints": [],
        "segmentation_path": os.path.join(output_dir, "segmentation", "segmentation.json"),
        "motion_observations_path": os.path.join(output_dir, "motion_observations", "motion_observations.json"),
        "warnings": [],
    }

    if args.check_only or args.skip_joint_fitting:
        _write_json(os.path.join(kin_dir, "kinematic_model.json"), model)
        return model

    if motion_observations.get("status") not in {"ok", "insufficient_states"}:
        model["status"] = "failed"
        model["warnings"].append("motion observations are missing or unusable.")
        _write_json(os.path.join(kin_dir, "kinematic_model.json"), model)
        return model

    joints = []
    for pair in motion_observations.get("pairs", []):
        joint = _fit_joint_for_pair(pair, args)
        joints.append(joint)
        _write_json(os.path.join(kin_dir, f"{joint['edge_id']}.json"), joint)

    model["joints"] = joints
    candidate_joints = [joint for joint in joints if joint.get("status") == "candidate"]
    if candidate_joints:
        model["status"] = "candidate"
    elif joints:
        model["status"] = "insufficient_states"
    else:
        model["status"] = "no_joints"
        model["warnings"].append("no part pairs were available for joint fitting.")

    if segmentation_summary.get("status") not in {"ok", "provided_part_meshes"}:
        model["warnings"].append("segmentation did not complete cleanly; part meshes may be incomplete.")

    path = os.path.join(kin_dir, "kinematic_model.json")
    _write_json(path, model)
    print(f"[JOINT] status={model['status']}, joints={len(joints)}")
    return model


def _write_state_manifests(output_dir: str, state_reports: List[dict]) -> List[str]:
    state_manifest_dir = os.path.join(output_dir, "states")
    os.makedirs(state_manifest_dir, exist_ok=True)
    paths = []
    for report in state_reports:
        path = os.path.join(state_manifest_dir, f"{report['state_id']}.json")
        _write_json(path, report)
        paths.append(path)
    return paths


def _identity4() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _resolve_optional_input_path(raw_path: Optional[str], session_path: str) -> Optional[str]:
    candidates = _expand_path_candidates(raw_path, session_path)
    if not candidates:
        return None
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _build_part_plan(args: argparse.Namespace, session_path: str, mesh_info: dict) -> dict:
    max_parts = int(args.max_parts)
    if max_parts < 1:
        raise ValueError("--max-parts must be >= 1.")
    min_parts = int(args.min_parts)
    if min_parts < 1:
        raise ValueError("--min-parts must be >= 1.")
    if min_parts > max_parts:
        raise ValueError("--min-parts cannot exceed --max-parts.")

    labels = list(args.part_labels or [])
    if len(labels) > max_parts:
        raise ValueError("len(--part-labels) cannot exceed --max-parts.")
    part_mesh_paths = list(args.part_mesh_paths or [])
    if len(part_mesh_paths) > max_parts:
        raise ValueError("len(--part-mesh-paths) cannot exceed --max-parts.")

    part_count = max(min_parts, len(labels), len(part_mesh_paths), 1)
    if part_count == 1 and not labels and not part_mesh_paths:
        labels = ["whole_object"]
        mode = "default_single_part"
    elif part_mesh_paths:
        if not labels:
            labels = [
                os.path.splitext(os.path.basename(os.path.expanduser(str(path))))[0] or f"part_{idx:03d}"
                for idx, path in enumerate(part_mesh_paths)
            ]
        labels = labels + [f"part_{idx:03d}" for idx in range(len(labels), part_count)]
        if len(part_mesh_paths) == part_count:
            mode = "provided_part_meshes"
        else:
            mode = "provided_part_meshes_plus_segmentation"
    elif labels:
        labels = labels + [f"part_{idx:03d}" for idx in range(len(labels), part_count)]
        mode = "named_minimum"
    else:
        labels = [f"part_{idx:03d}" for idx in range(part_count)]
        mode = "minimum_count"

    if part_count > max_parts:
        raise ValueError(f"planned part count {part_count} exceeds --max-parts {max_parts}.")

    input_mesh_path = mesh_info.get("mesh_path")
    parts = []
    warnings = []
    if part_mesh_paths and len(part_mesh_paths) < part_count:
        warnings.append(
            f"{len(part_mesh_paths)} provided part meshes for {part_count} planned slots; "
            "remaining slots will require segmentation from the input/composite mesh."
        )
    for idx in range(part_count):
        part_id = f"part_{idx:03d}"
        part_mesh_path = None
        mesh_source = "pending_segmentation"

        if idx < len(part_mesh_paths):
            part_mesh_path = _resolve_optional_input_path(part_mesh_paths[idx], session_path)
            mesh_source = "provided_part_mesh"
            if not part_mesh_path or not os.path.exists(part_mesh_path):
                warnings.append(f"{part_id}: provided part mesh path does not exist: {part_mesh_path}")
        elif part_count == 1 and idx == 0:
            part_mesh_path = input_mesh_path
            mesh_source = "whole_input_mesh"

        status = "ready" if part_mesh_path and os.path.exists(part_mesh_path) else "pending_segmentation"
        parts.append(
            {
                "part_id": part_id,
                "label": labels[idx],
                "required": True,
                "status": status,
                "mesh_path": part_mesh_path,
                "mesh_source": mesh_source,
                "source_input_mesh_path": input_mesh_path,
                "canonical_T_part": _identity4(),
            }
        )

    return {
        "schema": "paradex.articulated_object.part_plan.v0",
        "mode": mode,
        "max_parts": max_parts,
        "min_parts": min_parts,
        "part_count": part_count,
        "provided_part_mesh_count": len(part_mesh_paths),
        "forced": mode != "default_single_part",
        "parts": parts,
        "warnings": warnings,
    }


def _write_part_manifests(output_dir: str, part_plan: dict) -> List[str]:
    part_dir = os.path.join(output_dir, "parts")
    os.makedirs(part_dir, exist_ok=True)
    paths = []
    for part in part_plan["parts"]:
        path = os.path.join(part_dir, f"{part['part_id']}.json")
        _write_json(path, part)
        paths.append(path)
    _write_json(os.path.join(part_dir, "part_plan.json"), part_plan)
    return paths


def _build_articulated_object_model(
    session: dict,
    session_path: str,
    output_dir: str,
    mesh_info: dict,
    state_reports: List[dict],
    part_plan: dict,
    segmentation_summary: Optional[dict] = None,
    completion_summary: Optional[dict] = None,
    motion_observations: Optional[dict] = None,
    kinematic_model: Optional[dict] = None,
) -> dict:
    object_name = session.get("object_name") or os.path.basename(os.path.dirname(session_path))
    mesh_path = mesh_info.get("mesh_path")
    usable_states = [
        report
        for report in state_reports
        if report["raw_cameras"] and not report["blocking_warnings"] and not report["missing_undistorted_cameras"]
    ]

    return {
        "schema": "paradex.articulated_object.offline_model.v0",
        "status": "preprocessed",
        "object_name": object_name,
        "session_path": session_path,
        "output_dir": output_dir,
        "inputs": {
            "mesh": {
                "path": mesh_path,
                "role": "input",
                "exists": mesh_info.get("exists", False),
                "stats": mesh_info.get("mesh_stats"),
            }
        },
        "part_plan": {
            "mode": part_plan["mode"],
            "max_parts": part_plan["max_parts"],
            "min_parts": part_plan["min_parts"],
            "part_count": part_plan["part_count"],
            "provided_part_mesh_count": part_plan.get("provided_part_mesh_count", 0),
            "forced": part_plan["forced"],
            "warnings": part_plan["warnings"],
        },
        "parts": part_plan["parts"],
        "segmentation": {
            "status": (segmentation_summary or {}).get("status"),
            "method": (segmentation_summary or {}).get("method"),
            "path": os.path.join(output_dir, "segmentation", "segmentation.json"),
            "parts_dir": (segmentation_summary or {}).get("parts_dir"),
            "part_count": len((segmentation_summary or {}).get("parts", [])),
        },
        "completion": {
            "status": (completion_summary or {}).get("status"),
            "method": (completion_summary or {}).get("method"),
            "path": os.path.join(output_dir, "completion", "completion.json"),
            "parts_dir": (completion_summary or {}).get("parts_dir"),
            "part_count": len((completion_summary or {}).get("parts", [])),
        },
        "kinematic_graph": {
            "nodes": [part["part_id"] for part in part_plan["parts"]],
            "edges": (kinematic_model or {}).get("joints", []),
            "status": (kinematic_model or {}).get(
                "status",
                "pending_joint_fitting" if len(part_plan["parts"]) > 1 else "single_part_placeholder",
            ),
            "path": os.path.join(output_dir, "kinematic_model", "kinematic_model.json"),
        },
        "motion_observations": {
            "status": (motion_observations or {}).get("status"),
            "path": os.path.join(output_dir, "motion_observations", "motion_observations.json"),
            "part_pose_tracks_path": (motion_observations or {}).get("part_pose_tracks_path"),
            "reference_state_id": (motion_observations or {}).get("reference_state_id"),
            "pair_count": len((motion_observations or {}).get("pairs", [])),
        },
        "state_observations": [
            {
                "state_id": report["state_id"],
                "state_label": report["state_label"],
                "state_kind": report["state_kind"],
                "images_dir": report["undistorted_images_dir"],
                "raw_images_dir": report["raw_images_dir"],
                "camera_count": len(report["undistorted_cameras"]),
                "multiview": report.get("multiview"),
                "registration": report.get("registration"),
                "part_registration": report.get("part_registration"),
                "warnings": report["warnings"],
            }
            for report in state_reports
        ],
        "usable_state_ids": [report["state_id"] for report in usable_states],
        "pending_stages": [
            {
                "stage": "P1_state_reconstruction",
                "reason": "Needs a selected reconstruction backend such as COLMAP/MVS, depth fusion, or an external tracker.",
            },
            {
                "stage": "P2_mesh_registration",
                "reason": "Current registration refines a coarse PCA initialization with point-cloud ICP only; robust silhouette/mask alignment or tracker validation is still needed.",
            },
            {
                "stage": "P3_motion_segmentation",
                "reason": "Needs masks, segmented pointclouds, or motion consistency across at least three sufficiently different states.",
            },
            {
                "stage": "P4_joint_parameter_fitting",
                "reason": "Current joint model is a candidate estimate; validate with more states or runtime tracking before use.",
            },
            {
                "stage": "P5_hidden_surface_completion",
                "reason": "Current completion is sparse residual/surfel geometry; dense MVS/TSDF or watertight reconstruction is still needed for final tracking meshes.",
            },
        ],
    }


def _html_relpath(path: Optional[str], root: str) -> Optional[str]:
    if not path:
        return None
    try:
        if not os.path.exists(path):
            return None
        return os.path.relpath(path, root).replace(os.sep, "/")
    except OSError:
        return None


def _list_existing_files(root: str, parts: List[str], extensions: Tuple[str, ...], limit: int = 120) -> List[str]:
    directory = os.path.join(root, *parts)
    if not os.path.isdir(directory):
        return []
    found = []
    for current_dir, _dirnames, filenames in os.walk(directory):
        for filename in sorted(filenames):
            if filename.lower().endswith(extensions):
                found.append(os.path.join(current_dir, filename))
                if len(found) >= limit:
                    return found
    return found


def _write_visual_debug_report(output_dir: str, manifest: dict) -> Optional[str]:
    import html

    report_path = os.path.join(output_dir, "visual_debug_report.html")

    def link(path: Optional[str], text: Optional[str] = None) -> str:
        rel = _html_relpath(path, output_dir)
        if rel is None:
            return ""
        return f'<a href="{html.escape(rel)}">{html.escape(text or os.path.basename(path))}</a>'

    def image(path: Optional[str], caption: str) -> str:
        rel = _html_relpath(path, output_dir)
        if rel is None:
            return ""
        return (
            '<figure>'
            f'<a href="{html.escape(rel)}"><img src="{html.escape(rel)}" alt="{html.escape(caption)}"></a>'
            f'<figcaption>{html.escape(caption)}</figcaption>'
            '</figure>'
        )

    mv_states = (manifest.get("multiview") or {}).get("state_summaries", [])
    seg = manifest.get("segmentation") or {}
    seg_visual = seg.get("visual_diagnostics") or (seg.get("diagnostics") or {}).get("visualization") or {}
    seg_paths = seg_visual.get("paths") or {}
    overview_paths = seg_paths.get("overview_pngs") or {}
    registration_states = (manifest.get("registration") or {}).get("state_summaries", [])
    silhouette = manifest.get("silhouette_evidence") or {}
    orientation = manifest.get("silhouette_orientation") or {}
    carve = manifest.get("moving_part_carve") or {}

    rows = []
    for state in mv_states:
        diag = state.get("pair_diagnostics") or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(state.get('state_id')))}</td>"
            f"<td>{html.escape(str(state.get('status')))}</td>"
            f"<td>{html.escape(str(state.get('camera_count')))}</td>"
            f"<td>{html.escape(str(state.get('point_count')))}</td>"
            f"<td>{html.escape(str(diag.get('triangulated_pair_count')))}</td>"
            f"<td>{html.escape(str(diag.get('triangulated_points_median_all_pairs')))}</td>"
            f"<td>{link(diag.get('pairs_ranked_csv'), 'pairs ranked')}</td>"
            "</tr>"
        )

    seg_rows = []
    for part in seg.get("parts", []):
        seg_rows.append(
            "<tr>"
            f"<td>{html.escape(str(part.get('part_id')))}</td>"
            f"<td>{html.escape(str(part.get('label')))}</td>"
            f"<td>{html.escape(str(part.get('status')))}</td>"
            f"<td>{html.escape(str(part.get('vertex_count')))}</td>"
            f"<td>{html.escape(str(part.get('vertex_fraction')))}</td>"
            f"<td>{html.escape(str(part.get('face_count')))}</td>"
            f"<td>{link(part.get('mesh_path'), 'mesh')}</td>"
            "</tr>"
        )

    heatmaps = []
    for state in mv_states[:24]:
        diag = state.get("pair_diagnostics") or {}
        heatmaps.append(image(diag.get("triangulation_heatmap_png"), f"{state.get('state_id')} triangulation"))
        heatmaps.append(image(diag.get("ransac_inlier_heatmap_png"), f"{state.get('state_id')} RANSAC"))

    silhouette_images = [
        image(path, os.path.relpath(path, output_dir).replace(os.sep, "/"))
        for path in _list_existing_files(output_dir, ["silhouette_evidence"], (".png",), limit=48)
    ]
    orientation_images = [
        image(path, os.path.relpath(path, output_dir).replace(os.sep, "/"))
        for path in _list_existing_files(output_dir, ["silhouette_orientation"], (".png",), limit=48)
    ]
    registration_images = [
        image(path, os.path.relpath(path, output_dir).replace(os.sep, "/"))
        for path in _list_existing_files(output_dir, ["registration"], (".png",), limit=72)
        if "image_refine" in path.replace(os.sep, "/")
    ]

    proposal_links = [
        f"<li>{link(path)}</li>"
        for path in _list_existing_files(output_dir, ["segmentation", "proposals"], (".obj",), limit=40)
    ]
    ply_links = []
    ply_preview_candidates = []
    for path in [
        seg_paths.get("segmented_mesh_colored_ply"),
        seg_paths.get("segmented_parts_side_by_side_ply"),
        carve.get("combined_path"),
    ]:
        rendered = link(path)
        if rendered:
            ply_links.append(f"<li>{rendered}</li>")
            ply_preview_candidates.append((path, os.path.basename(path)))
    for state in registration_states[:24]:
        rendered = link(state.get("sparse_vs_aligned_mesh_overlay_path"), f"{state.get('state_id')} registration overlay")
        if rendered:
            ply_links.append(f"<li>{rendered}</li>")
            if len(ply_preview_candidates) < 8:
                ply_preview_candidates.append(
                    (state.get("sparse_vs_aligned_mesh_overlay_path"), f"{state.get('state_id')} registration overlay")
                )

    ply_previews = []
    for path, title in ply_preview_candidates[:8]:
        try:
            preview = _read_ascii_ply_preview(path, output_dir, str(title))
        except Exception as exc:
            preview = None
            ply_links.append(f"<li>{html.escape(str(title))}: preview unavailable ({html.escape(str(exc))})</li>")
        if preview is not None:
            ply_previews.append(preview)
    previews_json = json.dumps(ply_previews, ensure_ascii=False).replace("</", "<\\/")
    ply_viewer_html = "".join(
        (
            f'<div class="ply-viewer" data-ply-index="{idx}">'
            f'<div class="ply-title">{html.escape(item["title"])}</div>'
            f'<canvas width="520" height="360"></canvas>'
            f'<div class="ply-meta">{html.escape(str(item["shownVertexCount"]))}/{html.escape(str(item["vertexCount"]))} vertices, '
            f'{html.escape(str(item["shownFaceCount"]))}/{html.escape(str(item["faceCount"]))} faces - '
            f'<a href="{html.escape(str(item.get("path", "")))}">open PLY</a></div>'
            f'<button type="button" class="ply-reset">Reset view</button>'
            f'</div>'
        )
        for idx, item in enumerate(ply_previews)
    )
    ply_viewer_script = """
<script>
const PLY_PREVIEWS = __PLY_PREVIEWS__;

function setupPlyViewer(root, data) {
  const canvas = root.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const resetButton = root.querySelector(".ply-reset");
  let yaw = -0.7;
  let pitch = 0.45;
  let zoom = 1.0;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  const vertices = data.vertices || [];
  const colors = data.colors || [];
  const faces = data.faces || [];
  let center = [0, 0, 0];
  let radius = 1;
  if (vertices.length) {
    const minv = vertices[0].slice();
    const maxv = vertices[0].slice();
    for (const v of vertices) {
      for (let k = 0; k < 3; k++) {
        minv[k] = Math.min(minv[k], v[k]);
        maxv[k] = Math.max(maxv[k], v[k]);
      }
    }
    center = [(minv[0] + maxv[0]) * 0.5, (minv[1] + maxv[1]) * 0.5, (minv[2] + maxv[2]) * 0.5];
    radius = Math.max(maxv[0] - minv[0], maxv[1] - minv[1], maxv[2] - minv[2], 1e-9);
  }

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function transformedVertices() {
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const scale = Math.min(w, h) * 0.42 * zoom / radius;
    return vertices.map((v) => {
      const x0 = v[0] - center[0];
      const y0 = v[1] - center[1];
      const z0 = v[2] - center[2];
      const x1 = cy * x0 + sy * z0;
      const z1 = -sy * x0 + cy * z0;
      const y1 = cp * y0 - sp * z1;
      const z2 = sp * y0 + cp * z1;
      return {x: w * 0.5 + x1 * scale, y: h * 0.5 - y1 * scale, z: z2};
    });
  }

  function rgb(indices) {
    const c = [0, 0, 0];
    let n = 0;
    for (const idx of indices) {
      const color = colors[idx] || [190, 190, 190];
      c[0] += color[0]; c[1] += color[1]; c[2] += color[2];
      n += 1;
    }
    return `rgb(${Math.round(c[0] / n)}, ${Math.round(c[1] / n)}, ${Math.round(c[2] / n)})`;
  }

  function draw() {
    resizeCanvas();
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#f7f7f7";
    ctx.fillRect(0, 0, w, h);
    if (!vertices.length) {
      ctx.fillStyle = "#666";
      ctx.fillText("No vertices", 16, 24);
      return;
    }
    const tv = transformedVertices();
    if (faces.length) {
      const tris = faces
        .filter((f) => f.length >= 3 && tv[f[0]] && tv[f[1]] && tv[f[2]])
        .map((f) => ({face: f, z: (tv[f[0]].z + tv[f[1]].z + tv[f[2]].z) / 3}))
        .sort((a, b) => a.z - b.z);
      ctx.lineWidth = 0.4;
      for (const tri of tris) {
        const f = tri.face;
        ctx.beginPath();
        ctx.moveTo(tv[f[0]].x, tv[f[0]].y);
        ctx.lineTo(tv[f[1]].x, tv[f[1]].y);
        ctx.lineTo(tv[f[2]].x, tv[f[2]].y);
        ctx.closePath();
        ctx.fillStyle = rgb(f);
        ctx.globalAlpha = 0.84;
        ctx.fill();
        ctx.globalAlpha = 0.22;
        ctx.strokeStyle = "#202124";
        ctx.stroke();
      }
      ctx.globalAlpha = 1.0;
    } else {
      const order = tv.map((p, i) => [p.z, i]).sort((a, b) => a[0] - b[0]);
      const r = Math.max(1.1, Math.min(3.2, 4500 / Math.max(vertices.length, 1)));
      for (const item of order) {
        const i = item[1];
        const p = tv[i];
        const color = colors[i] || [190, 190, 190];
        ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  canvas.addEventListener("pointerdown", (ev) => {
    dragging = true;
    lastX = ev.clientX;
    lastY = ev.clientY;
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    yaw += (ev.clientX - lastX) * 0.01;
    pitch = Math.max(-1.45, Math.min(1.45, pitch + (ev.clientY - lastY) * 0.01));
    lastX = ev.clientX;
    lastY = ev.clientY;
    draw();
  });
  canvas.addEventListener("pointerup", () => { dragging = false; });
  canvas.addEventListener("pointercancel", () => { dragging = false; });
  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    zoom = Math.max(0.25, Math.min(8.0, zoom * Math.exp(-ev.deltaY * 0.001)));
    draw();
  }, {passive: false});
  resetButton.addEventListener("click", () => {
    yaw = -0.7;
    pitch = 0.45;
    zoom = 1.0;
    draw();
  });
  window.addEventListener("resize", draw);
  draw();
}

document.querySelectorAll(".ply-viewer").forEach((root) => {
  const index = Number(root.getAttribute("data-ply-index"));
  if (PLY_PREVIEWS[index]) setupPlyViewer(root, PLY_PREVIEWS[index]);
});
</script>
""".replace("__PLY_PREVIEWS__", previews_json)

    overview_html = "".join(
        image(overview_paths.get(view), f"segmentation {view.upper()} overview")
        for view in ("xy", "xz", "yz")
    )
    heatmap_html = "".join(item for item in heatmaps if item)
    registration_html = "".join(item for item in registration_images if item)
    silhouette_html = "".join(item for item in silhouette_images if item)
    orientation_html = "".join(item for item in orientation_images if item)

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Articulated Object Visual Debug Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; background: #fafafa; }}
    h1, h2 {{ margin: 0.8em 0 0.4em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; padding: 10px; background: white; border: 1px solid #ddd; }}
    img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 8px; font-size: 13px; color: #555; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 8px 0 18px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; text-align: left; }}
    th {{ background: #f0f0f0; }}
    code {{ background: #eee; padding: 1px 4px; }}
    .links {{ background: white; border: 1px solid #ddd; padding: 10px 18px; }}
    .viewer-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }}
    .ply-viewer {{ background: white; border: 1px solid #ddd; padding: 10px; }}
    .ply-viewer canvas {{ width: 100%; height: 360px; display: block; background: #f7f7f7; border: 1px solid #e2e2e2; touch-action: none; }}
    .ply-title {{ font-weight: 600; margin-bottom: 6px; }}
    .ply-meta {{ color: #555; font-size: 12px; margin-top: 6px; }}
    .ply-reset {{ margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Articulated Object Visual Debug Report</h1>
  <p>Object: <code>{html.escape(str(manifest.get('object_name')))}</code>,
     session: <code>{html.escape(str(manifest.get('session_name')))}</code>,
     created: <code>{html.escape(str(manifest.get('created_at')))}</code></p>

  <h2>Segmentation Overview</h2>
  <p>Status: <code>{html.escape(str(seg.get('status')))}</code>,
     method: <code>{html.escape(str(seg.get('method')))}</code>.
     Open the PLY links below in MeshLab or CloudCompare when the PNG view is ambiguous.</p>
  <div class="grid">{overview_html}</div>
  <table>
    <tr><th>part_id</th><th>label</th><th>status</th><th>vertices</th><th>vertex_fraction</th><th>faces</th><th>mesh</th></tr>
    {''.join(seg_rows)}
  </table>

  <h2>Multiview Matching</h2>
  <table>
    <tr><th>state</th><th>status</th><th>cameras</th><th>points</th><th>triangulated pairs</th><th>median all pairs</th><th>ranked pairs</th></tr>
    {''.join(rows)}
  </table>
  <div class="grid">{heatmap_html}</div>

  <h2>Registration Image Alignment</h2>
  <p>Orange is the initial registered mesh silhouette, green is the image-refined candidate,
     and magenta is the projected sparse-point image ROI used for scoring when available.</p>
  <div class="grid">{registration_html}</div>

  <h2>Silhouette / Image Evidence</h2>
  <p>Silhouette enabled: <code>{html.escape(str(silhouette.get('enabled')))}</code>,
     orientation sweep enabled: <code>{html.escape(str(orientation.get('enabled')))}</code>,
     moving carve enabled: <code>{html.escape(str(carve.get('enabled')))}</code>.</p>
  <div class="grid">{silhouette_html}{orientation_html}</div>

  <h2>3D Debug Assets</h2>
  <div class="viewer-grid">{ply_viewer_html}</div>
  <ul class="links">{''.join(ply_links) or '<li>No PLY debug assets found.</li>'}</ul>

  <h2>Top Proposal Meshes</h2>
  <ul class="links">{''.join(proposal_links) or '<li>No proposal OBJ files found.</li>'}</ul>
  {ply_viewer_script}
</body>
</html>
"""
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight and preprocess captured articulated-object state snapshots."
    )
    parser.add_argument("--session-path", default=None, help="Absolute path or path relative to shared_data.")
    parser.add_argument("--object-name", default=None, help="Object name under --capture-root.")
    parser.add_argument(
        "--capture-root",
        default=os.path.join("capture", "articulated_object"),
        help="Relative root under shared_data.",
    )
    parser.add_argument("--session-name", default=None, help="Session folder name. Defaults to latest.")
    parser.add_argument("--check-only", action="store_true", help="Only inspect; do not create undistorted images.")
    parser.add_argument("--force", action="store_true", help="Regenerate undistorted images even if images/ exists.")
    parser.add_argument(
        "--mesh-path",
        default=None,
        help=(
            "Input OBJ/mesh path used with captured images. If omitted, "
            "shared_data/mesh_blender/<object-name>/<object-name>.obj is inferred."
        ),
    )
    parser.add_argument(
        "--mesh-root-dir",
        default=None,
        help="Mesh root for the default mesh path. Defaults to shared_data/mesh_blender.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <session-path>/processed.",
    )
    parser.add_argument(
        "--skip-multiview",
        action="store_true",
        help="Skip feature matching and sparse multiview triangulation.",
    )
    parser.add_argument(
        "--feature-type",
        default="auto",
        choices=["auto", "sift", "orb"],
        help="Feature detector used for multiview matching.",
    )
    parser.add_argument("--max-features", type=int, default=3000, help="Max local features per camera image.")
    parser.add_argument("--match-ratio", type=float, default=0.75, help="Lowe ratio threshold for descriptor matching.")
    parser.add_argument("--min-pair-matches", type=int, default=25, help="Minimum matches/inliers per camera pair.")
    parser.add_argument(
        "--ransac-reproj-thresh",
        type=float,
        default=2.0,
        help="Pixel threshold for pairwise fundamental-matrix RANSAC.",
    )
    parser.add_argument(
        "--triangulation-reproj-thresh",
        type=float,
        default=3.0,
        help="Pixel threshold for keeping triangulated sparse points.",
    )
    parser.add_argument(
        "--min-triangulation-angle-deg",
        type=float,
        default=1.0,
        help="Minimum camera-ray angle for keeping triangulated sparse points.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=5.0,
        help="Maximum positive camera depth for kept sparse points. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max-points-per-state",
        type=int,
        default=200000,
        help="Cap exported sparse points per state. Set <=0 to disable.",
    )
    parser.add_argument(
        "--no-colmap-text",
        action="store_true",
        help="Do not export a COLMAP text model with known calibrated camera poses.",
    )
    parser.add_argument("--skip-registration", action="store_true", help="Skip input-mesh registration.")
    parser.add_argument(
        "--registration-method",
        default="pca_centroid",
        choices=["pca_centroid"],
        help="Initial input-mesh to sparse-pointcloud registration method.",
    )
    parser.add_argument(
        "--registration-refine-method",
        default="trimmed_icp",
        choices=["none", "trimmed_icp"],
        help="Optional refinement after the initial registration.",
    )
    parser.add_argument(
        "--registration-scale-mode",
        default="none",
        choices=["none", "uniform"],
        help="Whether initial registration may change mesh scale.",
    )
    parser.add_argument(
        "--registration-sample-count",
        type=int,
        default=5000,
        help="Number of mesh surface samples used for registration.",
    )
    parser.add_argument(
        "--registration-max-points",
        type=int,
        default=50000,
        help="Maximum sparse points used for registration.",
    )
    parser.add_argument(
        "--registration-eval-points",
        type=int,
        default=3000,
        help="Maximum points used to report nearest-neighbor registration quality.",
    )
    parser.add_argument(
        "--registration-export-points",
        type=int,
        default=5000,
        help="Maximum transformed mesh sample points exported for registration debugging.",
    )
    parser.add_argument("--icp-iterations", type=int, default=30, help="Maximum trimmed ICP iterations.")
    parser.add_argument(
        "--icp-trim-fraction",
        type=float,
        default=0.5,
        help="Fraction of closest correspondences kept in each ICP iteration.",
    )
    parser.add_argument(
        "--icp-max-correspondence-distance",
        type=float,
        default=0.0,
        help="Optional max ICP correspondence distance in world units. Set <=0 to disable.",
    )
    parser.add_argument(
        "--icp-tolerance",
        type=float,
        default=1.0e-5,
        help="Stop ICP when mean correspondence distance changes by this amount or less.",
    )
    parser.add_argument(
        "--icp-max-source-points",
        type=int,
        default=5000,
        help="Maximum transformed mesh sample points used as ICP source points.",
    )
    parser.add_argument(
        "--icp-max-target-points",
        type=int,
        default=10000,
        help="Maximum sparse observation points used as ICP target points.",
    )
    parser.add_argument(
        "--icp-min-correspondences",
        type=int,
        default=20,
        help="Minimum valid correspondences required for an ICP update.",
    )
    parser.add_argument(
        "--icp-acceptance-ratio",
        type=float,
        default=1.05,
        help="Accept ICP if final median NN distance is within this ratio of the coarse result.",
    )
    parser.add_argument(
        "--registration-object-crop-distance",
        type=float,
        default=0.15,
        help=(
            "Object ROI crop radius as a fraction of the mesh bbox diagonal for robust registration "
            "against scene-contaminated sparse clouds. Set <=0 to use the legacy full-cloud coarse+ICP path."
        ),
    )
    parser.add_argument(
        "--registration-crop-iterations",
        type=int,
        default=3,
        help="Number of crop-and-refit iterations used by object-ROI registration.",
    )
    parser.add_argument(
        "--registration-object-tight-fraction",
        type=float,
        default=0.05,
        help=(
            "Distance (fraction of the mesh bbox diagonal) counted as a tight object inlier when "
            "scoring registration orientation candidates and reporting object registration quality."
        ),
    )
    parser.add_argument(
        "--registration-coarse-method",
        default="hull_pca",
        choices=["hull_pca", "mask_silhouette"],
        help=(
            "How the coarse whole-object pose is found. 'hull_pca' (default) aligns the mesh to the "
            "visual-hull/sparse point cloud by PCA + cropped ICP. 'mask_silhouette' fits the known "
            "mesh directly to the multiview object masks (translation from mask-centroid ray "
            "intersection, orientation from a coarse SO(3) grid scored by a trimmed-mean silhouette "
            "vs. mask cost, then local refine); it falls back to hull_pca when too few masks exist."
        ),
    )
    parser.add_argument(
        "--registration-silhouette-min-cameras",
        type=int,
        default=3,
        help="mask_silhouette: minimum masked views required, else fall back to hull_pca.",
    )
    parser.add_argument(
        "--registration-silhouette-normal-dirs",
        type=int,
        default=48,
        help="mask_silhouette: number of face-normal directions in the coarse orientation grid.",
    )
    parser.add_argument(
        "--registration-silhouette-inplane-steps",
        type=int,
        default=12,
        help="mask_silhouette: number of in-plane rotation steps per normal direction.",
    )
    parser.add_argument(
        "--registration-silhouette-coarse-cameras",
        type=int,
        default=8,
        help="mask_silhouette: masked views used during the coarse orientation search (0 = all).",
    )
    parser.add_argument(
        "--registration-silhouette-coarse-downscale",
        type=int,
        default=4,
        help="mask_silhouette: mask/render downscale factor during the coarse orientation search.",
    )
    parser.add_argument(
        "--registration-silhouette-refine-cameras",
        type=int,
        default=12,
        help="mask_silhouette: masked views used during the local refine (0 = all).",
    )
    parser.add_argument(
        "--registration-silhouette-refine-seeds",
        type=int,
        default=2,
        help="mask_silhouette: number of best coarse orientations locally refined.",
    )
    parser.add_argument(
        "--registration-silhouette-refine-rounds",
        type=int,
        default=12,
        help="mask_silhouette: maximum greedy coordinate-descent rounds during local refine.",
    )
    parser.add_argument(
        "--registration-silhouette-trim-fraction",
        type=float,
        default=0.3,
        help=(
            "mask_silhouette: fraction of the worst-scoring views dropped from the multiview cost "
            "so a minority of FP/FN masks and the moving articulated part do not dominate."
        ),
    )
    parser.add_argument(
        "--registration-silhouette-max-faces",
        type=int,
        default=4000,
        help="mask_silhouette: cap on mesh triangles rasterized per silhouette (0 = no cap).",
    )
    parser.add_argument(
        "--registration-silhouette-score-points",
        type=int,
        default=2000,
        help=(
            "mask_silhouette: mesh surface points projected per view for the chamfer/distance-transform "
            "silhouette cost. This cost replaces per-candidate triangle rasterization (the runtime "
            "bottleneck) with one point projection plus distance-field lookups; higher is finer but slower."
        ),
    )
    parser.add_argument(
        "--registration-group-share-pose",
        dest="registration_group_share_pose",
        action="store_true",
        default=True,
        help=(
            "mask_silhouette: after per-state fits, enforce one shared body orientation per placement "
            "group (body is fixed within a placement; only the articulated part moves). Scores each "
            "state's orientation and its +/-90/180-degree in-plane variants against the whole group's "
            "pooled masks and adopts the group-best, which breaks the flat-object 90-degree ambiguity "
            "that a single state cannot. No effect under --placement-group-mode single."
        ),
    )
    parser.add_argument(
        "--no-registration-group-share-pose",
        dest="registration_group_share_pose",
        action="store_false",
        help="Disable placement-group shared-orientation post-pass (register every state independently).",
    )
    parser.add_argument(
        "--registration-image-refine",
        action="store_true",
        help=(
            "Use the sparse/ROI registration as an initial guess, then select a pose candidate by "
            "single-frame rendered-silhouette boundary alignment to image edges. This does not use "
            "reference-current image difference, so it can support moved-object captures."
        ),
    )
    parser.add_argument(
        "--registration-image-max-cameras",
        type=int,
        default=8,
        help="Maximum cameras per state used to score image-space registration refinement (0 = all).",
    )
    parser.add_argument(
        "--registration-image-overlay-cameras",
        type=int,
        default=6,
        help="Maximum cameras per state exported as before/after image registration overlays.",
    )
    parser.add_argument(
        "--registration-image-normal-step-deg",
        type=int,
        default=30,
        help="Step in degrees for sweeping rotation about the mesh thin/normal axis during image refinement. Set <=0 to disable.",
    )
    parser.add_argument(
        "--registration-image-tilt-deg",
        type=float,
        default=8.0,
        help="Small +/- tilt candidates around the two broad mesh PCA axes during image refinement. Set <=0 to disable.",
    )
    parser.add_argument(
        "--registration-image-translation-fraction",
        type=float,
        default=0.0,
        help=(
            "Object-local translation sweep span as a fraction of the mesh bbox diagonal for image refinement. "
            "Set >0 when the silhouette is near the object but shifted in image space."
        ),
    )
    parser.add_argument(
        "--registration-image-translation-steps",
        type=int,
        default=1,
        help=(
            "Number of positive/negative translation samples per object axis. A value of 1 samples "
            "-span, 0, +span for each axis."
        ),
    )
    parser.add_argument(
        "--registration-image-scale-sweep",
        type=float,
        default=0.0,
        help=(
            "Uniform object-space scale sweep around 1.0 for image refinement. For example 0.10 tests "
            "0.90, 1.00, and 1.10."
        ),
    )
    parser.add_argument(
        "--registration-image-roi-mode",
        default="projected_sparse",
        choices=["projected_sparse", "off"],
        help=(
            "Image ROI used by registration refinement. 'projected_sparse' projects the current object/ROI "
            "sparse points into each camera and scores silhouette edges only inside that image ROI."
        ),
    )
    parser.add_argument(
        "--registration-image-roi-padding-fraction",
        type=float,
        default=0.35,
        help="Padding around the projected sparse-point image bbox as a fraction of its larger side.",
    )
    parser.add_argument(
        "--registration-image-roi-min-points",
        type=int,
        default=8,
        help="Minimum projected sparse points required before using an image ROI for a camera.",
    )
    parser.add_argument(
        "--registration-image-roi-point-dilate",
        type=int,
        default=11,
        help="Pixel radius used to dilate projected sparse points into the image ROI mask.",
    )
    parser.add_argument(
        "--registration-image-bbox-weight",
        type=float,
        default=12.0,
        help="Penalty weight for silhouette bbox center/size mismatch against the image object ROI.",
    )
    parser.add_argument(
        "--registration-image-min-bbox-area-ratio",
        type=float,
        default=0.25,
        help="Penalty starts when rendered silhouette bbox area is smaller than this fraction of the image ROI bbox.",
    )
    parser.add_argument(
        "--registration-image-max-bbox-area-ratio",
        type=float,
        default=3.0,
        help="Penalty starts when rendered silhouette bbox area is larger than this multiple of the image ROI bbox.",
    )
    parser.add_argument(
        "--registration-image-frontback-flips",
        dest="registration_image_frontback_flips",
        action="store_true",
        default=True,
        help="Test 180-degree flips around broad mesh PCA axes during image registration refinement.",
    )
    parser.add_argument(
        "--no-registration-image-frontback-flips",
        dest="registration_image_frontback_flips",
        action="store_false",
        help="Disable 180-degree front/back flip candidates during image registration refinement.",
    )
    parser.add_argument(
        "--registration-image-edge-trim-fraction",
        type=float,
        default=0.70,
        help="Fraction of rendered silhouette boundary pixels with the smallest edge distances used for robust image score.",
    )
    parser.add_argument(
        "--registration-image-acceptance-ratio",
        type=float,
        default=0.90,
        help="Accept image refinement only if best edge score is below initial_score * this ratio.",
    )
    parser.add_argument(
        "--registration-image-score-mode",
        default="auto",
        choices=["auto", "mask", "edge"],
        help=(
            "Score channel for image registration refinement. 'mask' compares the rendered silhouette "
            "against the background-plate object masks (precision/recall, needs 2+ placements); 'edge' "
            "is the Canny edge-distance score; 'auto' uses masks when they exist, else edges."
        ),
    )
    parser.add_argument(
        "--registration-image-mask-downscale",
        type=int,
        default=2,
        help="Downscale factor for mask-channel silhouette scoring (speed; 1 = full resolution).",
    )
    parser.add_argument(
        "--registration-mask-precision-weight",
        type=float,
        default=1.0,
        help="Weight of (1 - silhouette-inside-mask fraction) in the mask score.",
    )
    parser.add_argument(
        "--registration-mask-recall-weight",
        type=float,
        default=0.5,
        help="Weight of (1 - mask-covered-by-silhouette fraction) in the mask score.",
    )
    parser.add_argument(
        "--registration-image-canny-low",
        type=int,
        default=50,
        help="Low Canny threshold for image-space registration scoring.",
    )
    parser.add_argument(
        "--registration-image-canny-high",
        type=int,
        default=150,
        help="High Canny threshold for image-space registration scoring.",
    )
    parser.add_argument(
        "--roi-crop-mode",
        choices=["auto", "off"],
        default="auto",
        help=(
            "Automatic background removal for every per-state cloud. 'auto' crops each whole-scene "
            "triangulation to a world-volume sphere (center from camera-axis convergence, radius from the "
            "mesh size) and removes the dominant supporting plane (floor). 'off' uses the raw scene cloud."
        ),
    )
    parser.add_argument(
        "--roi-radius-factor",
        type=float,
        default=1.5,
        help="ROI sphere radius as a multiple of the mesh bbox diagonal (auto background removal).",
    )
    parser.add_argument(
        "--roi-ground-plane-fraction",
        type=float,
        default=0.02,
        help="Ground-plane RANSAC inlier tolerance as a fraction of the mesh bbox diagonal.",
    )
    parser.add_argument(
        "--roi-ground-min-inlier-fraction",
        type=float,
        default=0.15,
        help="Minimum plane inlier fraction (of ROI points) required to treat the dominant plane as the floor.",
    )
    parser.add_argument(
        "--roi-ground-side-fraction",
        type=float,
        default=0.6,
        help="Minimum fraction of off-plane points that must lie on the camera side to accept ground removal.",
    )
    parser.add_argument(
        "--roi-ground-ransac-iterations",
        type=int,
        default=300,
        help="RANSAC iterations for the ground-plane fit in the ROI front-end.",
    )
    parser.add_argument(
        "--roi-min-keep-fraction",
        type=float,
        default=0.05,
        help="Guard: if ROI+ground would keep fewer than this fraction of raw points, skip that step (safe no-op).",
    )
    parser.add_argument(
        "--placement-group-mode",
        default="auto",
        choices=["auto", "manual", "single"],
        help=(
            "How states are grouped by object placement for hybrid captures. 'auto' clusters per-state "
            "object-cloud centroids; 'manual' uses --placement-groups; 'single' keeps the legacy "
            "one-group (fixed-body) behavior."
        ),
    )
    parser.add_argument(
        "--placement-groups",
        default="",
        help=(
            "Manual placement groups as semicolon-separated comma lists of state ids, "
            "e.g. '000,001;002,003;004,005'. States not listed get their own group."
        ),
    )
    parser.add_argument(
        "--placement-group-distance-fraction",
        type=float,
        default=0.35,
        help=(
            "Auto placement grouping: two states share a placement when their object-cloud centroids are "
            "within this fraction of the mesh bbox diagonal."
        ),
    )
    parser.add_argument(
        "--background-plate-mode",
        default="auto",
        choices=["auto", "off"],
        help=(
            "Median background plates + per-state object masks. 'auto' runs when 2+ placement groups "
            "exist (each camera's per-pixel median over all states recovers the empty background)."
        ),
    )
    parser.add_argument(
        "--object-mask-source",
        default="auto",
        choices=["auto", "plate", "external"],
        help=(
            "Provider of the per-state object masks under <output>/object_masks/. 'external' consumes "
            "masks written by another tool (generate_masks_sam3.py); plate building is skipped and "
            "plate trust gating is disabled. 'plate' forces the legacy median-plate masks. 'auto' "
            "prefers external whenever object_masks/masks_manifest.json declares a non-plate provider."
        ),
    )
    parser.add_argument(
        "--object-mask-threshold",
        type=float,
        default=30.0,
        help="Threshold on max-channel |image - background plate| for the object mask.",
    )
    parser.add_argument(
        "--object-mask-blur",
        type=int,
        default=5,
        help="Gaussian blur kernel applied to the plate difference before thresholding.",
    )
    parser.add_argument(
        "--object-mask-morph-iterations",
        type=int,
        default=2,
        help="Morphological open+close iterations cleaning the object mask.",
    )
    parser.add_argument(
        "--object-mask-min-component-fraction",
        type=float,
        default=0.0005,
        help="Drop mask components smaller than this fraction of the image pixels.",
    )
    parser.add_argument(
        "--object-mask-suppress-shadows",
        default="on",
        choices=["on", "off"],
        help=(
            "Remove pixels darker than the plate but with the plate's chromaticity (shadows on the "
            "white cloth) from the object mask."
        ),
    )
    parser.add_argument(
        "--object-mask-shadow-brightness-low",
        type=float,
        default=0.35,
        help="Shadow suppression: minimum image/plate brightness ratio treated as shadow.",
    )
    parser.add_argument(
        "--object-mask-shadow-brightness-high",
        type=float,
        default=0.97,
        help="Shadow suppression: maximum image/plate brightness ratio treated as shadow.",
    )
    parser.add_argument(
        "--object-mask-shadow-chroma-threshold",
        type=float,
        default=0.10,
        help="Shadow suppression: maximum L1 chromaticity distance to the plate treated as shadow.",
    )
    parser.add_argument(
        "--object-mask-overlay-cameras",
        type=int,
        default=4,
        help="Number of cameras exported as [image | mask overlay] panels per state.",
    )
    parser.add_argument(
        "--object-mask-shadow-plate-min-brightness",
        type=float,
        default=110.0,
        help=(
            "Cloth-likeness test: plate pixel mean at least this (cloth is bright). Bounds both "
            "shadow suppression and the mask trust map."
        ),
    )
    parser.add_argument(
        "--object-mask-shadow-plate-max-chroma",
        type=float,
        default=45.0,
        help=(
            "Cloth-likeness test: plate max-min channel spread at most this (cloth is unsaturated). "
            "Bounds both shadow suppression and the mask trust map."
        ),
    )
    parser.add_argument(
        "--mask-trust-mode",
        default="auto",
        choices=["auto", "off"],
        help=(
            "Trust gating for plate-based mask evidence. 'auto' builds a per-camera trust map "
            "(cloth-like plate pixels) and counts hull votes and registration mask scores only on "
            "trusted pixels; untrusted pixels are 'unknown', never negative evidence, so off-cloth "
            "mask holes cannot carve the hull or mis-rank poses. Within-placement diff channels "
            "(image_diff / silhouette evidence / carve) are never gated: side cameras keep their "
            "full articulation evidence. 'off' restores ungated behavior."
        ),
    )
    parser.add_argument(
        "--mask-trust-erode",
        type=int,
        default=1,
        help="Erode the trust map by this many pixels (keeps the soft cloth boundary out of the trusted region).",
    )
    parser.add_argument(
        "--registration-mask-min-trusted-px",
        type=int,
        default=150,
        help=(
            "Drop a camera from mask scoring when its trust-gated object mask has fewer pixels than "
            "this (at the scoring downscale); a fragmentary mask would mis-rank poses."
        ),
    )
    parser.add_argument(
        "--visual-hull-mode",
        default="auto",
        choices=["auto", "off"],
        help="Per-state visual hull from object masks. 'auto' runs whenever object masks exist.",
    )
    parser.add_argument(
        "--hull-grid-resolution",
        type=int,
        default=96,
        help="Voxels per axis for the visual hull volume.",
    )
    parser.add_argument(
        "--hull-min-view-fraction",
        type=float,
        default=0.7,
        help="Keep a voxel when it projects inside the object mask in at least this fraction of viewing cameras.",
    )
    parser.add_argument(
        "--hull-min-opportunities",
        type=int,
        default=3,
        help="Minimum cameras that must see a voxel before it can be kept.",
    )
    parser.add_argument(
        "--hull-ground-margin-fraction",
        type=float,
        default=0.01,
        help="Above-ground margin for hull voxels as a fraction of the ROI radius.",
    )
    parser.add_argument(
        "--registration-target",
        default="auto",
        choices=["auto", "hull", "sparse"],
        help=(
            "Whole-object registration target cloud. 'hull' registers the mesh against the per-state "
            "visual hull surface (volumetric, texture-free; recommended for weakly textured objects); "
            "'sparse' keeps the triangulated sparse cloud; 'auto' prefers the hull when it exists."
        ),
    )
    parser.add_argument(
        "--registration-hull-min-points",
        type=int,
        default=100,
        help="Minimum hull surface points required before the hull replaces the sparse registration target.",
    )
    parser.add_argument(
        "--hull-mask-erode",
        type=int,
        default=0,
        help=(
            "Erode each object mask by this many pixels before hull voting. Compensates the "
            "systematic silhouette bloat from mask blur/closing; holes are absorbed by the vote."
        ),
    )
    parser.add_argument(
        "--registration-image-refine-rounds",
        type=int,
        default=1,
        help=(
            "Greedy rounds of the image refinement candidate search. Each round applies the best "
            "improving delta and searches again, so rotations about different axes compose and an "
            "arbitrary-axis initial error can be corrected."
        ),
    )
    parser.add_argument(
        "--registration-image-round-acceptance-ratio",
        type=float,
        default=0.999,
        help="A refinement round is applied only if it improves the current score by this ratio.",
    )
    parser.add_argument(
        "--registration-image-tilt-multiples",
        type=int,
        default=1,
        help=(
            "Tilt candidates at +/- tilt-deg times 1..N about the wide/mid/normal mesh axes. "
            "Set 2-3 so tilts up to ~24 degrees are reachable within one round."
        ),
    )
    parser.add_argument(
        "--enable-silhouette-evidence",
        action="store_true",
        help=(
            "Road S stage 1: render the registered mesh body silhouette per view and export "
            "beyond-body change overlays (moving/separated part isolated in image space) for validation."
        ),
    )
    parser.add_argument(
        "--silhouette-body-dilate-iters",
        type=int,
        default=3,
        help="Dilation iterations applied to the rendered body silhouette before subtracting it from change.",
    )
    parser.add_argument(
        "--silhouette-max-cameras",
        type=int,
        default=8,
        help="Maximum cameras per state to export silhouette evidence overlays for (0 = all).",
    )
    parser.add_argument(
        "--silhouette-orientation-sweep",
        action="store_true",
        help=(
            "Diagnose registration orientation: rotate the mesh about its slab normal through a full "
            "turn, score silhouette-outline vs image-edge agreement, and export best-angle overlays."
        ),
    )
    parser.add_argument(
        "--silhouette-sweep-step-deg",
        type=int,
        default=15,
        help="Angular step (degrees) for the silhouette orientation sweep diagnostic.",
    )
    parser.add_argument(
        "--enable-moving-part-carve",
        action="store_true",
        help=(
            "Pose-free multiview carve of the moving part from inter-state image change (fixed-body "
            "captures). Needs no mesh registration or object masks; writes moving_part_carve/*.ply."
        ),
    )
    parser.add_argument(
        "--carve-grid-resolution",
        type=int,
        default=96,
        help="Voxels per axis for the moving-part carve volume (higher = finer, slower).",
    )
    parser.add_argument(
        "--carve-min-view-fraction",
        type=float,
        default=0.5,
        help="A voxel is carved when it lands in the image-change mask in at least this fraction of viewing cameras.",
    )
    parser.add_argument(
        "--carve-min-opportunities",
        type=int,
        default=3,
        help="Minimum cameras that must see a voxel (in front, in bounds) before it can be carved.",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=8,
        help="Upper bound for later automatic part hypotheses.",
    )
    parser.add_argument(
        "--min-parts",
        type=int,
        default=1,
        help="Minimum part slots to keep in the output model contract.",
    )
    parser.add_argument(
        "--part-labels",
        nargs="*",
        default=None,
        help="Force exact named part slots, e.g. --part-labels base lid handle.",
    )
    parser.add_argument(
        "--part-mesh-paths",
        nargs="*",
        default=None,
        help=(
            "Optional input mesh paths for known part slots. If fewer than planned parts are provided, "
            "remaining slots are filled by segmentation."
        ),
    )
    parser.add_argument(
        "--skip-part-registration",
        action="store_true",
        help="Skip per-part registration even when ready part meshes are provided.",
    )
    parser.add_argument(
        "--part-registration-refine-method",
        default="trimmed_icp",
        choices=["none", "trimmed_icp"],
        help="Optional per-part pose refinement after whole-object pose initialization.",
    )
    parser.add_argument(
        "--part-registration-sample-count",
        type=int,
        default=3000,
        help="Number of per-part mesh surface samples used for part registration.",
    )
    parser.add_argument(
        "--part-registration-max-points",
        type=int,
        default=50000,
        help="Maximum sparse observation points used for per-part registration.",
    )
    parser.add_argument(
        "--skip-segmentation-targets",
        action="store_true",
        help="Do not assign sparse points to generated/provided part meshes for per-part registration targets.",
    )
    parser.add_argument(
        "--part-target-sample-count",
        type=int,
        default=2000,
        help="Part mesh surface samples used when assigning sparse points to part targets.",
    )
    parser.add_argument(
        "--part-target-min-points",
        type=int,
        default=20,
        help="Minimum segmented sparse points required before using a part target.",
    )
    parser.add_argument(
        "--part-target-max-distance",
        type=float,
        default=0.0,
        help="Optional max canonical distance when assigning sparse points to part meshes. Set <=0 to disable.",
    )
    parser.add_argument(
        "--part-mask-root",
        default=None,
        help=(
            "Optional mask root. Searches <root>/<state_id>/<part_id>/<serial>.png, "
            "<root>/<state_id>/<label>/<serial>.png, <root>/<part_id>/<serial>.png, "
            "and <root>/<label>/<serial>.png. Defaults to <state>/masks."
        ),
    )
    parser.add_argument(
        "--part-mask-threshold",
        type=float,
        default=127.0,
        help="Mask pixel threshold for selecting projected sparse points.",
    )
    parser.add_argument(
        "--part-mask-min-views",
        type=int,
        default=1,
        help="Minimum number of masks that must contain a sparse point.",
    )
    parser.add_argument(
        "--part-mask-min-points",
        type=int,
        default=20,
        help="Minimum masked sparse points required before using a part-mask target.",
    )
    parser.add_argument(
        "--require-part-masks",
        action="store_true",
        help="Fail per-part registration instead of falling back to whole-state sparse points when masks are missing.",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip whole input mesh segmentation into planned part meshes.",
    )
    parser.add_argument(
        "--force-segmentation-with-provided-parts",
        action="store_true",
        help="Run mesh segmentation even when every planned part slot already has a provided mesh.",
    )
    parser.add_argument(
        "--segmentation-method",
        default="auto",
        choices=["auto", "mask_points", "moving_evidence", "geometry_proposals", "motion_residual"],
        help="How to split the input mesh into planned part meshes.",
    )
    parser.add_argument(
        "--segmentation-reference-state",
        default=None,
        help="State id used for mask-based mesh segmentation. Defaults to first usable registered state.",
    )
    parser.add_argument(
        "--segmentation-max-points-per-state",
        type=int,
        default=50000,
        help="Maximum sparse points per state used for mesh segmentation.",
    )
    parser.add_argument(
        "--segmentation-object-crop-distance",
        type=float,
        default=0.06,
        help=(
            "Keep only sparse points within this fraction of the mesh bbox diagonal from the "
            "registered mesh surface (canonical frame) before computing motion residual and "
            "moving evidence. Removes scene background/outliers and sharpens the moving-part "
            "signal. Set 0 to disable object cropping (use the full per-state sparse cloud)."
        ),
    )
    parser.add_argument(
        "--segmentation-min-states",
        type=int,
        default=2,
        help="Minimum usable states required for motion-residual mesh segmentation.",
    )
    parser.add_argument(
        "--segmentation-kmeans-iterations",
        type=int,
        default=50,
        help="Maximum k-means iterations for motion-residual mesh segmentation.",
    )
    parser.add_argument(
        "--segmentation-residual-clip",
        type=float,
        default=0.25,
        help="Clip normalized residual features before motion-residual clustering.",
    )
    parser.add_argument(
        "--segmentation-min-vertex-fraction",
        type=float,
        default=0.01,
        help="Warn when an exported part receives less than this fraction of mesh vertices.",
    )
    parser.add_argument(
        "--segmentation-adjacency-weld-tolerance",
        type=float,
        default=1.0e-6,
        help=(
            "Mesh-diagonal-relative tolerance for virtual vertex welding during segmentation graph construction. "
            "The original mesh is not modified. Use 0 for exact duplicate coordinates only, or <0 to disable."
        ),
    )
    parser.add_argument(
        "--skip-segmentation-cleanup",
        action="store_true",
        help="Disable small disconnected label-island cleanup after mesh segmentation.",
    )
    parser.add_argument(
        "--segmentation-cleanup-min-component-fraction",
        type=float,
        default=0.002,
        help="Reassign non-largest label components smaller than this mesh vertex fraction to adjacent labels.",
    )
    parser.add_argument(
        "--segmentation-cleanup-min-component-vertices",
        type=int,
        default=6,
        help="Minimum non-largest label component size kept during segmentation cleanup.",
    )
    parser.add_argument(
        "--segmentation-cleanup-report-limit",
        type=int,
        default=50,
        help="Maximum cleanup reassignment records kept in segmentation diagnostics.",
    )
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="Stop after mesh segmentation; skip part registration, completion, motion observations, and joint fitting.",
    )
    parser.add_argument(
        "--segmentation-diagnostic-top-proposals",
        type=int,
        default=20,
        help="Export this many top scored segmentation proposal OBJ files under processed/segmentation/proposals.",
    )
    parser.add_argument(
        "--segmentation-diagnostic-mesh-near-distance",
        type=float,
        default=0.05,
        help="Mesh-diagonal-normalized threshold for splitting moving evidence points into mesh-near and mesh-far PLY files.",
    )
    parser.add_argument(
        "--segmentation-relation-prior",
        default="articulated",
        choices=["articulated", "separable", "none"],
        help=(
            "Relation prior used by moving-evidence scoring. 'articulated' penalizes disconnected candidates "
            "unless they have direct moving seed support; 'separable' allows detached rigid candidates."
        ),
    )
    parser.add_argument(
        "--segmentation-moving-evidence-source",
        default="sparse",
        choices=["sparse", "image_diff", "combined"],
        help=(
            "Evidence source for moving_evidence segmentation. 'sparse' uses state-to-state sparse 3D residuals, "
            "'image_diff' votes mesh vertices from per-camera image differences, and 'combined' uses both."
        ),
    )
    parser.add_argument(
        "--segmentation-image-motion-threshold",
        type=float,
        default=30.0,
        help="Grayscale pixel-difference threshold used by --segmentation-moving-evidence-source image_diff/combined.",
    )
    parser.add_argument(
        "--segmentation-image-motion-blur",
        type=int,
        default=5,
        help="Odd Gaussian blur kernel used before image differencing. Values <=1 disable blur.",
    )
    parser.add_argument(
        "--segmentation-image-motion-dilate-iterations",
        type=int,
        default=2,
        help="3x3 dilation iterations applied to image-difference masks before projecting mesh vertices.",
    )
    parser.add_argument(
        "--segmentation-image-motion-min-score",
        type=float,
        default=0.05,
        help="Minimum per-vertex image-motion vote fraction used as the moving score threshold in image_diff mode.",
    )
    parser.add_argument(
        "--segmentation-image-motion-score-weight",
        type=float,
        default=1.0,
        help="Weight applied to image-motion scores when --segmentation-moving-evidence-source combined is used.",
    )
    parser.add_argument(
        "--segmentation-image-motion-min-changed-fraction",
        type=float,
        default=0.0005,
        help="Skip camera comparisons whose image-difference mask covers less than this image fraction.",
    )
    parser.add_argument(
        "--segmentation-image-motion-max-changed-fraction",
        type=float,
        default=0.50,
        help="Skip camera comparisons whose image-difference mask covers more than this image fraction. Set <=0 to disable.",
    )
    parser.add_argument(
        "--segmentation-image-motion-projection-mode",
        default="both",
        choices=["reference", "current", "both"],
        help="Vote vertices using the reference pose projection, current pose projection, or both.",
    )
    parser.add_argument(
        "--segmentation-image-motion-max-cameras-per-state",
        type=int,
        default=0,
        help="Maximum common cameras compared per non-reference state. Set 0 to use all common cameras.",
    )
    parser.add_argument(
        "--segmentation-moving-min-score",
        type=float,
        default=0.02,
        help="Minimum normalized per-vertex state residual range before a vertex can support a moving-part candidate.",
    )
    parser.add_argument(
        "--segmentation-moving-min-evidence-fraction",
        type=float,
        default=0.25,
        help="Minimum fraction of a candidate's vertices that must pass --segmentation-moving-min-score.",
    )
    parser.add_argument(
        "--segmentation-moving-min-observed-fraction",
        type=float,
        default=0.01,
        help="Minimum fraction of a moving candidate observed near sparse points in at least one state.",
    )
    parser.add_argument(
        "--segmentation-moving-observed-distance",
        type=float,
        default=0.03,
        help="Normalized mesh-diagonal distance for deciding whether a moving candidate vertex is observed.",
    )
    parser.add_argument(
        "--segmentation-moving-point-distance",
        type=float,
        default=0.03,
        help="Normalized distance used to extract state-to-state sparse moving evidence points.",
    )
    parser.add_argument(
        "--segmentation-moving-vertex-distance",
        type=float,
        default=0.03,
        help="Normalized mesh-diagonal distance for associating mesh vertices with moving evidence points.",
    )
    parser.add_argument(
        "--segmentation-moving-geometry-overlap-fraction",
        type=float,
        default=0.25,
        help="Minimum motion-evidence overlap before a geometry proposal can be used as an auxiliary moving candidate.",
    )
    parser.add_argument(
        "--segmentation-moving-geometry-auxiliary",
        dest="segmentation_moving_geometry_auxiliary",
        action="store_true",
        default=True,
        help="Allow geometry proposals that overlap moving evidence to become moving-part candidates.",
    )
    parser.add_argument(
        "--no-segmentation-moving-geometry-auxiliary",
        dest="segmentation_moving_geometry_auxiliary",
        action="store_false",
        help="Disable geometry proposals as auxiliary moving-part candidates.",
    )
    parser.add_argument(
        "--segmentation-moving-geometry-support-source",
        default="primary",
        choices=["primary", "primary_or_sparse"],
        help=(
            "Evidence used to admit auxiliary geometry candidates. In image_diff mode, 'primary' ignores sparse "
            "moving-point proximity and requires image-motion/seed support; 'primary_or_sparse' also allows sparse support."
        ),
    )
    parser.add_argument(
        "--segmentation-image-motion-min-geometry-fraction",
        type=float,
        default=0.0,
        help="In image_diff mode, reject auxiliary geometry candidates below this image-motion evidence fraction.",
    )
    parser.add_argument(
        "--segmentation-image-motion-geometry-penalty",
        type=float,
        default=0.0,
        help="Extra score penalty for auxiliary geometry candidates in image_diff mode.",
    )
    parser.add_argument(
        "--segmentation-moving-min-seed-vertices",
        type=int,
        default=3,
        help="Minimum high-confidence seed vertices that allow a grown moving candidate to survive mean-evidence filtering.",
    )
    parser.add_argument(
        "--segmentation-moving-raw-seed-penalty",
        type=float,
        default=0.80,
        help="Penalty for accepting the raw high-motion seed itself instead of a grown part candidate. Set 0 for tiny parts.",
    )
    parser.add_argument(
        "--segmentation-moving-seed-grow-rings",
        type=int,
        default=4,
        help="Number of mesh-adjacency rings grown from each moving seed to form larger part candidates.",
    )
    parser.add_argument(
        "--segmentation-moving-seed-relax-steps",
        type=int,
        default=4,
        help="Number of relaxed motion-threshold connected components grown from each moving seed.",
    )
    parser.add_argument(
        "--segmentation-moving-seed-relaxed-score-fraction",
        type=float,
        default=0.25,
        help="Lowest fraction of --segmentation-moving-min-score used when growing relaxed seed components.",
    )
    parser.add_argument(
        "--segmentation-moving-two-body-scoring",
        dest="segmentation_moving_two_body_scoring",
        action="store_true",
        default=True,
        help="Rescore moving candidates by how much a separate rigid part transform improves per-state residuals.",
    )
    parser.add_argument(
        "--no-segmentation-moving-two-body-scoring",
        dest="segmentation_moving_two_body_scoring",
        action="store_false",
        help="Disable separate rigid part residual-improvement scoring for moving candidates.",
    )
    parser.add_argument(
        "--segmentation-moving-two-body-weight",
        type=float,
        default=1.5,
        help="Score weight applied to the two-body rigid-fit residual improvement.",
    )
    parser.add_argument(
        "--segmentation-moving-two-body-min-improvement",
        type=float,
        default=0.0,
        help="Require this normalized residual improvement from two-body scoring. Keep 0 to use it as a soft score only.",
    )
    parser.add_argument(
        "--segmentation-moving-two-body-max-candidates",
        type=int,
        default=30,
        help="Maximum top moving candidates evaluated with two-body rigid-fit scoring.",
    )
    parser.add_argument(
        "--segmentation-moving-two-body-max-source-points",
        type=int,
        default=1200,
        help="Maximum candidate mesh vertices used as source points for two-body scoring.",
    )
    parser.add_argument(
        "--segmentation-moving-two-body-max-target-points",
        type=int,
        default=8000,
        help="Maximum sparse target points per state used for two-body scoring.",
    )
    parser.add_argument(
        "--segmentation-moving-two-body-min-correspondences",
        type=int,
        default=20,
        help="Minimum nearest-neighbor correspondences required for a candidate two-body fit.",
    )
    parser.add_argument(
        "--segmentation-moving-two-body-trim-fraction",
        type=float,
        default=0.35,
        help="Closest source-to-target correspondence fraction used for each candidate two-body rigid fit.",
    )
    parser.add_argument(
        "--segmentation-moving-two-body-report-states",
        type=int,
        default=12,
        help="Maximum per-state two-body scoring records stored in segmentation diagnostics.",
    )
    parser.add_argument(
        "--segmentation-moving-tail-min-fraction",
        type=float,
        default=0.03,
        help="Smallest high-motion vertex tail fraction tested for moving-evidence candidates.",
    )
    parser.add_argument(
        "--segmentation-moving-tail-max-fraction",
        type=float,
        default=0.30,
        help="Largest high-motion vertex tail fraction tested for moving-evidence candidates.",
    )
    parser.add_argument(
        "--segmentation-moving-tail-steps",
        type=int,
        default=6,
        help="Number of high-motion tail thresholds tested for moving-evidence candidates.",
    )
    parser.add_argument(
        "--segmentation-moving-max-evidence-points",
        type=int,
        default=20000,
        help="Maximum extracted sparse moving evidence points written for debugging.",
    )
    parser.add_argument(
        "--segmentation-proposal-min-vertex-fraction",
        type=float,
        default=0.002,
        help="Minimum mesh vertex fraction for geometry-proposal part candidates.",
    )
    parser.add_argument(
        "--segmentation-proposal-max-vertex-fraction",
        type=float,
        default=0.45,
        help="Maximum mesh vertex fraction for geometry-proposal part candidates.",
    )
    parser.add_argument(
        "--segmentation-proposal-max-boundary-fraction",
        type=float,
        default=0.50,
        help="Reject connected geometry proposals whose cut boundary is too broad relative to the candidate.",
    )
    parser.add_argument(
        "--segmentation-proposal-min-observed-fraction",
        type=float,
        default=0.01,
        help="Minimum candidate vertex fraction observed near sparse points in at least one state.",
    )
    parser.add_argument(
        "--segmentation-proposal-observed-distance",
        type=float,
        default=0.03,
        help="Normalized mesh-diagonal distance used to count observed vertices for geometry proposals.",
    )
    parser.add_argument(
        "--segmentation-proposal-min-score",
        type=float,
        default=0.02,
        help="Minimum score for accepting a geometry-proposal part candidate.",
    )
    parser.add_argument(
        "--segmentation-proposal-overlap-threshold",
        type=float,
        default=0.25,
        help="Maximum overlap allowed between accepted geometry-proposal candidates.",
    )
    parser.add_argument(
        "--segmentation-proposal-tail-min-fraction",
        type=float,
        default=0.03,
        help="Smallest tail fraction used when searching PCA/residual extrema for part proposals.",
    )
    parser.add_argument(
        "--segmentation-proposal-tail-max-fraction",
        type=float,
        default=0.30,
        help="Largest tail fraction used when searching PCA/residual extrema for part proposals.",
    )
    parser.add_argument(
        "--segmentation-proposal-tail-steps",
        type=int,
        default=6,
        help="Number of PCA/residual tail thresholds tested for geometry proposals.",
    )
    parser.add_argument(
        "--part-icp-acceptance-ratio",
        type=float,
        default=1.05,
        help="Accept part ICP if final median NN distance is within this ratio of the initial result.",
    )
    parser.add_argument(
        "--skip-completion",
        action="store_true",
        help="Skip sparse hidden-surface completion from per-part residual observations.",
    )
    parser.add_argument(
        "--completion-min-surface-distance",
        type=float,
        default=0.005,
        help="Minimum canonical distance from the current part mesh before a point is treated as missing-surface evidence.",
    )
    parser.add_argument(
        "--completion-max-surface-distance",
        type=float,
        default=0.0,
        help="Optional maximum canonical distance for completion evidence. Set <=0 to disable.",
    )
    parser.add_argument(
        "--completion-voxel-size",
        type=float,
        default=0.003,
        help="Voxel size for merging completion points in canonical part coordinates. Set <=0 to disable.",
    )
    parser.add_argument(
        "--completion-min-points",
        type=int,
        default=20,
        help="Minimum filtered completion points required to export a completed part candidate.",
    )
    parser.add_argument(
        "--completion-max-points-per-state",
        type=int,
        default=20000,
        help="Maximum per-state part target points used for completion.",
    )
    parser.add_argument(
        "--completion-surface-sample-count",
        type=int,
        default=5000,
        help="Surface samples from each existing part mesh used to reject already-explained points.",
    )
    parser.add_argument(
        "--completion-export-points",
        type=int,
        default=20000,
        help="Maximum completion points exported per completed part candidate.",
    )
    parser.add_argument(
        "--completion-surfel-radius",
        type=float,
        default=0.003,
        help="Radius of triangular surfels appended to the completed OBJ candidate. Set <=0 for vertex-only points.",
    )
    parser.add_argument(
        "--completion-allow-whole-targets",
        action="store_true",
        help="Allow whole-state sparse fallback targets for completion. Off by default to avoid cross-part contamination.",
    )
    parser.add_argument(
        "--skip-motion-observations",
        action="store_true",
        help="Skip part pose tracks and relative part-pair motion observation export.",
    )
    parser.add_argument(
        "--motion-reference-state",
        default=None,
        help="State id used as the reference for relative motion deltas. Defaults to first state with part poses.",
    )
    parser.add_argument(
        "--motion-min-states",
        type=int,
        default=2,
        help="Minimum states with both part poses required before a part pair is marked usable.",
    )
    parser.add_argument("--skip-joint-fitting", action="store_true", help="Skip candidate joint model fitting.")
    parser.add_argument(
        "--joint-min-states",
        type=int,
        default=2,
        help="Minimum relative-pose observations required for a joint candidate.",
    )
    parser.add_argument(
        "--joint-revolute-min-rotation-deg",
        type=float,
        default=5.0,
        help="Minimum observed relative rotation for a revolute candidate.",
    )
    parser.add_argument(
        "--joint-prismatic-min-translation",
        type=float,
        default=0.02,
        help="Minimum observed relative translation for a prismatic candidate.",
    )
    parser.add_argument(
        "--joint-separable-rotation-deg",
        type=float,
        default=25.0,
        help="Rotation threshold for marking a separable/free candidate.",
    )
    parser.add_argument(
        "--joint-separable-translation",
        type=float,
        default=0.10,
        help="Translation threshold for marking a separable/free candidate.",
    )
    parser.add_argument("--min-states", type=int, default=3, help="Recommended minimum state count for joint fitting.")
    parser.add_argument("--strict", action="store_true", help="Raise if required preprocessing warnings remain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    session_path = _resolve_session_path(args)
    if not os.path.isdir(session_path):
        raise FileNotFoundError(f"Session path not found: {session_path}")

    session, states = _load_session(session_path)
    expected_cameras = sorted(set(session.get("expected_cameras", [])))
    mesh_path, mesh_candidates = _resolve_mesh_path(args, session, session_path)
    mesh_info = _inspect_mesh(mesh_path, mesh_candidates)
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or os.path.join(session_path, "processed")))
    os.makedirs(output_dir, exist_ok=True)
    mesh_info = _prepare_mesh_info_from_provided_parts(args, session_path, output_dir, mesh_info)
    part_plan = _build_part_plan(args, session_path, mesh_info)

    state_reports = []
    for state in states:
        report = _process_state(
            state,
            expected_cameras,
            undistort=not args.check_only,
            force=args.force,
        )
        state_reports.append(report)
        if report["warnings"]:
            print(f"[WARN] {report['state_id']} {report['state_label']}: {report['warnings']}")
        else:
            print(
                f"[OK] {report['state_id']} {report['state_label']}: "
                f"raw={len(report['raw_cameras'])}, undistorted={len(report['undistorted_cameras'])}"
            )

    multiview_summaries = _process_multiview_states(state_reports, output_dir, args)
    object_roi_summary = _build_object_roi_clouds(state_reports, output_dir, mesh_info, args)
    placement_group_summary = _detect_placement_groups(state_reports, output_dir, mesh_info, args)
    background_plate_summary = _build_background_plates_and_masks(state_reports, output_dir, args)
    visual_hull_summary = _build_visual_hull_clouds(state_reports, output_dir, object_roi_summary, args)
    registration_summaries = _register_states(state_reports, output_dir, mesh_info, args)
    silhouette_evidence_summary = _render_silhouette_evidence(state_reports, output_dir, mesh_info, args)
    silhouette_orientation_summary = _silhouette_orientation_sweep(state_reports, output_dir, mesh_info, args)
    moving_part_carve_summary = _carve_moving_part_image_change(state_reports, output_dir, object_roi_summary, args)
    segmentation_summary = _segment_mesh_parts(state_reports, output_dir, mesh_info, part_plan, args, session_path)
    if args.segmentation_only:
        args.skip_part_registration = True
        args.skip_completion = True
        args.skip_motion_observations = True
        args.skip_joint_fitting = True
    part_registration_summaries = _register_parts_for_states(state_reports, output_dir, part_plan, args, session_path)
    completion_summary = _complete_hidden_surfaces(state_reports, output_dir, part_plan, args)
    motion_observations = _build_motion_observations(state_reports, output_dir, part_plan, args)
    kinematic_model = _build_kinematic_model(output_dir, part_plan, segmentation_summary, motion_observations, args)

    global_warnings = []
    if len(states) < args.min_states:
        global_warnings.append(
            f"Only {len(states)} states found; {args.min_states}+ states are recommended for joint fitting."
        )
    if not mesh_info.get("exists"):
        global_warnings.append("Input mesh is missing; downstream registration cannot run.")
    global_warnings.extend(mesh_info.get("warnings", []))
    global_warnings.extend(part_plan.get("warnings", []))
    global_warnings.extend(segmentation_summary.get("warnings", []))
    global_warnings.extend(completion_summary.get("warnings", []))

    state_manifest_paths = _write_state_manifests(output_dir, state_reports)
    part_manifest_paths = _write_part_manifests(output_dir, part_plan)
    model = _build_articulated_object_model(
        session,
        session_path,
        output_dir,
        mesh_info,
        state_reports,
        part_plan,
        segmentation_summary,
        completion_summary,
        motion_observations,
        kinematic_model,
    )
    model_path = os.path.join(output_dir, "articulated_object_model.json")
    _write_json(model_path, model)

    manifest = {
        "schema": "paradex.articulated_object.pipeline_manifest.v0",
        "session_path": session_path,
        "output_dir": output_dir,
        "object_name": session.get("object_name"),
        "session_name": session.get("session_name", os.path.basename(session_path)),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": "check_only" if args.check_only else ("segmentation_only" if args.segmentation_only else "preprocess"),
        "expected_cameras": expected_cameras,
        "input_mesh": mesh_info,
        "part_plan": part_plan,
        "global_warnings": global_warnings,
        "states": state_reports,
        "multiview": {
            "enabled": not args.check_only and not args.skip_multiview,
            "state_summaries": multiview_summaries,
        },
        "object_roi_crop": object_roi_summary,
        "placement_groups": placement_group_summary,
        "background_plates": background_plate_summary,
        "visual_hull": visual_hull_summary,
        "silhouette_evidence": silhouette_evidence_summary,
        "silhouette_orientation": silhouette_orientation_summary,
        "moving_part_carve": moving_part_carve_summary,
        "registration": {
            "enabled": not args.check_only and not args.skip_registration,
            "method": args.registration_method,
            "refine_method": args.registration_refine_method,
            "scale_mode": args.registration_scale_mode,
            "state_summaries": registration_summaries,
        },
        "segmentation": segmentation_summary,
        "part_registration": {
            "enabled": (
                not args.check_only
                and not args.skip_registration
                and not args.skip_part_registration
                and bool(_part_registration_candidates(part_plan))
            ),
            "refine_method": args.part_registration_refine_method,
            "target_scope": "part_mask_sparse_pointcloud_when_available",
            "mask_root": args.part_mask_root,
            "require_part_masks": args.require_part_masks,
            "candidate_part_count": len(_part_registration_candidates(part_plan)),
            "state_summaries": part_registration_summaries,
        },
        "completion": completion_summary,
        "motion_observations": motion_observations,
        "kinematic_model": kinematic_model,
        "state_manifest_paths": state_manifest_paths,
        "part_manifest_paths": part_manifest_paths,
        "articulated_object_model_path": model_path,
        "next_steps": [
            "Use processed/multiview/<state_id>/sparse_points.ply as the first sparse 3D observation for mesh registration.",
            "Use processed/multiview/<state_id>/colmap_text when a COLMAP/MVS backend is added.",
            "Use processed/registration/<state_id>/registration.json T_world_object as the current whole-object pose estimate.",
            "Use coarse_T_world_object when ICP refinement is rejected or unstable.",
            "Use processed/segmentation/parts/*.obj as the generated canonical part meshes.",
            "Use processed/part_registration/<state_id>/part_registration.json as candidate per-part pose observations when part meshes are provided.",
            "Use processed/completion/parts/<part_id>/completion_points.ply and completed_surfel_mesh.obj as sparse hidden-surface completion candidates.",
            "Use processed/motion_observations/motion_observations.json as the direct input for later joint fitting.",
            "Use processed/kinematic_model/kinematic_model.json as the offline articulated-object model candidate.",
            "Provide part masks under <state>/masks/<part_id>/<serial>.png or --part-mask-root to make per-part registration use segmented sparse targets.",
            "Do not run joint estimation until image completeness, sparse point quality, and part pose consistency are acceptable.",
        ],
    }
    visual_debug_report_path = _write_visual_debug_report(output_dir, manifest)
    manifest["visual_debug_report_path"] = visual_debug_report_path
    manifest_path = os.path.join(output_dir, "pipeline_manifest.json")
    _write_json(manifest_path, manifest)
    _write_json(os.path.join(session_path, "calculate_manifest.json"), manifest)
    _write_json(os.path.join(session_path, "calc_manifest.json"), manifest)

    for warning in global_warnings:
        print(f"[WARN] {warning}")
    print(f"Saved pipeline manifest: {manifest_path}")
    print(f"Saved articulated-object model stub: {model_path}")
    print(f"Saved visual debug report: {visual_debug_report_path}")

    if args.strict:
        remaining = list(global_warnings)
        for report in state_reports:
            remaining.extend(report["warnings"])
        if remaining:
            raise RuntimeError(f"Strict preprocessing failed with warnings: {remaining}")


if __name__ == "__main__":
    main()
