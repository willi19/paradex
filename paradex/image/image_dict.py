# paradex/image/collection.py
"""Image collection with batch operations"""

from typing import Dict, Callable, Optional, Any, Union
from pathlib import Path
import numpy as np
import cv2
import json
import os
import time

from paradex.calibration.utils import load_camparam, load_current_camparam, get_cammtx
from paradex.image.undistort import apply_undistort_map, precomute_undistort_map, undistort_img
from paradex.image.overlay import overlay_mask
from paradex.image.aruco import detect_aruco, detect_charuco
from paradex.transforms.triangulate import triangulation
from paradex.image.merge import merge_image

class ImageDict:
    def __init__(self, 
                 images: Dict[str, np.ndarray], 
                 intrinsic: Optional[Dict[str, Dict]] = None,
                 extrinsic: Optional[Dict[str, np.ndarray]] = None,
                 path: Optional[Union[str, Path]] = None):
        """
        Args:
            images: Dictionary mapping camera serial numbers to images
            intrinsic: Camera intrinsic parameters per serial number
            extrinsic: Camera extrinsic parameters per serial number
        """
        self.images = images
        self.intrinsic = intrinsic or {}
        self.extrinsic = extrinsic or {}
        self.path = path
        
        self._cache = {}  # For caching results of operations
    
    @classmethod
    def from_path(cls, 
                  path: Union[str, Path]) -> 'ImageDict':
        
        path = Path(path)
        images_dir = None
        
        if (path / "images").exists():
            images_dir = path / "images"
        
        elif (path / "raw" / "images").exists(): # undistorted images
            images_dir = path / "raw" / "images"
        
        else:
            images_dir = path
        
        calib_dir = path if (path / "cam_param").exists() else None
        
        # Load images
        images = {}
        
        for img_file in os.listdir(images_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                serial = Path(img_file).stem  # filename without extension
                img = cv2.imread(str(images_dir / img_file))
                if img is not None:
                    images[serial] = img
        
        if not images:
            raise ValueError(f"No images found in {path}")
        
        # Load calibration
        intrinsic = {}
        extrinsic = {}
        
        if calib_dir:
            intrinsic, extrinsic = load_camparam(calib_dir)
            
        ret =  cls(images, intrinsic, extrinsic, path)
        return ret
        
    # Dictionary-like interface
    def __getitem__(self, serial: str) -> np.ndarray:
        """Get image by serial number"""
        return self.images[serial]
    
    def __setitem__(self, serial: str, img: np.ndarray):
        """Set image by serial number"""
        self.images[serial] = img
    
    def __len__(self) -> int:
        """Number of cameras"""
        return len(self.images)
    
    def __iter__(self):
        """Iterate over (serial, image) pairs"""
        return iter(self.images.items())
    
    def __contains__(self, serial: str) -> bool:
        """Check if serial number exists"""
        return serial in self.images
    
    def update(self, images: Dict[str, np.ndarray]):
        """Update images with another dictionary"""
        self.images.update(images)
        
    def update_path(self, path: Union[str, Path], reload_images: bool = True):
        """
        Update path and optionally reload images.
        
        Args:
            path: New path
            reload_images: If True, reload images from new path
            
        Note:
            Preserves undistort maps if calibration hasn't changed.
        """
        self.path = path
        
        if not reload_images:
            return
        
        images_dir = Path(path)
        if (images_dir / "images").exists():
            images_dir = images_dir / "images"
        elif (images_dir / "raw" / "images").exists():
            images_dir = images_dir / "raw" / "images"
        
        new_images = {}
        for img_file in os.listdir(images_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                serial = Path(img_file).stem
                if serial in self.serial_list:  # Only load existing serials
                    img = cv2.imread(str(images_dir / img_file))
                    if img is not None:
                        new_images[serial] = img
        
        self.images = new_images
        
    def set_camparam(self, 
                     intrinsic: Dict[str, Dict], 
                     extrinsic: Dict[str, np.ndarray]):
        """Set camera parameters"""
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic

    @property
    def serial_list(self):
        """List of camera serial numbers"""
        return list(self.images.keys())
    
    def keys(self):
        return self.images.keys()
    
    def values(self):
        return self.images.values()
    
    def items(self):
        return self.images.items()
    
    def load_current_camparam(self):
        intrinsic, extrinsic = load_current_camparam()
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
    
    def copy(self) -> 'ImageDict':
        """Create a deep copy of ImageDict"""
        import copy
        return ImageDict(
            copy.deepcopy(self.images),
            copy.deepcopy(self.intrinsic),
            copy.deepcopy(self.extrinsic)
        )
    
    def __repr__(self):
        return f"ImageDict(serials={self.serials}, n_cameras={len(self)})"
    
    def save(self, path=None):
        save_path = path or self.path
        if save_path is None:
            raise ValueError("No path specified for saving images.")
        os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
        
        for serial, img in self.images.items():
            filename = f"{serial}.png"
            cv2.imwrite(os.path.join(save_path, "images", filename), img)

    def save_camparam(self, path=None):
        save_path = path or self.path
        if save_path is None:
            raise ValueError("No path specified for saving camera parameters.")

        calib_path = Path(save_path) / "cam_param"
        calib_path.mkdir(parents=True, exist_ok=True)
        intrinsic_path = calib_path / "intrinsics.json"
        extrinsic_path = calib_path / "extrinsics.json"
        
        if self.intrinsic is not None:
            with open(intrinsic_path, 'w') as f:
                json.dump(self.intrinsic, f, indent=4)
        
        if self.extrinsic is not None:     
            with open(extrinsic_path, 'w') as f:
                json.dump({k: v.tolist() for k, v in self.extrinsic.items()}, f, indent=4)

    def apply(self, func: Callable, use_camparam=False, **kwargs) -> Dict:
        """
        Apply a single-image function to all images.
        
        Args:
            func: Function with signature func(img, **kwargs) -> result
            cache_key: Optional key to cache results
            **kwargs: Additional arguments passed to func
            
        Returns:
            Dictionary mapping serial numbers to results
            
        Example:
            >>> from paradex.image.detection import detect_aruco
            >>> results = img_dict.apply(detect_aruco, dict_type='6X6_1000')
        """
        results = {}
        
        for serial, img in self.images.items():
            if use_camparam:
                if serial in self.intrinsic and serial in self.extrinsic:
                    results[serial] = func(
                        img, 
                        intrinsic=self.intrinsic[serial],
                        extrinsic=self.extrinsic[serial],
                        **kwargs
                    )
                else:
                    raise ValueError(f"Camera parameters not found for serial {serial}")
            else:
                results[serial] = func(img, **kwargs)
        
        return results
    
    def map_images(self, func: Callable, use_camparam=False, **kwargs) -> 'ImageDict':
        new_images = {}
        for serial, img in self.images.items():
            if use_camparam:
                if serial in self.intrinsic and serial in self.extrinsic:
                    new_images[serial] = func(
                        img, 
                        intrinsic=self.intrinsic[serial],
                        extrinsic=self.extrinsic[serial],
                        **kwargs
                    )
                else:
                    raise ValueError(f"Camera parameters not found for serial {serial}")
            else:
                new_images[serial] = func(img, **kwargs)
                
        return ImageDict(new_images, self.intrinsic, self.extrinsic, self.path)
    
    # Image processing operations
    def undistort(self, save_path: Optional[Union[str, Path]] = None) -> 'ImageDict':
        for serial, img in self.images.items():
            if 'undistort_map' not in self._cache:
                self._cache['undistort_map'] = {}
                for serial in self.images.keys():
                    self._cache['undistort_map'][serial] = precomute_undistort_map(self.intrinsic[serial])
            if serial not in self.intrinsic:
                raise ValueError(f"Intrinsic parameters not found for serial {serial}")
        
        undistort_image = {}
        for serial in self.images.keys():
            undistort_image[serial] = apply_undistort_map(
                self.images[serial],
                self._cache['undistort_map'][serial][1],
                self._cache['undistort_map'][serial][2]
            )

        undistort_image_dict = ImageDict(undistort_image, self.intrinsic, self.extrinsic, save_path)

        if save_path is not None:
            undistort_image_dict.save(save_path)

        elif self.path is not None and not (self.path / "images").exists():
            undistort_image_dict.save(self.path)
            
        return undistort_image_dict
    
    # # Multi-view geometry
    def triangulate_markers(self, dict_type: str = '6X6_1000') -> Dict[int, np.ndarray]:
        """Triangulate ArUco markers from multiple views"""
        
        if 'proj_mtx' not in self._cache:
            self._cache['proj_mtx'] = get_cammtx(self.intrinsic, self.extrinsic)
            
        result = self.apply(
            func=detect_aruco,
            dict_type=dict_type
        )
        
        marker_2d = {}
        for serial, (kypts, ids) in result.items():
            if ids is None or len(ids) == 0:
                continue
            
            for id, k in zip(ids.reshape(-1), kypts):
                k = k.squeeze()
                if id not in marker_2d:
                    marker_2d[id] = {"2d": [], "cammtx": []}
                cammtx = self._cache['proj_mtx'][serial]
                marker_2d[id]["2d"].append(k)
                marker_2d[id]["cammtx"].append(cammtx)

        marker_3d = {id: triangulation(np.array(marker_2d[id]["2d"]), np.array(marker_2d[id]["cammtx"]))
                     for id in marker_2d}
        return marker_2d, marker_3d

    def triangulate_charuco(self) -> Dict[str, np.ndarray]:
        """Triangulate ChArUco corners from multiple views"""
        
        if 'proj_mtx' not in self._cache:
            self._cache['proj_mtx'] = get_cammtx(self.intrinsic, self.extrinsic)
            
        result = self.apply(
            func=detect_charuco
        )
        
        charuco_2d = {}
        for serial, det_dict in result.items():
            for board_id in det_dict.keys():
                kypts = det_dict[board_id]["checkerCorner"]
                ids = det_dict[board_id]["checkerIDs"]
                
                if ids is None or len(ids) == 0:
                    continue
                
                if board_id not in charuco_2d:
                    charuco_2d[board_id] = {}
                    
                for id, k in zip(ids.reshape(-1), kypts):
                    k = k.squeeze()
                    if id not in charuco_2d[board_id]:
                        charuco_2d[board_id][id] = {"2d": [], "cammtx": []}
                    cammtx = self._cache['proj_mtx'][serial]
                    charuco_2d[board_id][id]["2d"].append(k)
                    charuco_2d[board_id][id]["cammtx"].append(cammtx)

        charuco_3d = {}
        for board_id, ids in charuco_2d.items():
            charuco_3d[board_id] = {'checkerIDs': [], 'checkerCorner': []}
            for id, data in ids.items():
                pt3d = triangulation(np.array(data["2d"]), np.array(data["cammtx"]))
                if board_id not in charuco_3d:
                    charuco_3d[board_id] = {}
                if pt3d is not None:
                    charuco_3d[board_id]['checkerIDs'].append(id)
                    charuco_3d[board_id]['checkerCorner'].append(pt3d)
            charuco_3d[board_id]['checkerIDs'] = np.array(charuco_3d[board_id]['checkerIDs'])
            charuco_3d[board_id]['checkerCorner'] = np.array(charuco_3d[board_id]['checkerCorner'])
            
        return charuco_3d
    
    def traingulate_points(self, points_2d: np.ndarray) -> np.ndarray:
        """Triangulate 3D points from multiple views
        
        Args:
            points_2d: Dict mapping serial numbers to Nx2 arrays of 2D points   
        Returns:
            points_3d: Nx3 array of triangulated 3D points
        """
        if self._cache.get('proj_mtx') is None:
            self._cache['proj_mtx'] = get_cammtx(self.intrinsic, self.extrinsic)
        
        proj_mtxs = []
        pts_2d_list = []
        
        for serial, pts_2d in points_2d.items():
            if serial in self._cache['proj_mtx']:
                proj_mtxs.append(self._cache['proj_mtx'][serial])
                pts_2d_list.append(pts_2d)
            else:
                raise ValueError(f"Projection matrix not found for serial {serial}")
        
        points_3d = triangulation(
            points_2d=np.array(pts_2d_list),  # List of Nx2 arrays
            proj_mtx=np.array(proj_mtxs)      # List of 3x4 projection matrices
        )
        
        return points_3d
        
    def project_pointcloud(self, points_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """Project 3D points onto all camera views"""
        if self._cache.get('proj_mtx') is None:
            self._cache['proj_mtx'] = get_cammtx(self.intrinsic, self.extrinsic)
        
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)
            
        proj_points = {}
        for serial, proj_mtx in self._cache['proj_mtx'].items():
            homog_points = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # Nx4
            img_points_homog = (proj_mtx @ homog_points.T).T  # Nx3
            img_points = img_points_homog[:, :2] / img_points_homog[:, 2:3]  # Nx2
            proj_points[serial] = img_points
        
        return proj_points

    def project_mesh(self, object, color=(0, 255, 0), alpha=0.5):
        if self._cache.get('proj_mtx') is None:
            self._cache['proj_mtx'] = get_cammtx(self.intrinsic, self.extrinsic)
        
        if self._cache.get('render') is None:
            from paradex.image.projection import BatchRenderer
            self._cache['render'] = BatchRenderer(
                self.intrinsic,
                self.extrinsic
            )
        renderer = self._cache['render']
        # if type(object) == list:
        #     return renderer.render_multi(object) # color_dict, mask_dict, depth_dict, id_dict
        _, mask, _ =  renderer.render(object) # color_dict, mask_dict, depth_dict
        
        start_time = time.time()
        images = {}
        for serial, m in mask.items():
            images[serial] = overlay_mask(self.images[serial], m, color=color, alpha=alpha)
        
        new_img_dict = ImageDict(images, self.intrinsic, self.extrinsic, self.path)
        return new_img_dict
        
            
    def merge(self, image_text: Optional[Dict[str, str]] = None) -> np.ndarray:
        """Merge all images into a grid layout"""
        return merge_image(self.images, image_text)
    
    def draw_keypoint(self, keypoints: Dict[str, np.ndarray], 
                      color: tuple = (0, 255, 0), 
                      radius: int = 3, 
                      thickness: int = -1) -> 'ImageDict':
        """Draw keypoints on images
        
        Args:
            keypoints: Dict mapping serial numbers to Nx2 arrays of keypoints
            color: Color of keypoints
            radius: Radius of keypoint circles
            thickness: Thickness of circle outline (-1 for filled)
            
        Returns:
            New ImageDict with keypoints drawn
        """
        new_images = {serial: img.copy() for serial, img in self.images.items()}
        for serial, img in new_images.items():
            if serial in keypoints:
                for pt in keypoints[serial]:
                    x, y = map(int, pt)
                    cv2.circle(img, (x, y), radius, color, thickness)
        return ImageDict(new_images, intrinsic=self.intrinsic, extrinsic=self.extrinsic, path=self.path)