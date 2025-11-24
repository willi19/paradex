# paradex/image/collection.py
"""Image collection with batch operations"""

from typing import Dict, Callable, Optional, Any, Union
from pathlib import Path
import numpy as np
import cv2
import json
import os

from paradex.calibration.util import load_camparam, load_current_camparam, get_cammtx
from paradex.image.undistort import apply_undistort_map, precomute_undistort_map
from paradex.image.aruco import detect_aruco
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
            if self._cache['undistort_map'] is None:
                self._cache['undistort_map'] = {}
                for serial in self.images.keys():
                    self._cache['undistort_map'][serial] = precomute_undistort_map(self.intrinsic[serial])
            if serial not in self.intrinsic:
                raise ValueError(f"Intrinsic parameters not found for serial {serial}")
        
        undistorted_images = self.map_images(
            func=apply_undistort_map,
            use_camparam=False,
            mapx=self._cache['undistort_map'][serial][1],
            mapy=self._cache['undistort_map'][serial][2]
        )
        
        if save_path is not None:
            undistorted_images.save(save_path)
            
        elif self.path is not None:
            undistorted_images.save(self.path)
            
        return undistorted_images
    
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
        return marker_3d
        
    def project():
        pass
    
    
    def merge(self, image_text: Optional[Dict[str, str]] = None) -> np.ndarray:
        """Merge all images into a grid layout"""
        return merge_image(self.images, image_text)
    
    def project_mesh_nvdiff(object, renderer):
        # from paradex.visualization_.renderer import BatchRenderer
        
        img, mask = renderer.render(object)
        return img, mask
    
    # def project_points(self, points_3d: np.ndarray, 
    #                   color: tuple = (255, 0, 0)) -> 'ImageDict':
    #     """Project 3D points onto all camera views"""
    #     from paradex.image.projection import get_cammtx, project_point
        
    #     cammat = get_cammtx(self.intrinsic, self.extrinsic)
        
    #     projected_imgs = {}
    #     for serial, img in self.images.items():
    #         if serial in cammat:
    #             projected_imgs[serial] = project_point(
    #                 points_3d, cammat[serial], img.copy(), color
    #             )
    #         else:
    #             projected_imgs[serial] = img.copy()
        
    #     return ImageDict(projected_imgs, self.intrinsic, self.extrinsic)