import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
import open3d as o3d
from scipy.spatial.distance import cdist
from skimage.draw import polygon

from paradex.utils.file_io import load_mesh

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
from scipy.spatial.distance import cdist
from skimage.draw import polygon
import cv2

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
from scipy.spatial.distance import cdist
from skimage.draw import polygon
import cv2
try:
    from scipy.ndimage import binary_fill_holes
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class O3DMeshCrossSection:
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        """
        Open3D 메쉬 초기화
        
        Args:
            mesh: Open3D TriangleMesh 객체
        """
        self.mesh = mesh
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        
        print(f"메쉬 정보:")
        print(f"- 정점 수: {len(self.vertices)}")
        print(f"- 삼각형 수: {len(self.faces)}")
        print(f"- X 범위: {self.vertices[:, 0].min():.3f} ~ {self.vertices[:, 0].max():.3f}")
        print(f"- Y 범위: {self.vertices[:, 1].min():.3f} ~ {self.vertices[:, 1].max():.3f}")
        print(f"- Z 범위: {self.vertices[:, 2].min():.3f} ~ {self.vertices[:, 2].max():.3f}")
    
    def line_plane_intersection(self, p1: np.ndarray, p2: np.ndarray, z_plane: float) -> Optional[np.ndarray]:
        """선분과 z평면의 교점을 구함"""
        if (p1[2] - z_plane) * (p2[2] - z_plane) > 0:
            return None
        
        if abs(p1[2] - p2[2]) < 1e-10:
            return None
        
        t = (z_plane - p1[2]) / (p2[2] - p1[2])
        
        if 0 <= t <= 1:
            intersection = p1 + t * (p2 - p1)
            return intersection
        
        return None
    
    def get_cross_section(self, z_plane: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        """z평면으로 메쉬를 자른 단면 선분들을 구함"""
        cross_section_segments = []
        
        for face in self.faces:
            v1 = self.vertices[face[0]]
            v2 = self.vertices[face[1]]
            v3 = self.vertices[face[2]]
            # print(v1[2] > z_plane, v2[2] > z_plane, v3[2] > z_plane)
            intersections = []
            
            intersection = self.line_plane_intersection(v1, v2, z_plane)
            if intersection is not None:
                intersections.append(intersection)
            
            intersection = self.line_plane_intersection(v2, v3, z_plane)
            if intersection is not None:
                intersections.append(intersection)

            intersection = self.line_plane_intersection(v3, v1, z_plane)
            if intersection is not None:
                intersections.append(intersection)
            
            if len(intersections) == 2:
                cross_section_segments.append((intersections[0], intersections[1]))
        
        return cross_section_segments
    
    def get_ordered_cross_section_contour(self, z_plane: float, tolerance=1e-6) -> List[np.ndarray]:
        """단면을 연결된 윤곽선으로 정리"""
        segments = self.get_cross_section(z_plane)
        if not segments:
            return []
        
        remaining_segments = segments.copy()
        contours = []
        
        while remaining_segments:
            current_contour = [remaining_segments[0][0], remaining_segments[0][1]]
            remaining_segments.pop(0)
            
            changed = True
            while changed and remaining_segments:
                changed = False
                current_end = current_contour[-1]
                
                for i, (p1, p2) in enumerate(remaining_segments):
                    if np.linalg.norm(current_end - p1) < tolerance:
                        current_contour.append(p2)
                        remaining_segments.pop(i)
                        changed = True
                        break
                    elif np.linalg.norm(current_end - p2) < tolerance:
                        current_contour.append(p1)
                        remaining_segments.pop(i)
                        changed = True
                        break
            
            contours.append(np.array(current_contour))
        
        return contours
    
    def cross_section_to_mask(self, z_plane: float, resolution=256, thickness=0) -> Tuple[np.ndarray, dict]:
        """
        단면을 2D 마스크로 변환
        
        Args:
            z_plane: 자를 z평면의 z좌표값
            resolution: 마스크 해상도 (resolution x resolution)
            method: 마스크 생성 방법
                - 'contour_fill': 윤곽선을 찾아서 내부를 채움 (추천)
                - 'point_density': 점 밀도 기반
                - 'rasterize': 선분 래스터화
            thickness: 마스크 두께 확장 (픽셀 단위)
                - 0: 원본 크기
                - 1-5: 경계를 N픽셀만큼 확장 (추천: 1-2)
        
        Returns:
            mask: 2D boolean 배열 (True=내부, False=외부)
            info: 변환 정보 (좌표계 변환 등)
        """
        segments = self.get_cross_section(z_plane)
        if not segments:
            return np.zeros((resolution, resolution), dtype=bool), {}
        
        # 단면 점들 추출 (X, Y 좌표만)
        points_2d = []
        for seg in segments:
            points_2d.extend([seg[0][:2], seg[1][:2]])
        points_2d = np.array(points_2d)
        
        # 좌표계 정보 계산
        x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
        y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
        
        # 정사각형으로 만들기 위해 패딩 추가
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
        x_range, y_range = x_max - x_min, y_max - y_min
        max_range = max(x_range, y_range) * 1.1  # 10% 패딩
        
        x_min_padded = x_center - max_range / 2
        x_max_padded = x_center + max_range / 2
        y_min_padded = y_center - max_range / 2
        y_max_padded = y_center + max_range / 2
        
        info = {
            'x_min': x_min_padded,
            'x_max': x_max_padded,
            'y_min': y_min_padded,
            'y_max': y_max_padded,
            'resolution': resolution,
            'pixel_size': max_range / resolution,
            'z_plane': z_plane
        }
        
        mask = self._mask_contour_fill(segments, resolution, info, thickness)
        
        return mask, info
    
    def _mask_contour_fill(self, segments, resolution, info, thickness=0) -> np.ndarray:
        """윤곽선 기반 마스크 생성 (가장 정확한 방법)"""
        contours = self.get_ordered_cross_section_contour(info['z_plane'])
        mask = np.zeros((resolution, resolution), dtype=bool)
        
        for contour in contours:
            if len(contour) < 3:  # 삼각형도 안 되는 윤곽선은 무시
                continue
            
            # 월드 좌표를 픽셀 좌표로 변환
            x_pixels = ((contour[:, 0] - info['x_min']) / (info['x_max'] - info['x_min']) * resolution).astype(int)
            y_pixels = ((contour[:, 1] - info['y_min']) / (info['y_max'] - info['y_min']) * resolution).astype(int)
            
            # 범위 클리핑
            x_pixels = np.clip(x_pixels, 0, resolution - 1)
            y_pixels = np.clip(y_pixels, 0, resolution - 1)
            
            # 다각형 채우기
            try:
                rr, cc = polygon(y_pixels, x_pixels, shape=(resolution, resolution))
                mask[rr, cc] = True
            except:
                # 다각형이 유효하지 않은 경우 점들로 마스크 생성
                mask[y_pixels, x_pixels] = True
        
        # 구멍 채우기 (닫힌 윤곽선 내부를 확실히 채움)
        if HAS_SCIPY:
            try:
                mask = binary_fill_holes(mask)
            except:
                # scipy가 있어도 실패하면 OpenCV 사용
                mask = self._fill_holes_opencv(mask)
        else:
            # scipy가 없으면 OpenCV 사용
            mask = self._fill_holes_opencv(mask)
        
        # 두께 확장
        if thickness > 0:
            mask = self._expand_mask(mask, thickness)
        
        return mask
        
    def _expand_mask(self, mask, thickness):
        """마스크를 지정된 두께만큼 확장 (morphological dilation)"""
        if thickness <= 0:
            return mask
        
        # OpenCV를 사용한 확장 (dilation)
        kernel_size = thickness * 2 + 1  # 홀수 크기 커널
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        mask_uint8 = mask.astype(np.uint8) * 255
        expanded_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
        
        return expanded_mask > 0
    
    def _fill_holes_opencv(self, mask):
        """OpenCV를 사용해서 구멍 채우기"""
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(mask_uint8, contours, 255)
        return mask_uint8 > 0
    
    def world_to_pixel(self, world_coords, info):
        """월드 좌표를 픽셀 좌표로 변환"""
        x_pixel = (world_coords[0] - info['x_min']) / (info['x_max'] - info['x_min']) * info['resolution']
        y_pixel = (world_coords[1] - info['y_min']) / (info['y_max'] - info['y_min']) * info['resolution']
        return np.array([x_pixel, y_pixel])
    
    def pixel_to_world(self, pixel_coords, info):
        """픽셀 좌표를 월드 좌표로 변환"""
        x_world = info['x_min'] + (pixel_coords[0] / info['resolution']) * (info['x_max'] - info['x_min'])
        y_world = info['y_min'] + (pixel_coords[1] / info['resolution']) * (info['y_max'] - info['y_min'])
        return np.array([x_world, y_world])

# 사용 예제
if __name__ == "__main__":
    # 메쉬 로드
    mesh = load_mesh("pringles")
    cross_section = O3DMeshCrossSection(mesh)
    
    z_min, z_max = cross_section.vertices[:, 2].min(), cross_section.vertices[:, 2].max()
    z_center = (z_min + z_max) / 2
    
    test_z = z_center+0.01
    info, mask = cross_section.cross_section_to_mask(test_z, resolution=256, thickness=1)
    