from multiprocessing import shared_memory
import numpy as np
import time
import os
import re

from paradex.utils.system import get_camera_list
from paradex.utils.path import pc_name

class CameraReader:
    """Shared memory에서 카메라 이미지를 읽어오는 클래스"""
    
    def __init__(self, camera_name, frame_shape=(1536, 2048, 3), timeout=5.0):
        """
        Args:
            camera_name: 카메라 이름 (shared memory 이름에 사용)
            frame_shape: 프레임 shape (height, width, channels)
            timeout: shared memory 연결 대기 시간 (초)
        """
        self.name = camera_name
        self.frame_shape = frame_shape
        self.timeout = timeout
        
        self._connect_shared_memory()
    
    def _connect_shared_memory(self):
        """기존 shared memory에 연결"""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            try:
                # Buffer 2개에 연결
                self.image_shm_a = shared_memory.SharedMemory(
                    name=self.name + "_image_a"
                )
                self.image_shm_b = shared_memory.SharedMemory(
                    name=self.name + "_image_b"
                )
                
                # Frame ID 2개에 연결
                self.fid_shm_a = shared_memory.SharedMemory(
                    name=self.name + "_fid_a"
                )
                self.fid_shm_b = shared_memory.SharedMemory(
                    name=self.name + "_fid_b"
                )
                
                # Write buffer flag에 연결
                self.write_flag_shm = shared_memory.SharedMemory(
                    name=self.name + "_flag"
                )
                
                # Arrays 생성
                self.image_array_a = np.ndarray(
                    self.frame_shape, dtype=np.uint8, buffer=self.image_shm_a.buf
                )
                self.image_array_b = np.ndarray(
                    self.frame_shape, dtype=np.uint8, buffer=self.image_shm_b.buf
                )
                self.fid_array_a = np.ndarray(
                    (1,), dtype=np.int64, buffer=self.fid_shm_a.buf
                )
                self.fid_array_b = np.ndarray(
                    (1,), dtype=np.int64, buffer=self.fid_shm_b.buf
                )
                self.write_flag = np.ndarray(
                    (1,), dtype=np.uint8, buffer=self.write_flag_shm.buf
                )
                
                print(f"Successfully connected to shared memory for camera: {self.name}")
                return
                
            except FileNotFoundError:
                time.sleep(0.1)
        
        raise RuntimeError(
            f"Failed to connect to shared memory for camera '{self.name}' "
            f"within {self.timeout} seconds. Make sure the camera is initialized."
        )
    
    def get_image(self, copy=True):
        """
        현재 읽을 수 있는 이미지와 frame ID를 가져옴
        
        Args:
            copy: True면 이미지 복사본 반환, False면 shared memory 참조 반환
            
        Returns:
            tuple: (image, frame_id)
                - image: numpy array (frame_shape)
                - frame_id: int, 프레임 ID (0이면 아직 데이터 없음)
        """
        # write_flag의 반대 버퍼에서 읽기
        if self.write_flag[0] == 0:
            # Writer가 buffer A에 쓰는 중이므로 B에서 읽기
            image = self.image_array_b
            frame_id = int(self.fid_array_b[0])
        else:
            # Writer가 buffer B에 쓰는 중이므로 A에서 읽기
            image = self.image_array_a
            frame_id = int(self.fid_array_a[0])
        
        if copy:
            return image.copy(), frame_id
        else:
            return image, frame_id
    
    def wait_for_new_frame(self, last_frame_id=None, timeout=1.0, poll_interval=0.001):
        """
        새로운 프레임이 올 때까지 대기
        
        Args:
            last_frame_id: 마지막으로 받은 frame ID (None이면 현재 frame ID 사용)
            timeout: 최대 대기 시간 (초)
            poll_interval: 폴링 간격 (초)
            
        Returns:
            tuple: (image, frame_id) or (None, None) if timeout
        """
        if last_frame_id is None:
            _, last_frame_id = self.get_image(copy=False)
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            image, frame_id = self.get_image(copy=True)
            
            if frame_id > last_frame_id and frame_id > 0:
                return image, frame_id
            
            time.sleep(poll_interval)
        
        return None, None
    
    def is_active(self):
        """카메라가 현재 활성화되어 있는지 확인 (frame_id > 0)"""
        _, frame_id = self.get_image(copy=False)
        return frame_id > 0
    
    def get_frame_id(self):
        """현재 frame ID만 가져옴"""
        _, frame_id = self.get_image(copy=False)
        return frame_id
    
    def close(self):
        """Shared memory 연결 해제"""
        self.image_array_a = None
        self.image_array_b = None
        self.fid_array_a = None
        self.fid_array_b = None
        self.write_flag = None
        
        self.image_shm_a.close()
        self.image_shm_b.close()
        self.fid_shm_a.close()
        self.fid_shm_b.close()
        self.write_flag_shm.close()
        
        print(f"Closed shared memory connection for camera: {self.name}")
    

class MultiCameraReader:
    """여러 카메라의 shared memory를 동시에 읽는 클래스"""
    
    def __init__(self, camera_names=None, frame_shape=(1536, 2048, 3), timeout=5.0):
        """
        Args:
            camera_names: 카메라 이름 리스트 (None이면 자동으로 탐색)
            frame_shape: 프레임 shape
            timeout: shared memory 연결 대기 시간
        """
        # 카메라 목록이 제공되지 않으면 자동 탐색
        if camera_names is None:
            camera_names = get_camera_list(pc_name)
            if not camera_names:
                raise RuntimeError("No cameras found. Please start cameras first.")
        
        self.readers = [
            CameraReader(name, frame_shape, timeout) 
            for name in camera_names
        ]
        self.camera_names = camera_names
    
    def get_images(self, copy=True):
        """
        모든 카메라의 이미지를 가져옴
        
        Returns:
            dict: {camera_name: (image, frame_id)}
        """
        return {
            name: reader.get_image(copy=copy)
            for name, reader in zip(self.camera_names, self.readers)
        }
    
    def wait_for_new_frames(self, last_frame_ids=None, timeout=1.0):
        """
        모든 카메라의 새 프레임 대기
        
        Args:
            last_frame_ids: dict {camera_name: last_frame_id} or None
            timeout: 최대 대기 시간
            
        Returns:
            dict: {camera_name: (image, frame_id)}
        """
        if last_frame_ids is None:
            last_frame_ids = {name: None for name in self.camera_names}
        
        return {
            name: reader.wait_for_new_frame(
                last_frame_ids.get(name), 
                timeout=timeout
            )
            for name, reader in zip(self.camera_names, self.readers)
        }
    
    def get_frame_ids(self):
        """모든 카메라의 frame ID 가져옴"""
        return {
            name: reader.get_frame_id()
            for name, reader in zip(self.camera_names, self.readers)
        }
    
    def are_active(self):
        """각 카메라의 활성 상태 확인"""
        return {
            name: reader.is_active()
            for name, reader in zip(self.camera_names, self.readers)
        }
    
    def close(self):
        """모든 shared memory 연결 해제"""
        for reader in self.readers:
            reader.close()
    