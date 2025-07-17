import json
from threading import Event, Lock, Thread
import time
import numpy as np
import cv2
import os

import PySpin as ps

from paradex.io.camera.camera import Camera
from paradex.utils.file_io import home_path, config_dir

def spin2cv(pImg, h, w):
    """
    Convert Spinnaker image to OpenCV format.

    Args:
        pImg: Spinnaker image object
        h (int): Image height
        w (int): Image width

    Returns:
        cvImg (np.ndarray): Converted OpenCV image
    """
    image_data = pImg.GetData()
    cvImg = np.array(image_data, dtype=np.uint8).reshape((h, w))
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BayerRG2RGB)
    return cvImg

class CameraManager:
    def __init__(self, mode, serial_list = None, syncMode=True):
        self.exit = Event()
        self.start_capture = Event()
    
        self.mode = mode
        if self.mode == "image":
            syncMode = False

        self.autoforce_ip()

        self.connected_serial_list = self.get_serial_list()
        
        if serial_list is None:
            self.serial_list = self.connected_serial_list

        else:
            for serial in serial_list:
                serial = str(serial)
                if serial not in self.connected_serial_list:
                    raise ValueError(f"Camera with serial {serial} not found.")
            self.serial_list = serial_list

        for i in range(len(self.serial_list)):
            self.serial_list[i] = str(self.serial_list[i])
            
        self.num_cameras = len(self.serial_list)

        self.connect_flag = [Event() for _ in range(self.num_cameras)]
        self.connect_success = [Event() for _ in range(self.num_cameras)]
        
        self.cam_start_flag = [Event() for _ in range(self.num_cameras)]
        self.capture_end_flag = [Event() for _ in range(self.num_cameras)]
        
        self.save_finish_flag = [Event() for _ in range(self.num_cameras)]

        self.height, self.width = 1536, 2048

        if self.mode == "stream":
            self.image_array = np.zeros((self.num_cameras, self.height, self.width, 3), dtype=np.uint8)
            self.frame_num = np.zeros((self.num_cameras,), dtype=np.uint64)
            self.locks = [Lock() for _ in range(self.num_cameras)]

        self.camera_threads = []
        self.syncMode = syncMode

        self.lens_info = json.load(open(os.path.join(config_dir, "camera/lens_info.json"), "r"))
        self.cam_info = json.load(open(os.path.join(config_dir,"camera/camera.json"), "r"))

        self.capture_threads = [
            Thread(target=self.run_camera, args=(i,))
            for i in range(self.num_cameras)
        ]

        for i in range(self.num_cameras):
            self.capture_threads[i].start()
        
        ret = self.wait_for_connection()
        if not ret:
            self.quit()
            for i in range(self.num_cameras):
                self.capture_threads[i].join()
            raise RuntimeError("Failed to connect to all cameras.")

    def get_serial_list(self):
        system = ps.System.GetInstance()
        cam_list = system.GetCameras()
        serial_list = []

        for cam in cam_list:
            device_nodemap = cam.GetTLDeviceNodeMap()
            serialnum_entry = device_nodemap.GetNode(
                "DeviceSerialNumber"
            )  # .GetValue()
            serialnum = ps.CStringPtr(serialnum_entry).GetValue()
            serial_list.append(serialnum)
            del cam

        cam_list.Clear()
        system.ReleaseInstance()

        return serial_list

    def autoforce_ip(self):
        system = ps.System.GetInstance()
        interfaceList = system.GetInterfaces() # virtual port included
        for pInterface in interfaceList:
            nodeMapInterface = pInterface.GetTLNodeMap()
            camera_list = pInterface.GetCameras()
            cam_num = len(camera_list)
            camera_list.Clear()

            if cam_num == 1:
                curIPNode = nodeMapInterface.GetNode("GevDeviceIPAddress")    
                if ps.IsAvailable(curIPNode) and ps.IsReadable(curIPNode):
                    ip_int = ps.CIntegerPtr(curIPNode).GetValue()
                
                ip_str = f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"
                if ip_str[:2] != "11":
                    ptrAutoForceIP = nodeMapInterface.GetNode("GevDeviceAutoForceIP")
                    if ps.IsAvailable(ptrAutoForceIP) and ps.IsWritable(ptrAutoForceIP) and ps.IsWritable(pInterface.TLInterface.DeviceSelector.GetAccessMode()):
                        pInterface.TLInterface.DeviceSelector.SetValue(0)
                        pInterface.TLInterface.GevDeviceAutoForceIP.Execute()

            del pInterface
            
        
        interfaceList.Clear()
        system.ReleaseInstance()
        return

    def get_videostream(self, savePath):
        videoStream = ps.SpinVideo()

        videoOption = ps.AVIOption()

        videoOption.frameRate = 30
        videoOption.height = self.height
        videoOption.width = self.width

        videoStream.Open(str(savePath), videoOption)
        return videoStream

    def wait_for_connection(self):
        for i in range(self.num_cameras):
            self.connect_flag[i].wait()
            if not self.connect_success[i].is_set():
                print(f"Camera {self.serial_list[i]} failed to connect.")
                return False
        return True
    
    def wait_for_camstart(self):
        for i in range(self.num_cameras):
            self.cam_start_flag[i].wait()
            
    def wait_for_saveend(self):
        for i in range(self.num_cameras):
            self.capture_end_flag[i].wait()
    
    def run_camera(self, index):
        serial_num = self.serial_list[index]

        system = ps.System.GetInstance()
        cam_list = system.GetCameras()

        camPtr = cam_list.GetBySerial(serial_num)

        lens_id = str(self.cam_info[serial_num]["lens"])

        gain = self.lens_info[lens_id]["Gain"]
        exposure = self.lens_info[lens_id]["Exposure"]
        frame_rate = self.lens_info[lens_id]["fps"]
        
        try:
            cam = Camera(camPtr, gain, exposure, frame_rate, self.mode, self.syncMode)

            self.connect_flag[index].set()
            self.connect_success[index].set()

        except Exception as e:
            print(f"Failed to connect to camera {serial_num}: {e}")
            self.connect_flag[index].set()

            cam.release()    
            del camPtr
            cam_list.Clear()
            system.ReleaseInstance()

            return 

        while not self.exit.is_set():
            while not self.start_capture.is_set():
                time.sleep(0.01)
                if self.exit.is_set():
                    break
            if self.exit.is_set():
                break
            
            if self.mode == "video":
                save_dir = f"{home_path}/captures{index // 2 + 1}/{self.save_dir}"
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"{serial_num}")
                videostream = self.get_videostream(save_path)

                timestamp_path = os.path.join(save_dir, f"{serial_num}_timestamps.json")
                timestamps = dict([("timestamps", []), ("frameID", []), ("pc_time", [])])

            if self.mode == "image":
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = os.path.join(self.save_dir, f"{serial_num}.png")
            
            if self.mode == "stream":
                self.frame_num[index] = 0

            cam.start()  
            if self.syncMode:
                self.cam_start_flag[index].set()
            # Todo
            # Potential error part. If we use automatic sync then we might start signal generator before starting the camera
            # We handle this by waiting 1 second but we should track with thread
            # cam start takes 0.02 second so should not be a problem(maybe)
            
            while self.start_capture.is_set():
                if self.exit.is_set():
                    break
                raw_frame = cam.get_image()
                framenum = raw_frame.GetFrameID()
                
                if raw_frame.IsIncomplete():
                    if self.mode == "stream":
                        with self.locks[index]:
                            np.copyto(self.image_array[index], np.zeros((self.height, self.width, 3), dtype=np.uint8))
                            self.frame_num[index] = framenum
                    continue
                
                if self.mode == "video":
                    videostream.Append(raw_frame)
                    timestamps["timestamps"].append(raw_frame.GetChunkData().GetTimestamp())
                    timestamps["frameID"].append(framenum)
                    timestamps["pc_time"].append(time.time())

                elif self.mode == "image":
                    frame = spin2cv(raw_frame, self.height, self.width)
                    cv2.imwrite(save_path, frame)

                    self.start_capture.clear()
                    self.capture_end_flag[index].set()
                    break

                elif self.mode == "stream":
                    with self.locks[index]:
                        np.copyto(self.image_array[index], spin2cv(raw_frame, self.height, self.width))
                        self.frame_num[index] = framenum
                raw_frame.Release()
            
            if self.mode == "video":
                json.dump(
                    timestamps, open(timestamp_path, "w"), indent="\t"
                )
                videostream.Close()
            cam.stop()
            
            self.capture_end_flag[index].set()
            
        
        cam.release()
        del camPtr
        cam_list.Clear()
        system.ReleaseInstance()
    
    def wait_for_capture_end(self):
        for i in range(self.num_cameras):
            self.capture_end_flag[i].wait()    

    def start(self, save_dir=None):
        self.save_dir = save_dir
        for index in range(self.num_cameras):
            if self.save_dir is not None:
                self.capture_end_flag[index].clear()
            if self.syncMode:
                self.cam_start_flag[index].clear()
            
        self.start_capture.set()
        if self.syncMode:
            self.wait_for_camstart()
            
    def end(self):
        for i in range(self.num_cameras):
            self.save_finish_flag.clear()
        self.start_capture.clear()
        if self.save_dir is not None:
            self.wait_for_saveend()

    def quit(self):
        self.exit.set()
        for p in self.capture_threads:
            p.join()

    def get_frameid(self, index):
        if self.mode != "stream":
            return
        
        with self.locks[index]:
            return self.frame_num[index]
        
    def get_data(self, index):
        if self.mode != "stream":
            return 
        
        with self.locks[index]:
            return {"image":self.image_array[index].copy(), "frameid":self.frame_num[index]}
        
if __name__ == "__main__":
    manager = CameraManager(num_cameras=4, duration=180, save_dir="/home/capture16/captures1", is_streaming=False)
    manager.start()
