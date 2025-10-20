import PySpin as ps
from typing import Literal, Optional
import numpy as np
import cv2
import json
import os

from paradex.utils.file_io import config_dir

if os.path.exists(os.path.join(config_dir, "camera/lens_info.json")):
    lens_info = json.load(open(os.path.join(config_dir, "camera/lens_info.json"), "r"))
    cam_info = json.load(open(os.path.join(config_dir,"camera/camera.json"), "r"))

else:
    lens_info = {}
    cam_info = {}
    
def get_serial_list():
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

def load_camera(serialnum):
    system = ps.System.GetInstance()
    cam_list = system.GetCameras()

    try:
        camPtr = cam_list.GetBySerial(serialnum)
    except:
        raise ValueError(f"Camera with serial number {serialnum} not found.")
    
    if serialnum in cam_info and cam_info[serialnum]["lens_id"] in lens_info:
        lens_params = lens_info[cam_info[serialnum]["lens_id"]]
        gain = lens_params["gain"]
        exposure = lens_params["exposure_time"]
    
    else:
        gain = 10.0
        exposure = 10000.0
        
    return PyspinCamera(camPtr, gain, exposure)

class PyspinCameraConfig:
    """Camera configuration constants"""
    # Buffer settings
    VIDEO_BUFFER_COUNT = 10
    IMAGE_BUFFER_COUNT = 1
    
    # Network settings
    PACKET_SIZE = 9000
    PACKET_DELAY_MULTIPLIER = 2
    THROUGHPUT_ALIGNMENT = 16000
    
    # Acquisition settings
    IMAGE_TIMEOUT_MS = 100

class PyspinCameraConfigurationError(Exception):
    """Exception raised when camera configuration fails.
    
    This exception is raised during camera initialization or mode changes
    when configuration of camera parameters fails.
    """
    pass

class PyspinCamera():
    def __init__(
        self,
        camPtr,
        gain,
        exposure_time,
        frame_rate=0.2,
        mode="video",
        syncMode=True,
    ):
        """Initialize and configure a FLIR camera.
        
        :param camPtr: PySpin camera pointer
        :param gain: Camera gain (dB)
        :param exposure_time: Exposure time (microseconds)
        :param frame_rate: Frame rate in Hz (only when syncMode=False)
        :param mode: 'image', 'video', or 'stream'
        :param syncMode: Use hardware trigger if True
        """
        assert mode == "image" or mode == "video" or mode == "stream", "mode must be image or video"

        self.syncMode = syncMode  
        self.mode = mode
        self.gain = gain
        self.exposure_time = exposure_time
        self.frame_rate = frame_rate
        
        camPtr.Init()  # initialize camera
        self.cam = camPtr
        self.serialnum = self._serialnum()

        self.stream_nodemap = camPtr.GetTLStreamNodeMap() 
        self.nodeMap = camPtr.GetNodeMap() 
        
        self._configure()
                    
    def _serialnum(self)-> str:
        """Retrieve camera serial number.
        
        :return: Camera serial number
        :rtype: str
        """
        device_nodemap = self.cam.GetTLDeviceNodeMap()
        serialnum_entry = device_nodemap.GetNode(
            "DeviceSerialNumber"
        )  
        serialnum = ps.CStringPtr(serialnum_entry).GetValue()
        return serialnum
        
    def set_mode(self, mode, syncMode)-> None:
        """Change camera operation mode and reconfigure.
        
        :param mode: 'image', 'video', or 'stream'
        :param syncMode: Use hardware trigger if True
        """
        assert mode == "image" or mode == "video" or mode == "stream", "mode must be image or video"
        
        self.mode = mode
        self.syncMode = syncMode if mode != "image" else False

        # Reconfigure camera for new mode
        self._configureBuffer()
        self._configureAcquisition()
        if not self.syncMode:
            self._configureFrameRate()
        else:
            self._configureTrigger()
        
    def get_image(self):
        """Get next image from camera.
    
        Image mode: Retries with 100ms timeout, auto-restarts on failure.
        Video/Stream mode: Blocks until image available.
        
        :return: Raw image pointer (call Release() after use)
        :rtype: PySpin.ImagePtr
        """
        if self.mode == "image":
            while True:
                try:
                    pImageRaw = self.cam.GetNextImage(100)
                    return self._spin2cv(pImageRaw, pImageRaw.GetHeight(), pImageRaw.GetWidth())
                    
                except:
                    self.stop()
                    self.start()
        else:
            pImageRaw = self.cam.GetNextImage()
            return self._spin2cv(pImageRaw, pImageRaw.GetHeight(), pImageRaw.GetWidth())

    def start(self):
        """
        Start image acquisition.
        """
        self.cam.BeginAcquisition()
        return
    
    def stop(self):
        """
        Stop image acquisition. Camera connection remains active.
        
        Note: Only stops acquiring images. Use release() to disconnect camera.
        """
        self.cam.EndAcquisition()
        return
    
    def release(self):
        """Release camera resources. Call stop() first. Cannot reuse after."""
        self.cam.DeInit()
        del self.cam
        return
    
    @staticmethod
    def _spin2cv(pImg, h, w):
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

    @staticmethod
    def _get_node(nodemap, name, node_type, readable=True, writable=True) -> ps.BasePtr:
        """Retrieve a camera node by name and type."""
        node = nodemap.GetNode(name)
        if node_type == "bool":
            node = ps.CBooleanPtr(node)
        elif node_type == "int":
            node = ps.CIntegerPtr(node)
        elif node_type == "float":
            node = ps.CFloatPtr(node)
        elif node_type == "enum":
            node = ps.CEnumerationPtr(node)
        elif node_type == "string":
            node = ps.CStringPtr(node)
        else:
            raise ValueError(f"Unsupported node type: {node_type}")

        if not ps.IsAvailable(node) or (readable and not ps.IsReadable(node)) or (writable and not ps.IsWritable(node)):
            raise CameraConfigurationError(f"Unable to get or set {name} (node retrieval). Aborting...")
        return node

    @staticmethod
    def _set_node_value(node, node_type, value)-> None:
        """Set the value of a camera node."""
        if node_type == "enum":
            enum_entry = ps.CEnumEntryPtr(node.GetEntryByName(value))
            if not ps.IsReadable(enum_entry):
                raise PyspinCameraConfigurationError(f"Unable to get {value} (enum entry retrieval). Aborting...")
            node.SetIntValue(enum_entry.GetValue())
        
        if node_type == "float":
            node.SetValue(value)

    def _configure(self) -> None:
        """configure camera settings based on initialization parameters."""
        self._configureGain()
        self._configureThroughPut()

        if not self.syncMode:
            self._configureFrameRate()  # we use trigger anyway
        else:
            self._configureTrigger()
        
        self._configurePacketSize(self.nodeMap)
        self._configurePacketDelay(self.nodeMap)
        self._configureExposure(self.nodeMap)
        self._configureAcquisition(self.nodeMap)
        self._configureChunk(self.nodeMap)
        self._configureBuffer(self.stream_nodemap)

    def _configureGain(self) -> None:
        """configure camera gain settings."""

        # Disable automatic gain
        ptrGainAuto = self._get_node(self.nodeMap, "GainAuto", "enum", readable=True, writable=True)
        self._set_node_value(ptrGainAuto, "enum", "Off")
        
        Gainval = self._get_node(self.nodeMap, "Gain", "float", readable=True, writable=True)

        # Ensure desired exposure time does not exceed the maximum
        self.gain = min(self.gain, Gainval.GetMax())
        
        # Set gain value
        self._set_node_value(Gainval, "float", self.gain)
        
    def _configureThroughPut(self) -> None:
        """configure camera throughput settings."""
        
        ThroughputLimit = self._get_node(self.nodeMap, "DeviceLinkThroughputLimit", "int", readable=True, writable=True)

        # Align to nearest multiple of THROUGHPUT_ALIGNMENT (value must be multiple of 16000)
        ValMax = ThroughputLimit.GetMax()
        ValMin = ThroughputLimit.GetMin()
        posValMax = ((ValMax - ValMin) // PyspinCameraConfig.THROUGHPUT_ALIGNMENT) * PyspinCameraConfig.THROUGHPUT_ALIGNMENT + ValMin 
        
        # Set throughput limit 
        ThroughputLimit.SetValue(posValMax)
        
    def _configureFrameRate(self) -> None:
        """configure camera frame rate settings."""

        # Disable trigger mode
        triggermode = self._get_node(self.nodeMap, "TriggerMode", "enum", readable=True, writable=True)
        self._set_node_value(triggermode, "enum", "Off")

        # Disable automatic frame rate
        try:
            autoframerate = self._get_node(self.nodeMap, "AcquisitionFrameRateAuto", "enum", readable=True, writable=True)
            self._set_node_value(autoframerate, "enum", "Off")
        except:
            # New spinnaker version 
            autoframerate = self._get_node(self.nodeMap, "AcquisitionFrameRateEnable", "bool", readable=True, writable=True)
            self._set_node_value(autoframerate, "bool", False)

        # Set frame rate value
        framerate = self._get_node(self.nodeMap, "AcquisitionFrameRate", "float", readable=True, writable=True)
        self._set_node_value(framerate, "float", self.frame_rate)

    def _configureTrigger(self) -> None:
        """configure camera trigger settings for hardware synchronization."""
        triggerMode = self._get_node(self.nodeMap, "TriggerMode", "enum", readable=True, writable=True)
        self._set_node_value(triggerMode, "enum", "On")

        # What line to use as trigger (Black)
        triggerSource = self._get_node(self.nodeMap, "TriggerSource", "enum", readable=True, writable=True)
        self._set_node_value(triggerSource, "enum", "Line0")

        triggerSelector = self._get_node(self.nodeMap, "TriggerSelector", "enum", readable=True, writable=True)
        self._set_node_value(triggerSelector, "enum", "FrameStart")

        triggerActivation = self._get_node(self.nodeMap, "TriggerActivation", "enum", readable=True, writable=True)
        self._set_node_value(triggerActivation, "enum", "RisingEdge")
        
        # This is not from official doc but from claude AI
        # Off
        # Trigger1 -> [Exposure] -> [Readout] -> Trigger2 available
        
        # Readout
        #  Trigger1 -> [Exposure] -> [Readout]
        #                           Trigger2 available ->  [Exposure]
        
        # PreviousFrame : 
        # Trigger1 -> [Exposure] -> [Readout]
        #             Trigger2 available ->  [Exposure] -> [Readout]

        triggerOverlap = self._get_node(self.nodeMap, "TriggerOverlap", "enum", readable=True, writable=True)
        self._set_node_value(triggerOverlap, "enum", "ReadOut")

    def _configurePacketDelay(self) -> None:
        """configure GigE Vision packet delay.
        
        Sets inter-packet delay to prevent network congestion. This is particularly
        useful in multi-camera setups or when experiencing packet loss.
        
        The delay is calculated as: PACKET_SIZE * PACKET_DELAY_MULTIPLIER
        Default: 9000 * 2 = 18000 nanoseconds = 18 microseconds
        
        Args:
            nodeMap: Camera node map
            
        Raises:
            CameraConfigurationError: If configuration fails
            
        Note:
            Only relevant for GigE cameras. This setting is optional and disabled
            by default as throughput limit usually provides sufficient control.
        """
        # GevSCPD (GigE Vision Stream Channel Packet Delay) controls the inter-packet
        # delay in nanoseconds. This delay prevents overwhelming the network switch/NIC
        # when transmitting high-bandwidth image data split across multiple packets.
        #
        # Why is this needed?
        # - GigE cameras split image data into many small packets (typically 9000 bytes each)
        # - Sending packets too rapidly can cause:
        #   * Network switch buffer overflow → packet drops
        #   * Collisions with other cameras in multi-camera setups
        #   * PC network card buffer overflow
        #
        # The delay adds a small pause between packets:
        # Packet1 → [Delay] → Packet2 → [Delay] → Packet3 → ...
        #
        # Calculation: PACKET_SIZE * MULTIPLIER = 9000 * 2 = 18000 ns = 18 µs
        # This means each packet waits 18 microseconds before sending the next one.
        packetDelay = self._get_node(self.nodeMap, "GevSCPD", "int", readable=True, writable=True)
        packetDelay.SetValue(PyspinCameraConfig.PACKET_DELAY_MULTIPLIER * PyspinCameraConfig.PACKET_SIZE)

    def _configureExposure(self) -> None:
        """configure camera exposure settings."""

        # Disable automatic exposure
        exposureAuto = self._get_node(self.nodeMap, "ExposureAuto", "enum", readable=True, writable=True)
        self._set_node_value(exposureAuto, "enum", "Off")

        exposureTime = self._get_node(self.nodeMap, "ExposureTime", "float", readable=True, writable=True)
        self.exposure_time = min(self.exposure_time, exposureTime.GetMax())
        self._set_node_value(exposureTime, "float", self.exposure_time)

    def _configureAcquisition(self) -> None:
        """configure camera acquisition settings based on operation mode."""
        # SingleFrame for image mode, Continuous for video or stream mode
        acq_mode = "SingleFrame" if self.mode == "image" else "Continuous"
        
        # Set acquisition mode
        acquisitionMode = self._get_node(self.nodeMap, "AcquisitionMode", "enum", readable=True, writable=True)
        self._set_node_value(acquisitionMode, "enum", acq_mode)
        return True

    def _configurePacketSize(self) -> None:
        """configure GigE Vision packet size for optimal network performance."""
        ptrPayloadSize = self._get_node(self.nodeMap, "GevSCPSPacketSize", "int", readable=True, writable=True)
        ptrPayloadSize.SetValue(PyspinCameraConfig.PACKET_SIZE)

    def _configureChunk(self) -> None:
        """Enable chunk data to include timestamps in image metadata."""
        # Chunk Data allows you the ability to send additional information with the image data. This can be helpful when debugging issues or looking at what settings have been applied to the acquired image.
        # Use ChunkModeActive to enable chunk data for images.
        # The following information is available as chunk data:
        # Image - enabled by default and cannot be disabled.
        # Image CRC - enabled by default and cannot be disabled.
        # FrameID
        # OffsetX
        # OffsetY
        # Width
        # Height
        # Exposure Time
        # Gain
        # Black Level
        # Pixel Format
        # ImageTimestamp
        
        chunkModeActive = self._get_node(self.nodeMap, "ChunkModeActive", "bool", readable=True, writable=True)
        chunkModeActive.SetValue(True)

        chunkSelector = self._get_node(self.nodeMap, "ChunkSelector", "enum", readable=True, writable=True)
        self._set_node_value(chunkSelector, "enum", "Timestamp")

        chunkEnable = self._get_node(self.nodeMap, "ChunkEnable", "bool", readable=True, writable=True)
        chunkEnable.SetValue(True)
        
    def _configureBuffer(self) -> None:
        """Configure camera buffer settings based on operation mode."""
        
        # Set stream buffer Count Mode to manual
        BuferCountMode = self._get_node(self.stream_nodemap, "StreamBufferCountMode", "enum", readable=True, writable=True)
        self._set_node_value(BuferCountMode, "enum", "Manual")
        
        # Set stream buffer Count
        bufferCount = self._get_node(self.stream_nodemap, "StreamBufferCountManual", "int", readable=True, writable=True)
        if self.mode == "video":
            bufferCount.SetValue(PyspinCameraConfig.VIDEO_BUFFER_COUNT)
        else:
            bufferCount.SetValue(PyspinCameraConfig.IMAGE_BUFFER_COUNT)
    
def autoforce_ip(self):
    """
    
    """
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


class TimecodeReceiver():
    def __init__(
        self
    ):
        self.timestamps = dict([("timestamps", []), ("frameID", []), ("pc_time", [])])
        self.autoforce_ip()
        self.exit = Event()
        self.start_capture = Event()
        self.cam_start = Event()
        self.connect_flag = Event()
        self.save_end = Event()
        
        self.save_path = None
        self.lock = Lock()
        
        self.thread = Thread(target=self.run)
        self.thread.start()
        
        self.wait_for_connection()

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
    
    def wait_for_connection(self):
        self.connect_flag.wait()
        
    def wait_for_cam_start(self):
        self.cam_start.wait()
    
    def wait_for_save(self):
        self.save_end.wait()
        
    def start(self, save_path=None):
        self.save_path = save_path
            
        self.start_capture.set()
        self.wait_for_cam_start()
    
    def end(self):
        self.save_end.clear()
        self.start_capture.clear()
        self.wait_for_save()
        self.save_path = None
        
    def get_data(self):
        with self.lock:
            return self.cur_frame, self.frame_time
    
    def run(self):
        system = ps.System.GetInstance()
        cam_list = system.GetCameras()

        if cam_list.GetSize() < 1:
            print(f"No cameras found. Exiting...")
            cam_list.Clear()
            system.ReleaseInstance()
            return

        self.cam = cam_list.GetByIndex(0)
        self.cam .Init()  # initialize camera
        self.device_nodemap = self.cam.GetTLDeviceNodeMap()  #
        self.stream_nodemap = self.cam.GetTLStreamNodeMap()  #
        self.nodeMap = self.cam.GetNodeMap()  #
        
        self.configureSettings(self.nodeMap)
        self.connect_flag.set()
        
        self.cur_frame = -1
        self.frame_time = -1
        
        while not self.exit.is_set():
            while not self.start_capture.is_set():
                time.sleep(0.01)
                if self.exit.is_set():
                    break
            if self.exit.is_set():
                break
            
            self.cam.BeginAcquisition()
            self.cam_start.set()
            
            while self.start_capture.is_set():
                if self.exit.is_set():
                    break
                
                pImageRaw = self.cam.GetNextImage()  # get from buffer
                framenum = pImageRaw.GetFrameID()
                capture_time = time.time()
                
                if not pImageRaw.IsIncomplete():
                    chunkData = pImageRaw.GetChunkData()
                    ts = chunkData.GetTimestamp()
                    
                    self.timestamps["timestamps"].append(ts)
                    self.timestamps["frameID"].append(framenum)
                    self.timestamps["pc_time"].append(capture_time)
                    
                    with self.lock:
                        self.cur_frame = framenum
                        self.frame_time = capture_time
                        
                else:
                    print(ps.Image_GetImageStatusDescription(pImageRaw.GetImageStatus()))
            
                pImageRaw.Release()

            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)
                json.dump(
                    self.timestamps, open(os.path.join(self.save_path,"camera_timestamp.json"), "w"), indent="\t"
                )
                
                self.timestamps = dict([("timestamps", []), ("frameID", []), ("pc_time", [])])
                
            self.cam.EndAcquisition()
            self.save_end.set()

        self.cam.DeInit()
        del self.cam
        
        cam_list.Clear()
        system.ReleaseInstance()

    def quit(self):
        """Stops the serial reader process."""
        if self.save_path is not None:
            self.end()
            
        self.exit.set()
        self.thread.join()

    def configureThroughPut(self, nodeMap):
        # Set throughput limit 
        ptrVal = ps.CIntegerPtr(nodeMap.GetNode("DeviceLinkThroughputLimit"))
        if not ps.IsReadable(ptrVal) or not ps.IsWritable(ptrVal):
            print("Unable to get or set throughput. Aborting...")
            return
        # Ensure desired exposure time does not exceed the maximum
        ValMax = ptrVal.GetMax()
        ValMin = ptrVal.GetMin()

        posValMax = ((ValMax - ValMin) // 16000) * 16000 + ValMin
        ptrVal.SetValue(posValMax)
        print("Throughput limit set to ", posValMax)
        return
    
    def configureTrigger(self, nodeMap):
        print("trigger mode on")
        ptrTriggerMode = ps.CEnumerationPtr(nodeMap.GetNode("TriggerMode"))
        if not ps.IsReadable(ptrTriggerMode):
            # print("Unable to disable trigger mode (node retrieval). Aborting...")
            return False
        ptrTriggerModeOff = ps.CEnumEntryPtr(ptrTriggerMode.GetEntryByName("Off"))
        if not ps.IsReadable(ptrTriggerModeOff):
            # print("Unable to disable trigger mode (enum entry retrieval). Aborting...")
            return False
        ptrTriggerMode.SetIntValue(ptrTriggerModeOff.GetValue())
        ptrTriggerSelector = ps.CEnumerationPtr(nodeMap.GetNode("TriggerSelector"))
        if not ps.IsReadable(ptrTriggerSelector) or not ps.IsWritable(ptrTriggerSelector):
            # print("Unable to get or set trigger selector (node retrieval). Aborting...")
            return False
        ptrTriggerSelectorFrameStart = ps.CEnumEntryPtr(ptrTriggerSelector.GetEntryByName("FrameStart"))
        if not ps.IsReadable(ptrTriggerSelectorFrameStart):
            # print("Unable to get trigger selector FrameStart (enum entry retrieval). Aborting...")
            return False
        
        # Set Frame start
        ptrTriggerSelector.SetIntValue(ptrTriggerSelectorFrameStart.GetValue())
        ptrTriggerSource = ps.CEnumerationPtr(nodeMap.GetNode("TriggerSource"))
        if not ps.IsReadable(ptrTriggerSource) or not ps.IsWritable(ptrTriggerSource):
            print("Unable to get or set trigger mode (node retrieval). Aborting...")
            return False
        # Set Rising-edge trigger
        ptrTriggerActivation = ps.CEnumerationPtr(nodeMap.GetNode("TriggerActivation"))
        if not ps.IsReadable(ptrTriggerActivation) or not ps.IsWritable(ptrTriggerActivation):
            print("Unable to get or set trigger activation (node retrieval). Aborting...")
            return False
        
        ptrRisingEdge = ps.CEnumEntryPtr(ptrTriggerActivation.GetEntryByName("RisingEdge"))
        if not ps.IsReadable(ptrRisingEdge):
            print("Unable to enable trigger mode RisingEdge (enum entry retrieval). Aborting...")
            return False
        ptrTriggerActivation.SetIntValue(ptrRisingEdge.GetValue())

        # Set trigger mode to hardware ('Line0')
        ptrTriggerSourceHardware = ps.CEnumEntryPtr(ptrTriggerSource.GetEntryByName("Line0"))
        if not ps.IsReadable(ptrTriggerSourceHardware):
            print("Unable to set trigger mode Line0. Aborting...")
            return False
        ptrTriggerSource.SetIntValue(ptrTriggerSourceHardware.GetValue())

        # Turn trigger mode on
        ptrTriggerModeOn = ps.CEnumEntryPtr(ptrTriggerMode.GetEntryByName("On"))
        if not ps.IsReadable(ptrTriggerModeOn):
            print("Unable to enable trigger On (enum entry retrieval). Aborting...")
            return False
        ptrTriggerMode.SetIntValue(ptrTriggerModeOn.GetValue())

        # Set Trigger Overlap mode
        ptrTriggerOverlap = ps.CEnumerationPtr(nodeMap.GetNode("TriggerOverlap"))
        if not ps.IsReadable(ptrTriggerOverlap) or not ps.IsWritable(ptrTriggerOverlap):
            print("Unable to get or set trigger overlap (node retrieval). Aborting...")
            return False        
        ptrReadOut = ps.CEnumEntryPtr(ptrTriggerOverlap.GetEntryByName("ReadOut"))
        if not ps.IsReadable(ptrReadOut):
            print("Unable to enable trigger Overlap readout(enum entry retrieval). Aborting...")
            return False
        ptrTriggerOverlap.SetIntValue(ptrReadOut.GetValue())
        # NOTE: Blackfly and Flea3 GEV cameras need 1 second delay after trigger mode is turned on
        return True
    
    def configurePacketSize(self, nodeMap):
        ptrPayloadSize = ps.CIntegerPtr(nodeMap.GetNode("GevSCPSPacketSize"))
        if not ps.IsAvailable(ptrPayloadSize) or not ps.IsWritable(ptrPayloadSize):
            print("Unable to set packet size. Aborting...")
            return
        ptrPayloadSize.SetValue(9000)
        print("Packet size set to 9000")
        return

    def configureAcquisition(self, nodeMap):
        ptrAcquisitionMode = ps.CEnumerationPtr(nodeMap.GetNode("AcquisitionMode"))
        if not ps.IsReadable(ptrAcquisitionMode) or not ps.IsWritable(ptrAcquisitionMode):
            print("Unable to set acquisition mode to continuous (enum retrieval). Aborting...")
            exit(-1)
        # Retrieve entry node from enumeration node
        ptrAcquisitionModeContinuous = ps.CEnumEntryPtr(ptrAcquisitionMode.GetEntryByName("Continuous"))
        if not ps.IsReadable(ptrAcquisitionModeContinuous): 
            print("Unable to get or set acquisition mode to continuous (entry retrieval). Aborting...")
            exit(-1)
        
        # Retrieve integer value from entry node
        acquisitionModeContinuous = ptrAcquisitionModeContinuous.GetValue()
        # Set integer value from entry node as new value of enumeration node
        ptrAcquisitionMode.SetIntValue(acquisitionModeContinuous)
        return

    def configureChunk(self, nodeMap):
        ptrChunkModeActive = ps.CBooleanPtr(nodeMap.GetNode("ChunkModeActive"))
        if not ps.IsAvailable(ptrChunkModeActive) or not ps.IsWritable(ptrChunkModeActive):
            print("Camera is Unable to activate chunk mode. Aborting...")
            return
        
        ptrChunkModeActive.SetValue(True)
        ptrChunkSelector = ps.CEnumerationPtr(nodeMap.GetNode("ChunkSelector"))
        if not ps.IsAvailable(ptrChunkSelector) or not ps.IsWritable(ptrChunkSelector):
            print("Camera Chunk Selector is not writable")
            return

        # Select Timestamp for Chunk data
        ptrChunkSelectorTimestamp = ps.CEnumEntryPtr(ptrChunkSelector.GetEntryByName("Timestamp"))
        if not ps.IsAvailable(ptrChunkSelectorTimestamp) or not ps.IsReadable(ptrChunkSelectorTimestamp):
            print("Camera is Unable to set Chunk Selector (node retrieval). Aborting...")
            return
        
        ptrChunkSelector.SetIntValue(ptrChunkSelectorTimestamp.GetValue())
        # Else select also others
        ptrChunkEnable = ps.CBooleanPtr(nodeMap.GetNode("ChunkEnable"))
        # Retrieve corresponding boolean

        # Enable the boolean, thus enabling the corresponding chunk data
        if not ps.IsAvailable(ptrChunkEnable): print("Camera chunk not available")
        elif ptrChunkEnable.GetValue(): print("Camera chunk enabled")
        elif ps.IsWritable(ptrChunkEnable):
            ptrChunkEnable.SetValue(True)
            print("Camera chunk enabled")
        else:
            print("Camera chunk not writable")
        return

    def configureBuffer(self, nodeMap):
        handling_mode = ps.CEnumerationPtr(nodeMap.GetNode('StreamBufferHandlingMode'))
        if not ps.IsAvailable(handling_mode) or not ps.IsWritable(handling_mode):
            print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
            return False

        handling_mode_entry = ps.CEnumEntryPtr(handling_mode.GetCurrentEntry())
        if not ps.IsAvailable(handling_mode_entry) or not ps.IsReadable(handling_mode_entry):
            print('Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')
            return False


        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = ps.CEnumerationPtr(nodeMap.GetNode('StreamBufferCountMode'))
        if not ps.IsAvailable(stream_buffer_count_mode) or not ps.IsWritable(stream_buffer_count_mode):
            print('Unable to set Buffer Count Mode (node retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode_manual = ps.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
        if not ps.IsAvailable(stream_buffer_count_mode_manual) or not ps.IsReadable(stream_buffer_count_mode_manual):
            print('Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())
        print('Stream Buffer Count Mode set to manual...')

        # Retrieve and modify Stream Buffer Count
        buffer_count = ps.CIntegerPtr(nodeMap.GetNode('StreamBufferCountManual'))
        if not ps.IsAvailable(buffer_count) or not ps.IsWritable(buffer_count):
            print('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')
            return False

        buffer_count.SetValue(10)
        print('Stream Buffer Count set to 10...')
        return 
    
    def configureSettings(self, nodeMap):
        self.configureThroughPut(nodeMap)
        self.configureTrigger(nodeMap)
        
        self.configureAcquisition(nodeMap)
        self.configureChunk(nodeMap)

        self.configureBuffer(self.stream_nodemap)
        self.configurePacketSize(self.nodeMap)