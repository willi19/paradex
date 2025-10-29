import PySpin as ps
from typing import Literal, Optional
import numpy as np
import cv2
import json
import os
import time

from paradex.utils.file_io import config_dir

if os.path.exists(os.path.join(config_dir, "camera/lens_info.json")):
    lens_info = json.load(open(os.path.join(config_dir, "camera/lens_info.json"), "r"))
    cam_info = json.load(open(os.path.join(config_dir,"camera/camera.json"), "r"))

else:
    lens_info = {}
    cam_info = {}
    
system = ps.System.GetInstance()

def get_serial_list():
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
    return serial_list 

def load_camera(serialnum):
    cam_list = system.GetCameras()

    try:
        camPtr = cam_list.GetBySerial(serialnum)
    except:
        raise ValueError(f"Camera with serial number {serialnum} not found.")
    
    if serialnum in cam_info and cam_info[serialnum]["lens"] in lens_info:
        lens_params = lens_info[cam_info[serialnum]["lens"]]
        gain = lens_params["gain"]
        exposure = lens_params["exposure_time"]
    
    else:
        gain = 3.0
        exposure = 2500.0
    
    cam = PyspinCamera(camPtr, gain, exposure)
    
    cam_list.Clear()
        
    return cam

class PyspinCameraConfig:
    """Camera configuration constants"""
    # Buffer settings
    BUFFER_COUNT = 10
    
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
        exposure_time
    ):
        """Initialize and configure a FLIR camera.
        
        :param camPtr: PySpin camera pointer
        :param gain: Camera gain (dB)
        :param exposure_time: Exposure time (microseconds)
        :param frame_rate: Frame rate in Hz (only when syncMode=False)
        :param mode: 'image', 'video', or 'stream'
        :param syncMode: Use hardware trigger if True
        """
        self.mode = None
        self.syncMode = None
        self.frame_rate = None
        
        self.gain = gain
        self.exposure_time = exposure_time
        
        camPtr.Init()  # initialize camera
        self.cam = camPtr
        self.serial_num = self._serialnum()

        self.stream_nodemap = camPtr.GetTLStreamNodeMap() 
        self.nodeMap = camPtr.GetNodeMap() 
        self._init_configure()
        self._read_current_state()
    
    def _read_current_state(self)-> None:
        """Read current camera state for parameters."""
        # Read current mode
        acqModeNode = self._get_node(self.nodeMap, "AcquisitionMode", "enum", readable=True, writable=False)
        acqModeValue = acqModeNode.GetCurrentEntry()
        acqModeStr = acqModeValue.GetSymbolic()
        self.mode = "single" if acqModeStr == "SingleFrame" else "continuous"
        
        # Read current sync mode
        triggerModeNode = self._get_node(self.nodeMap, "TriggerMode", "enum", readable=True, writable=False)
        triggerModeValue = triggerModeNode.GetCurrentEntry()
        triggerModeStr = triggerModeValue.GetSymbolic()
        self.syncMode = True if triggerModeStr == "On" else False
        
        # Read current frame rate if not in sync mode
        if not self.syncMode:
            framerateNode = self._get_node(self.nodeMap, "AcquisitionFrameRate", "float", readable=True, writable=False)
            self.frame_rate = framerateNode.GetValue()
        
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
        
    def get_image(self):
        """Get next image from camera.
    
        Image mode: Retries with 100ms timeout, auto-restarts on failure.
        Video/Stream mode: Blocks until image available.
        
        :return: Raw image pointer (call Release() after use)
        :rtype: PySpin.ImagePtr
        """
        # if self.mode == "single":
        #     while True:
        #         try:
        #             pImageRaw = self.cam.GetNextImage()
        #             return self._spin2cv(pImageRaw, pImageRaw.GetHeight(), pImageRaw.GetWidth())
                    
        #         except:
        #             self.stop()
        #             self.start()
        # else:
        pImageRaw = self.cam.GetNextImage()
        if pImageRaw.IsIncomplete():
            return None, None
        
        frame_data = {"pc_time":time.time(), "frameID": pImageRaw.GetFrameID()}
        return self._spin2cv(pImageRaw, pImageRaw.GetHeight(), pImageRaw.GetWidth()), frame_data

    def start(self, mode, syncMode, frame_rate=None, gain=None, exposure_time=None):
        """
        Start image acquisition.
        """
        assert mode in ["single", "continuous"]
        
        if mode != self.mode:
            self.mode = mode
            self._configureAcquisition()
        
        if syncMode != self.syncMode and syncMode:
            self._configureTrigger()
            
        if ((not syncMode and syncMode != self.syncMode) or (frame_rate is not None and frame_rate != self.frame_rate)):
            self.frame_rate = frame_rate
            self._configureFrameRate()
        
        if gain is not None and gain != self.gain:
            self.gain = gain
            self._configureGain()
        
        if exposure_time is not None and exposure_time != self.exposure_time:
            self.exposure_time = exposure_time
            self._configureExposure()
            
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
        cvImg = np.array(image_data, dtype=np.uint8).reshape((h, w)).copy()
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BayerRG2RGB)
        return cvImg

    @staticmethod
    def _get_node(nodemap, name, node_type, readable=True, writable=True):
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
            raise PyspinCameraConfigurationError(f"Unable to get or set {name} (node retrieval). Aborting...")
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

    def _init_configure(self) -> None:
        """configure camera settings based on initialization parameters."""
        self._configureGain()
        self._configureThroughPut()

        self._configurePacketSize()
        # self._configurePacketDelay()
        self._configureExposure()
        self._configureChunk()
        self._configureBuffer()

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
        if self.frame_rate is not None:
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
        acq_mode = "SingleFrame" if self.mode == "single" else "Continuous"
        
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
        bufferCount.SetValue(PyspinCameraConfig.BUFFER_COUNT)
    
def autoforce_ip():
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