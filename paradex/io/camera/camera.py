"""
PySpin Camera Control Module
============================

This module provides a high-level interface for controlling a single FLIR camera.

Key Features:
    - **Configuration**: Configure camera parameters including gain, exposure time, 
      frame rate, trigger mode, buffer settings, and network parameters
    - **Image Acquisition**: Capture single frames or continuous video streams

This module wraps the PySpin SDK to abstract the complex nodemap structure of 
FLIR cameras into a simple Python interface.

Classes:
    CameraMode: Enumeration of camera operation modes
    CameraConfig: Configuration constants for camera settings
    Camera: Main camera control class for a single camera instance
    CameraConfigurationError: Exception for configuration failures
    
Typical Workflow:
    1. Initialize camera with desired parameters
    2. Configure settings (done automatically during initialization)
    3. Start acquisition
    4. Get images
    5. Stop acquisition
    6. Release camera resources

Example:
    >>> import PySpin as ps
    >>> from camera import Camera
    >>> 
    >>> system = ps.System.GetInstance()
    >>> cam_list = system.GetCameras()
    >>> cam_ptr = cam_list[0]
    >>> camera = Camera(
    ...     camPtr=cam_ptr,
    ...     gain=10.0,
    ...     exposure_time=10000.0,
    ...     frame_rate=30.0,
    ...     mode='image',
    ...     syncMode=True
    ... )
    >>> 
    >>> camera.start()
    >>> image = camera.get_image()
    >>> camera.stop()
    >>> camera.release()
"""


import PySpin as ps
from typing import Literal, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CameraMode(Enum):
    """Enumeration of camera operation modes.
    
    Attributes:
        IMAGE: Single frame capture mode with retry logic
        VIDEO: Continuous video capture with 10-buffer queue
        STREAM: Streaming mode (similar to video)
    """
    IMAGE = "image"
    VIDEO = "video"
    STREAM = "stream"

class CameraConfig:
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

class CameraConfigurationError(Exception):
    """Exception raised when camera configuration fails.
    
    This exception is raised during camera initialization or mode changes
    when configuration of camera parameters fails.
    """
    pass

class Camera():
    def __init__(
        self,
        camPtr,
        gain,
        exposure_time,
        frame_rate=0.2,
        mode="video",
        syncMode=True,
    ):
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
        
        self.configure()
                    
    def _serialnum(self):
        device_nodemap = self.cam.GetTLDeviceNodeMap()
        serialnum_entry = device_nodemap.GetNode(
            "DeviceSerialNumber"
        )  
        serialnum = ps.CStringPtr(serialnum_entry).GetValue()
        return serialnum
        
    def set_mode(self, mode, syncMode):
        assert mode == "image" or mode == "video" or mode == "stream", "mode must be image or video"
        
        self.mode = mode
        self.syncMode = syncMode

        # Reconfigure camera for new mode
        self.configureBuffer()
        self.configureAcquisition()
        if not self.syncMode:
            self.configureFrameRate()
        else:
            self.configureTrigger()
        
    def get_image(self):
        if self.mode == "image":
            while True:
                try:
                    pImageRaw = self.cam.GetNextImage(100)
                    return pImageRaw
                except:
                    self.stop()
                    self.start()
        else:
            pImageRaw = self.cam.GetNextImage()
            return pImageRaw

    def start(self):
        self.cam.BeginAcquisition()
        return
    
    def stop(self):
        self.cam.EndAcquisition()
        return
    
    def release(self):
        self.cam.DeInit()
        del self.cam
        return

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
    def _set_node_value(self, node, node_type, value)-> None:
        """Set the value of a camera node."""
        if node_type == "enum":
            enum_entry = ps.CEnumEntryPtr(node.GetEntryByName(value))
            if not ps.IsReadable(enum_entry):
                raise CameraConfigurationError(f"Unable to get {value} (enum entry retrieval). Aborting...")
            node.SetIntValue(enum_entry.GetValue())
        
        if node_type == "float":
            node.SetValue(value)

    def configure(self) -> None:
        """Configure camera settings based on initialization parameters."""
        self.configureGain()
        self.configureThroughPut()

        if not self.syncMode:
            self.configureFrameRate()  # we use trigger anyway
        else:
            self.configureTrigger()
        
        self.configurePacketSize(self.nodeMap)
        self.configurePacketDelay(self.nodeMap)
        self.configureExposure(self.nodeMap)
        self.configureAcquisition(self.nodeMap)
        self.configureChunk(self.nodeMap)
        self.configureBuffer(self.stream_nodemap)

    def configureGain(self) -> None:
        """Configure camera gain settings."""

        # Disable automatic gain
        ptrGainAuto = self._get_node(self.nodeMap, "GainAuto", "enum", readable=True, writable=True)
        self._set_node_value(ptrGainAuto, "enum", "Off")
        
        Gainval = self._get_node(self.nodeMap, "Gain", "float", readable=True, writable=True)

        # Ensure desired exposure time does not exceed the maximum
        self.gain = min(self.gain, Gainval.GetMax())
        
        # Set gain value
        self._set_node_value(Gainval, "float", self.gain)
        
    def configureThroughPut(self) -> None:
        """Configure camera throughput settings."""
        
        ThroughputLimit = self._get_node(self.nodeMap, "DeviceLinkThroughputLimit", "int", readable=True, writable=True)

        # Align to nearest multiple of THROUGHPUT_ALIGNMENT (value must be multiple of 16000)
        ValMax = ThroughputLimit.GetMax()
        ValMin = ThroughputLimit.GetMin()
        posValMax = ((ValMax - ValMin) // CameraConfig.THROUGHPUT_ALIGNMENT) * CameraConfig.THROUGHPUT_ALIGNMENT + ValMin 
        
        # Set throughput limit 
        ThroughputLimit.SetValue(posValMax)
        
    def configureFrameRate(self) -> None:
        """Configure camera frame rate settings."""

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

    def configureTrigger(self) -> None:
        """Configure camera trigger settings for hardware synchronization."""
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

    def configurePacketDelay(self) -> None:
        """Configure GigE Vision packet delay.
        
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
        packetDelay.SetValue(CameraConfig.PACKET_DELAY_MULTIPLIER * CameraConfig.PACKET_SIZE)

    def configureExposure(self) -> None:
        """Configure camera exposure settings."""

        # Disable automatic exposure
        exposureAuto = self._get_node(self.nodeMap, "ExposureAuto", "enum", readable=True, writable=True)
        self._set_node_value(exposureAuto, "enum", "Off")

        exposureTime = self._get_node(self.nodeMap, "ExposureTime", "float", readable=True, writable=True)
        self.exposure_time = min(self.exposure_time, exposureTime.GetMax())
        self._set_node_value(exposureTime, "float", self.exposure_time)

    def configureAcquisition(self) -> None:
        """Configure camera acquisition settings based on operation mode."""
        # SingleFrame for image mode, Continuous for video or stream mode
        acq_mode = "SingleFrame" if self.mode == "image" else "Continuous"
        
        # Set acquisition mode
        acquisitionMode = self._get_node(self.nodeMap, "AcquisitionMode", "enum", readable=True, writable=True)
        self._set_node_value(acquisitionMode, "enum", acq_mode)
        return True

    def configurePacketSize(self) -> None:
        """Configure GigE Vision packet size for optimal network performance."""
        ptrPayloadSize = self._get_node(self.nodeMap, "GevSCPSPacketSize", "int", readable=True, writable=True)
        ptrPayloadSize.SetValue(CameraConfig.PACKET_SIZE)

    def configureChunk(self) -> None:
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
        
    def configureBuffer(self) -> None:
        """Configure camera buffer settings based on operation mode."""
        
        # Set stream buffer Count Mode to manual
        BuferCountMode = self._get_node(self.stream_nodemap, "StreamBufferCountMode", "enum", readable=True, writable=True)
        self._set_node_value(BuferCountMode, "enum", "Manual")
        
        # Set stream buffer Count
        bufferCount = self._get_node(self.stream_nodemap, "StreamBufferCountManual", "int", readable=True, writable=True)
        if self.mode == "video":
            bufferCount.SetValue(CameraConfig.VIDEO_BUFFER_COUNT)
        else:
            bufferCount.SetValue(CameraConfig.IMAGE_BUFFER_COUNT)