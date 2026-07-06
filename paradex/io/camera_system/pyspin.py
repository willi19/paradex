import PySpin as ps
from typing import Literal, Optional
import numpy as np
import cv2
import time
import random
import traceback

from paradex.utils.system import get_camera_config
from paradex.utils.log import get_logger

logger = get_logger("camera")

cam_info = get_camera_config()

system = ps.System.GetInstance()

# Max time to wait for a frame in get_image(). On timeout GetNextImage raises
# SpinnakerException and we return (None, None) so the acquisition loop can
# re-check its start/exit events instead of blocking forever (LAN drop, trigger
# stop, etc.). TODO: move to system config (see design/camera-recording-redesign.md P3).
GRAB_TIMEOUT_MS = 1000

def get_serial_list():
    """Enumerate the serial numbers of all cameras visible to the system.

    Returns
    -------
    list of str
        Serial number of every camera currently detected by the Spinnaker
        system instance.
    """
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

def _cfg_for_serial(serialnum, cfg=None):
    """Return the effective per-camera config for ``serialnum``.

    ``CameraLoader`` passes the already-read camera.json entry so reloads and
    tests use one source of truth. Direct callers keep backwards compatibility by
    falling back to this module's camera.json snapshot.
    """
    if cfg is not None:
        return cfg
    return cam_info.get(serialnum, {})


def load_camera(serialnum, cfg=None):
    """Open a camera by serial number and wrap it in a :class:`PyspinCamera`.

    Gain/exposure and optional hardware tuning keys are taken from ``cfg`` when
    provided; direct callers fall back to this module's camera.json snapshot.
    Missing gain/exposure values use fixed defaults (gain 3.0, exposure 2500.0).

    Parameters
    ----------
    serialnum : str
        Serial number of the camera to open.
    cfg : dict, optional
        Per-serial ``camera.json`` entry supplied by ``CameraLoader``.

    Returns
    -------
    PyspinCamera
        Initialized and configured camera wrapper.

    Raises
    ------
    ValueError
        If no camera with the given serial number is found.
    """
    cam_list = system.GetCameras()

    try:
        camPtr = cam_list.GetBySerial(serialnum)
    except:
        raise ValueError(f"Camera with serial number {serialnum} not found.")
    
    cfg = _cfg_for_serial(serialnum, cfg)
    gain = cfg.get("gain", 3.0)
    exposure = cfg.get("exposure", 2500.0)
    if cfg:
        logger.info(f"serial_num Gain :{gain}, exposure {exposure}")
    
    cam = PyspinCamera(camPtr, gain, exposure, cfg=cfg)

    cam_list.Clear()
        
    return cam

def load_timestamp_monitor(serialnum, cfg=None):
    """Open a camera by serial number as a :class:`PyspinTimestampMonitor`.

    Like :func:`load_camera` but returns a lightweight monitor that only reads
    frame timestamps/IDs (used for sync diagnostics). Gain/exposure and hardware
    tuning keys come from ``cfg`` when available, otherwise defaults are used.

    Parameters
    ----------
    serialnum : str
        Serial number of the camera to open.
    cfg : dict, optional
        Per-serial ``camera.json`` entry supplied by the caller.

    Returns
    -------
    PyspinTimestampMonitor
        Initialized timestamp monitor for the camera.

    Raises
    ------
    ValueError
        If no camera with the given serial number is found.
    """
    cam_list = system.GetCameras()

    try:
        camPtr = cam_list.GetBySerial(serialnum)
    except:
        raise ValueError(f"Camera with serial number {serialnum} not found.")
    
    cfg = _cfg_for_serial(serialnum, cfg)
    gain = cfg.get("gain", 3.0)
    exposure = cfg.get("exposure", 2500.0)  # camera.json key is "exposure"

    cam = PyspinTimestampMonitor(camPtr, gain, exposure, cfg=cfg)

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
    """Thin wrapper around a single FLIR (Spinnaker) camera.

    Owns an initialized PySpin camera pointer and exposes a small, hardware-
    agnostic API (:meth:`start`, :meth:`get_image`, :meth:`stop`,
    :meth:`release`) used by the higher-level :class:`~paradex.io.camera_system.camera.Camera`
    capture thread. Construction initializes the camera and applies the baseline
    configuration (gain, exposure, packet size, chunk data, buffers, pixel
    format) via :meth:`_init_configure`.

    Parameters
    ----------
    camPtr : PySpin.CameraPtr
        Pointer to the Spinnaker camera to wrap (not yet ``Init``-ed).
    gain : float
        Camera gain in dB.
    exposure_time : float
        Exposure time in microseconds.
    """

    def __init__(
        self,
        camPtr,
        gain,
        exposure_time,
        cfg=None
    ):
        """Initialize and configure a FLIR camera.

        :param camPtr: PySpin camera pointer
        :param gain: Camera gain (dB)
        :param exposure_time: Exposure time (microseconds)
        :param cfg: Optional per-camera config dict (camera.json entry). Keys like
            ``packet_size``, ``buffer_count``, ``buffer_mode``, ``pixel_format``
            override the :class:`PyspinCameraConfig` defaults for this camera; any
            key that is absent falls back to the default.
        """
        self.mode = None
        self.syncMode = None
        self.frame_rate = None

        self.cfg = cfg or {}
        self.gain = gain
        self.exposure_time = exposure_time
        
        camPtr.Init()  # initialize camera
        self.cam = camPtr
        self.serial_num = self._serialnum()

        self.stream_nodemap = camPtr.GetTLStreamNodeMap() 
        self.nodeMap = camPtr.GetNodeMap() 
        self._init_configure()

        self.init_time = time.time()
    
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
        """Grab the next frame from the camera.

        Waits up to ``GRAB_TIMEOUT_MS`` for a frame. On timeout (no frame
        arrived — LAN drop, trigger stopped, etc.) ``GetNextImage`` raises a
        ``SpinnakerException`` which is caught and reported as ``(None, None)``
        so the acquisition loop can re-check its start/exit events instead of
        hanging forever. An incomplete or zero-size frame yields
        ``(None, frame_data)``.

        Returns
        -------
        frame : numpy.ndarray or None
            Debayered BGR image of shape ``(H, W, 3)``, or ``None`` on grab
            timeout or an incomplete/zero-size frame.
        frame_data : dict or None
            ``{"pc_time": float, "frameID": int}`` metadata for the frame, or
            ``None`` on grab timeout.

        Raises
        ------
        PySpin.SpinnakerException
            On any non-timeout grab error (e.g. the camera disconnected). It is
            re-raised so :meth:`Camera.continuous_acquire` records it in the
            camera's error state and the remote controller can see it via the
            heartbeat / ``get_status()``.
        """
        try:
            pImageRaw = self.cam.GetNextImage(getattr(self, 'grab_timeout_ms', GRAB_TIMEOUT_MS))
        except ps.SpinnakerException as e:
            # A grab timeout just means no frame arrived (dead trigger / LAN drop):
            # return None so the loop retries. Any OTHER Spinnaker error is real —
            # re-raise so continuous_acquire records it and the remote can see it.
            # Detect timeout by error code (fallback: message) to avoid turning a
            # dead trigger into a spurious error.
            is_timeout = (getattr(e, 'errorcode', None) == getattr(ps, 'SPINNAKER_ERR_TIMEOUT', -1011)
                          or 'timeout' in str(e).lower() or 'timed out' in str(e).lower())
            if is_timeout:
                return None, None
            raise

        frame_data = {"pc_time":time.time(), "frameID": pImageRaw.GetFrameID()}
        
        if frame_data['frameID'] % 1 == 0:
            print(f"Frame ID: {frame_data['frameID']}", self.serial_num, time.time() - self.init_time)
        if pImageRaw.IsIncomplete() or pImageRaw.GetWidth() == 0 or pImageRaw.GetHeight() == 0:
            if pImageRaw.IsIncomplete():
                logger.info(f"Image incomplete with image status {pImageRaw.GetImageStatus()}")
            else:
                logger.info("Image has zero width or height")
            pImageRaw.Release()
            return None, frame_data
        
        frame = self._spin2cv(pImageRaw, pImageRaw.GetHeight(), pImageRaw.GetWidth())
        # print(f"Frame ID: {frame_data['frameID']}", self.serial_num, time.time() - self.init_time)
        # image_copy = pImageRaw.GetNDArray().copy()
        # frame = cv2.cvtColor(image_copy, cv2.COLOR_BayerRG2RGB)

        pImageRaw.Release()
        return frame, frame_data
    
    def start(self, mode, syncMode, frame_rate=None, gain=None, exposure_time=None):
        """Configure the camera for the requested mode and begin acquisition.

        Reads the current camera state, applies only the settings that changed
        (throughput, trigger vs. free-run frame rate, gain, exposure,
        acquisition mode), then calls ``BeginAcquisition``.

        Parameters
        ----------
        mode : {'single', 'continuous'}
            ``'single'`` for one-shot image capture, ``'continuous'`` for
            video/stream capture.
        syncMode : bool
            Use the hardware trigger if ``True``; free-run otherwise.
        frame_rate : float, optional
            Free-run frame rate in Hz; only applied when ``syncMode`` is
            ``False``.
        gain : float, optional
            Gain override in dB; ``None`` keeps the current value.
        exposure_time : float, optional
            Exposure override in microseconds; ``None`` keeps the current value.

        Raises
        ------
        AssertionError
            If ``mode`` is not ``'single'`` or ``'continuous'``.
        """
        assert mode in ["single", "continuous"]
        
        self._read_current_state()
        self._configureThroughPut()
        
        if syncMode:
            logger.info("Configuring camera for hardware sync mode.")
            self.syncMode = syncMode
            self._configureTrigger()
            
        if ((not syncMode and syncMode != self.syncMode) or (frame_rate is not None and frame_rate != self.frame_rate)) and (not syncMode):
            logger.info("Configuring camera for free-run mode.")
            self.frame_rate = frame_rate
            self._configureFrameRate()
        
        if gain is not None and gain != self.gain:
            self.gain = gain
            self._configureGain()
        
        if exposure_time is not None and exposure_time != self.exposure_time:
            self.exposure_time = exposure_time
            self._configureExposure()
        
        if mode != self.mode:
            self.mode = mode
            self._configureAcquisition()

        # Grab timeout: at least the base, but >= 3 frame periods for low fps so a
        # slow-but-healthy stream is not mistaken for a dead trigger.
        if frame_rate and frame_rate > 0:
            self.grab_timeout_ms = max(GRAB_TIMEOUT_MS, int(3 * 1000 / frame_rate))
        else:
            self.grab_timeout_ms = GRAB_TIMEOUT_MS

        self.cam.BeginAcquisition()
        return
    
    def set_gain(self, gain):
        """Apply a new gain (dB) live — Gain is writable during acquisition."""
        self.gain = float(gain)
        self._configureGain()

    def set_exposure(self, exposure_time):
        """Apply a new exposure (microseconds) live — ExposureTime is writable
        during acquisition."""
        self.exposure_time = float(exposure_time)
        self._configureExposure()

    def stop(self):
        """Stop acquisition and drain the buffer; keep the connection open.

        Aborts streaming (to avoid ``EndAcquisition`` hanging on a lost
        trigger), ends acquisition, then flushes any images left in the buffer.
        Only stops acquiring — use :meth:`release` to disconnect the camera.
        """
        self._abort_streaming()
        try:
            self.cam.EndAcquisition()
        except ps.SpinnakerException as e:
            logger.warning(f"{self.serial_num} EndAcquisition failed: {e}")
        # Flush buffer command (있으면)
        try:
            while True:
                image = self.cam.GetNextImage(1)  # timeout=0 (즉시 반환)
                image.Release()
        except ps.SpinnakerException:
            pass  # 버퍼가 비면 예외 발생

        return

    def _abort_streaming(self):
        """trigger 끊김 등으로 EndAcquisition 이 packet 대기로 hang 되는 것 방지.
        StreamAbortStreaming 으로 stream socket 강제 해제."""
        try:
            tlstream_nodemap = self.cam.GetTLStreamNodeMap()
            abort_node = ps.CCommandPtr(tlstream_nodemap.GetNode('StreamAbortStreaming'))
            if ps.IsAvailable(abort_node) and ps.IsWritable(abort_node):
                abort_node.Execute()
        except ps.SpinnakerException:
            pass
    
    def release(self):
        """Deinitialize the camera and drop the pointer.

        Disconnects the hardware camera (``DeInit``) and deletes the internal
        reference. Call :meth:`stop` first; the wrapper cannot be reused
        afterward.
        """
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
    def _set_node_value(node, node_type, value):
        if node_type == "enum":
            entry = ps.CEnumEntryPtr(node.GetEntryByName(value))
            if not ps.IsReadable(entry):
                raise PyspinCameraConfigurationError(f"Enum entry {value} not readable")
            node.SetIntValue(entry.GetValue())
        elif node_type == "float":
            node.SetValue(float(value))
        elif node_type == "int":
            node.SetValue(int(value))
        elif node_type == "bool":
            node.SetValue(bool(value))
        elif node_type == "command":
            node.Execute()
        else:
            raise ValueError(f"Unsupported node type: {node_type}")

    def _init_configure(self) -> None:
        """configure camera settings based on initialization parameters."""
        self._configureGain()

        self._configurePacketSize()
        # self._configurePacketDelay()
        self._configureExposure()
        self._configureChunk()
        self._configureBuffer()
        self._configurePixelFormat()

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
        posValMax = (int((ValMax - ValMin)) // PyspinCameraConfig.THROUGHPUT_ALIGNMENT) * PyspinCameraConfig.THROUGHPUT_ALIGNMENT + ValMin
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
            self._set_node_value(autoframerate, "bool", True)

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

        # Disable per-camera frame-rate cap so a stale AcquisitionFrameRate
        # value (e.g. 10 left by a previous free-run session) doesn't throttle
        # hardware-triggered acquisition.
        try:
            rate_auto = self._get_node(self.nodeMap, "AcquisitionFrameRateAuto", "enum", readable=True, writable=True)
            self._set_node_value(rate_auto, "enum", "Off")
        except Exception:
            pass
        try:
            rate_enable = self._get_node(self.nodeMap, "AcquisitionFrameRateEnable", "bool", readable=True, writable=True)
            self._set_node_value(rate_enable, "bool", False)
        except Exception:
            pass

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
        packetDelay.SetValue(PyspinCameraConfig.PACKET_DELAY_MULTIPLIER * PyspinCameraConfig.PACKET_SIZE + random.randrange(-500, 500))

        streamPacketResend = self._get_node(self.stream_nodemap, "StreamPacketResendEnable", "bool", True, True)
        self._set_node_value(streamPacketResend, "bool", True)

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
        ptrPayloadSize.SetValue(self.cfg.get("packet_size", PyspinCameraConfig.PACKET_SIZE))

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
        buf_mode =  self._get_node(self.stream_nodemap, "StreamBufferHandlingMode", "enum", readable=True, writable=True)
        self._set_node_value(buf_mode, "enum", self.cfg.get("buffer_mode", "OldestFirst"))

        # Set stream buffer Count Mode to manual
        BufferCountMode = self._get_node(self.stream_nodemap, "StreamBufferCountMode", "enum", readable=True, writable=True)
        self._set_node_value(BufferCountMode, "enum", "Manual")
        
        # Set stream buffer Count
        bufferCount = self._get_node(self.stream_nodemap, "StreamBufferCountManual", "int", readable=True, writable=True)
        bufferCount.SetValue(self.cfg.get("buffer_count", PyspinCameraConfig.BUFFER_COUNT))

    def _configurePixelFormat(self) -> None:
        """Configure pixel format to BayerRG8."""
        try:
            pixelFormat = self._get_node(self.nodeMap, "PixelFormat", "enum", readable=True, writable=True)
            self._set_node_value(pixelFormat, "enum", self.cfg.get("pixel_format", "BayerRG8"))
            
        except Exception as e:
            pass

class PyspinTimestampMonitor():
    def __init__(
        self,
        camPtr,
        gain,
        exposure_time,
        cfg=None
    ):
        """Initialize and configure a FLIR camera.

        :param camPtr: PySpin camera pointer
        :param gain: Camera gain (dB)
        :param exposure_time: Exposure time (microseconds)
        :param cfg: Optional per-camera config dict (camera.json entry); same
            override keys as :class:`PyspinCamera`.
        """
        self.mode = None
        self.syncMode = None
        self.frame_rate = None

        self.cfg = cfg or {}
        self.gain = gain
        self.exposure_time = exposure_time
        
        camPtr.Init()  # initialize camera
        self.cam = camPtr
        self.serial_num = self._serialnum()

        self.stream_nodemap = camPtr.GetTLStreamNodeMap() 
        self.nodeMap = camPtr.GetNodeMap() 
        self._init_configure()

        self.init_time = time.time()
    
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
        
    def get_timestamp(self, timeout_ms=0):
        """Get next image from camera."""
        if timeout_ms > 0:
            try:
                pImageRaw = self.cam.GetNextImage(timeout_ms)
                frame_data = {"pc_time":time.time(), "frameID": pImageRaw.GetFrameID()}
                pImageRaw.Release()
                return frame_data

            except Exception as e:
                return None
        else:
            pImageRaw = self.cam.GetNextImage()
        
        frame_data = {"pc_time":time.time(), "frameID": pImageRaw.GetFrameID()}
        
        pImageRaw.Release()
        return frame_data
    
    def start(self):
        """
        Start image acquisition.
        """
        
        self._configureThroughPut()
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
    def _set_node_value(node, node_type, value):
        if node_type == "enum":
            entry = ps.CEnumEntryPtr(node.GetEntryByName(value))
            if not ps.IsReadable(entry):
                raise PyspinCameraConfigurationError(f"Enum entry {value} not readable")
            node.SetIntValue(entry.GetValue())
        elif node_type == "float":
            node.SetValue(float(value))
        elif node_type == "int":
            node.SetValue(int(value))
        elif node_type == "bool":
            node.SetValue(bool(value))
        elif node_type == "command":
            node.Execute()
        else:
            raise ValueError(f"Unsupported node type: {node_type}")

    def _init_configure(self) -> None:
        """configure camera settings based on initialization parameters."""
        self._read_current_state()
        self._configureExposure()
        
        if not self.syncMode:
            self.syncMode = True
            self._configureTrigger()
        
        if self.mode != "continuous":
            self.mode = "continuous"
            self._configureAcquisition()
        
        self._configureGain()

        self._configurePacketSize()
        # self._configurePacketDelay()
        self._configureExposure()
        self._configureChunk()
        self._configureBuffer()
        self._configurePixelFormat()

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
        posValMax = (int((ValMax - ValMin)) // PyspinCameraConfig.THROUGHPUT_ALIGNMENT) * PyspinCameraConfig.THROUGHPUT_ALIGNMENT + ValMin
        # Set throughput limit 
        ThroughputLimit.SetValue(posValMax)
    
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

        # Disable per-camera frame-rate cap so a stale AcquisitionFrameRate
        # value (e.g. 10 left by a previous free-run session) doesn't throttle
        # hardware-triggered acquisition.
        try:
            rate_auto = self._get_node(self.nodeMap, "AcquisitionFrameRateAuto", "enum", readable=True, writable=True)
            self._set_node_value(rate_auto, "enum", "Off")
        except Exception:
            pass
        try:
            rate_enable = self._get_node(self.nodeMap, "AcquisitionFrameRateEnable", "bool", readable=True, writable=True)
            self._set_node_value(rate_enable, "bool", False)
        except Exception:
            pass

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
        packetDelay.SetValue(PyspinCameraConfig.PACKET_DELAY_MULTIPLIER * PyspinCameraConfig.PACKET_SIZE + random.randrange(-500, 500))

        streamPacketResend = self._get_node(self.stream_nodemap, "StreamPacketResendEnable", "bool", True, True)
        self._set_node_value(streamPacketResend, "bool", True)

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
        ptrPayloadSize.SetValue(self.cfg.get("packet_size", PyspinCameraConfig.PACKET_SIZE))

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
        buf_mode =  self._get_node(self.stream_nodemap, "StreamBufferHandlingMode", "enum", readable=True, writable=True)
        self._set_node_value(buf_mode, "enum", "NewestOnly")

        # Set stream buffer Count Mode to manual
        BufferCountMode = self._get_node(self.stream_nodemap, "StreamBufferCountMode", "enum", readable=True, writable=True)
        self._set_node_value(BufferCountMode, "enum", "Manual")
        
        # Set stream buffer Count
        bufferCount = self._get_node(self.stream_nodemap, "StreamBufferCountManual", "int", readable=True, writable=True)
        bufferCount.SetValue(self.cfg.get("buffer_count", PyspinCameraConfig.BUFFER_COUNT))

    def _configurePixelFormat(self) -> None:
        """Configure pixel format to BayerRG8."""
        try:
            pixelFormat = self._get_node(self.nodeMap, "PixelFormat", "enum", readable=True, writable=True)
            self._set_node_value(pixelFormat, "enum", self.cfg.get("pixel_format", "BayerRG8"))
            
        except Exception as e:
            pass


def autoforce_ip():
    """Force GigE cameras onto the rig subnet if their IP is off it.

    Iterates over every Spinnaker interface that has exactly one camera
    attached, reads the camera's current ``GevDeviceIPAddress``, and if the
    dotted-quad does not start with ``"11"`` (the rig subnet), executes
    ``GevDeviceAutoForceIP`` to reassign the camera a valid address on that
    subnet. Interfaces with zero or multiple cameras are skipped. Releases the
    Spinnaker system instance before returning.
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
