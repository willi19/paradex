Camera Module
=============

.. automodule:: camera
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Camera module provides a high-level interface for controlling a single FLIR camera 
using the PySpin SDK. It handles both **configuration** and **image acquisition** for 
one camera instance.

Key Features
~~~~~~~~~~~~

* **Configuration**: Gain, exposure time, frame rate, trigger mode, buffer settings
* **Image Acquisition**: Single frame, continuous video, or streaming modes
* **Synchronization**: Hardware trigger or free-running modes

Quick Example
-------------

::

    import PySpin as ps
    from camera import Camera
    
    # Get camera
    system = ps.System.GetInstance()
    cam_list = system.GetCameras()
    cam_ptr = cam_list[0]
    
    # Initialize and configure camera
    camera = Camera(
        camPtr=cam_ptr,
        gain=10.0,
        exposure_time=10000.0,
        frame_rate=30.0,
        mode='video',
        syncMode=True
    )
    
    # Acquire images
    camera.start()
    image = camera.get_image()
    camera.stop()
    camera.release()

Classes
-------

Camera
~~~~~~

.. autoclass:: camera.Camera
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Main camera control class. Manages configuration and acquisition for a single camera.

**Public Methods:**

* :meth:`~camera.Camera.configure` - Configure all camera settings
* :meth:`~camera.Camera.change_mode` - Change operation mode
* :meth:`~camera.Camera.get_image` - Retrieve next image
* :meth:`~camera.Camera.start` - Start acquisition
* :meth:`~camera.Camera.stop` - Stop acquisition
* :meth:`~camera.Camera.release` - Release camera resources

CameraMode
~~~~~~~~~~

.. autoclass:: camera.CameraMode
   :members:
   :undoc-members:

Enumeration of camera operation modes:

* ``IMAGE``: Single frame capture
* ``VIDEO``: Continuous video capture
* ``STREAM``: Streaming mode

CameraConfig
~~~~~~~~~~~~

.. autoclass:: camera.CameraConfig
   :members:
   :undoc-members:

Configuration constants for camera operation:

* ``VIDEO_BUFFER_COUNT``: Number of buffers for video mode (10)
* ``IMAGE_BUFFER_COUNT``: Number of buffers for image mode (1)
* ``PACKET_SIZE``: Network packet size (9000 bytes)
* ``PACKET_DELAY_MULTIPLIER``: Packet delay multiplier (2)
* ``THROUGHPUT_ALIGNMENT``: Throughput alignment value (16000)
* ``IMAGE_TIMEOUT_MS``: Image acquisition timeout (100ms)

CameraConfigurationError
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: camera.CameraConfigurationError

Exception raised when camera configuration fails.

Usage Guide
-----------

Initialization
~~~~~~~~~~~~~~

Create a Camera instance with your desired configuration:

::

    camera = Camera(
        camPtr=cam_ptr,           # PySpin camera pointer
        gain=10.0,                # Camera gain
        exposure_time=10000.0,    # Exposure in microseconds
        frame_rate=30.0,          # Frame rate in Hz
        mode='video',             # 'image', 'video', or 'stream'
        syncMode=True             # Hardware trigger on/off
    )

The camera is automatically initialized and configured upon instantiation.

Acquisition Modes
~~~~~~~~~~~~~~~~~

**Image Mode** (``mode='image'``):

* Single frame capture
* Uses 1 buffer
* Retry logic on acquisition failure
* 100ms timeout

**Video Mode** (``mode='video'``):

* Continuous capture
* Uses 10 buffers for smooth streaming
* Infinite timeout

**Stream Mode** (``mode='stream'``):

* Similar to video mode
* Uses 1 buffer

Synchronization Modes
~~~~~~~~~~~~~~~~~~~~~

**Hardware Trigger** (``syncMode=True``):

* Waits for external trigger signal on Line0
* Rising edge activation
* Trigger overlap mode: ReadOut
* Ideal for multi-camera synchronization

**Free-Running** (``syncMode=False``):

* Captures at configured frame rate
* Uses internal timer
* No external trigger needed

Changing Modes
~~~~~~~~~~~~~~

You can change the operation mode after initialization:

::

    camera.change_mode(mode='image', syncMode=False)

This will reconfigure buffers and trigger settings accordingly.

Image Acquisition
~~~~~~~~~~~~~~~~~

::

    camera.start()
    
    # Get images
    while capturing:
        image = camera.get_image()
        # Process image...
    
    camera.stop()

Cleanup
~~~~~~~

Always release camera resources when done:

::

    camera.release()

After calling ``release()``, the camera object cannot be reused.

Configuration Details
---------------------

The Camera class configures the following parameters automatically:

Gain Configuration
~~~~~~~~~~~~~~~~~~

* Disables automatic gain control
* Sets gain value (clamped to hardware maximum)

Exposure Configuration
~~~~~~~~~~~~~~~~~~~~~~

* Disables automatic exposure
* Sets exposure time in microseconds (clamped to maximum)

Frame Rate Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

(Only in free-running mode)

* Disables trigger mode
* Disables automatic frame rate
* Sets acquisition frame rate

Trigger Configuration
~~~~~~~~~~~~~~~~~~~~~

(Only in hardware trigger mode)

* Sets trigger selector to FrameStart
* Sets trigger source to Line0 (hardware)
* Sets trigger activation to RisingEdge
* Enables trigger overlap (ReadOut mode)

Buffer Configuration
~~~~~~~~~~~~~~~~~~~~

* Sets buffer handling mode
* Sets buffer count mode to Manual
* Sets buffer count based on operation mode

Network Configuration
~~~~~~~~~~~~~~~~~~~~~

(For GigE cameras)

* Sets packet size to 9000 bytes
* Optimizes throughput limit
* Configures packet delay (if needed)

Chunk Data Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

* Enables chunk mode
* Enables timestamp chunk data for frame timing

Error Handling
--------------

The Camera class uses logging for warnings and errors:

::

    import logging
    logging.basicConfig(level=logging.INFO)

Configuration failures raise ``CameraConfigurationError``.

Best Practices
--------------

1. **Always release resources**: Call ``camera.release()`` when done
2. **Check hardware limits**: Gain and exposure are automatically clamped
3. **Use appropriate mode**: Image mode for single shots, video for continuous
4. **Enable logging**: Set logging level to INFO or DEBUG for troubleshooting
5. **Hardware trigger delays**: Note that some cameras need 1 second after trigger setup

Example: Multi-Camera Setup
----------------------------

::

    import PySpin as ps
    from camera import Camera
    
    system = ps.System.GetInstance()
    cam_list = system.GetCameras()
    
    # Create separate Camera instance for each physical camera
    cameras = []
    for i, cam_ptr in enumerate(cam_list):
        camera = Camera(
            camPtr=cam_ptr,
            gain=10.0,
            exposure_time=10000.0,
            mode=