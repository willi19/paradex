"""
paradex.io - Input/Output Module
=================================

This module provides various I/O interfaces for the Paradex system.

Common Architecture
-------------------

All I/O controllers in this module follow a consistent **thread-based asynchronous control pattern**:

**Core Components:**

1. **Background Thread**
   - Each controller runs a dedicated background thread for continuous operation
   - Thread is started in ``__init__()`` and runs until ``quit()`` is called
   - Main control loop: ``move_arm()``, ``move_hand()``, ``run_camera()``, ``run()``

2. **Thread Synchronization Primitives**
   - ``threading.Event``: Signal flags for state management
     
     - ``exit``: Global termination signal
     - ``start_capture/start_event``: Begin data acquisition
     - ``save_event/save_end``: Save completion notification
   
   - ``threading.Lock``: Thread-safe data access
     
     - Protects shared state (``target_action``, ``data``, etc.)
     - Used in ``with self.lock:`` blocks for atomic operations

3. **Data Recording System**
   - ``start(save_path)``: Initializes data buffers and sets save location
   - Continuous data collection during operation (position, time, action, etc.)
   - ``end()``: Triggers ``save()`` to persist data as ``.npy`` files
   - ``save()``: Writes collected data to disk in NumPy format

4. **Lifecycle Methods**
   
   .. code-block:: python
   
       # 1. Initialization
       controller = XArmController()  # Spawns background thread
       
       # 2. Optional: Set home position
       controller.home_robot(homepose)
       
       # 3. Start recording
       controller.start(save_path="/path/to/data")
       
       # 4. Control loop (from main thread)
       while running:
           controller.set_action(action)  # Thread-safe update
           current_state = controller.get_qpos()
       
       # 5. Stop recording
       controller.end()  # Saves data automatically
       
       # 6. Cleanup
       controller.quit()  # Stops thread and releases resources

**Design Benefits:**

- **Non-blocking Control**: Main thread stays responsive while I/O operations run asynchronously
- **Real-time Performance**: Dedicated threads ensure consistent control loop timing (typically 100Hz)
- **Thread Safety**: Lock-protected state ensures data consistency across threads
- **Automatic Data Logging**: Seamless recording without manual intervention

Submodules
----------
camera
    Camera control and video capture interfaces
    
    - ``CameraManager``: Multi-camera synchronization with hardware triggering
    - ``Camera``: Individual camera configuration (gain, exposure, trigger modes)
    - ``TimecodeReceiver``: Hardware timestamp synchronization

capture_pc
    PC-based capture system for multi-camera recording
    
robot_controller
    Controllers for various robot manipulators
    
    - ``FrankaController``: Franka Emika Panda (7-DOF arm)
    - ``XArmController``: UFactory XArm (6-DOF arm) 
    - ``AllegroController``: Allegro Hand (16-DOF)
    - ``InspireController``: Inspire Hand (6-DOF)

contact
    Contact sensor data processing

signal_generator
    Signal generation utilities for hardware synchronization

teleop
    Teleoperation interfaces
    
    - ``occulus.receiver``: Oculus VR controller input
    - ``xsens.receiver``: Xsens motion capture data

Implementation Notes
--------------------

**Thread Control Pattern**

All controllers use a standard thread control loop:

.. code-block:: python

    def move_arm(self):  # or move_hand(), run_camera(), etc.
        while not self.exit.is_set():
            start_time = time.time()
            
            with self.lock:
                # 1. Read target action
                action = self.target_action.copy()
                
                # 2. Execute hardware command
                self.arm.set_servo_cartesian(action)
                
                # 3. Read current state
                state = self.arm.get_joint_states()
                
                # 4. Record data if capturing
                if self.save_path is not None:
                    self.data["position"].append(state)
                    self.data["time"].append(start_time)
            
            # 5. Maintain control frequency
            elapsed = time.time() - start_time
            time.sleep(max(0, 1/self.fps - elapsed))

**Data Recording Format**

All controllers save data as NumPy arrays:

.. code-block:: text

    save_path/
    ├── position.npy    # Actual measured state
    ├── action.npy      # Commanded target state  
    ├── time.npy        # Timestamps
    ├── torque.npy      # Force/torque readings (if available)
    └── velocity.npy    # Velocity data (if available)

Examples
--------

**Robot Controller**

>>> from paradex.io.robot_controller import XArmController
>>> 
>>> # Initialize and home
>>> robot = XArmController()
>>> robot.home_robot(home_pose)
>>> 
>>> # Start recording
>>> robot.start(save_path="./data/episode_01")
>>> 
>>> # Control loop
>>> for step in range(1000):
>>>     target = compute_target_pose()
>>>     robot.set_action(target)
>>>     time.sleep(0.01)
>>> 
>>> # Save and cleanup
>>> robot.end()
>>> robot.quit()

**Multi-Camera Capture**

>>> from paradex.io.camera.camera_loader import CameraManager
>>> 
>>> # Initialize with hardware sync
>>> cameras = CameraManager(
>>>     mode="video",
>>>     serial_list=['24070014', '24070015'],
>>>     syncMode=True
>>> )
>>> 
>>> # Start synchronized recording
>>> cameras.start(save_dir="./captures/demo")
>>> time.sleep(10)  # Record for 10 seconds
>>> cameras.end()
>>> 
>>> # Change mode dynamically
>>> cameras.change_mode("stream", syncMode=False)
>>> cameras.start()
>>> img = cameras.get_data(index=0)  # Real-time access
>>> cameras.quit()

See Also
--------
- :doc:`api/robot_controller` - Robot manipulator controllers
- :doc:`api/camera` - Camera capture system
- :doc:`api/teleop` - Teleoperation interfaces
"""