# src/capture/camera/

## Overview
This system is a distributed camera capture system that captures images and videos from cameras distributed across multiple PCs. 

## Image Capture Module

### 1. `image.py` 
**Purpose**: Script for capturing images on a single PC using keyboard input

**Main Functions**:
- **Main loop**: Detects keyboard events to execute image capture

**Input/Output**:
- **Input**: 
  - `--save_path`: Save path (required)
  - Keyboard input: `c` (capture), `q` (quit)
- **Output**: Saves image files to specified path

**Execution**:
```bash
python src/capture/camera/image.py --save_path <save_path>
```

---

### 2. `image_client.py` 
**Purpose**: Client that receives commands from server to capture images

**Main Functions**:
- **`listen_for_commands()`**: Receives and processes commands from server
  - Input: ZMQ socket messages
  - Output: Executes actions based on commands

**Input/Output**:
- **Input**:
  - `--save_path`: Save path (required)
  - ZMQ messages: `capture:<index>`, `quit`
- **Output**: 
  - Saves image files
  - Sends status messages to server

**Execution**:
```bash
python src/capture/camera/image_client.py --save_path <save_path>
```
---

### 3. `image_main.py` 
**Purpose**: Manages multiple PC camera clients and performs synchronized image capture

**Main Functions**:
- **`listen_socket(pc_name, socket)`**: Monitors each PC's socket
  - Input: PC name, ZMQ socket
  - Output: Status updates
- **`wait_for_camera_ready()`**: Waits until all cameras are ready
- **`wait_for_capture()`**: Waits until all captures are completed

**Input/Output**:
- **Input**:
  - `--save_path`: Save path (required)
  - Keyboard input: `c` (capture), `q` (quit)
  - PC configuration file: `config/environment/pc.json`
- **Output**: 
  - Sends capture commands to each PC
  - Synchronized image saving

**Execution**:
```bash
python src/capture/camera/image_main.py --save_path <save_path>
```

**Operation Process**:
1. Load PC configuration file
2. Execute Git pull and client scripts on all PCs
3. Set up ZMQ connections with all PCs
4. Wait for all cameras to be ready
5. Send simultaneous capture commands to all PCs when `c` key is pressed
6. Wait for all captures to complete before preparing for next capture

---

## Video Capture Module

### 4. `video.py` 
**Purpose**: Continuous video capture on a single PC

**Main Functions**:
- **Main loop**: Starts video recording and waits for quit signal

**Input/Output**:
- **Input**: 
  - `--save_path`: Save path (required)
  - Keyboard input: `q` (quit)
- **Output**: Saves video files to specified path

**Execution**:
```bash
python src/capture/camera/video.py --save_path <save_path>
```
---

### 5. `video_client.py` 
**Purpose**: Client that receives commands from server to capture video

**Main Functions**:
- **`listen_for_commands()`**: Receives quit commands from server
  - Input: ZMQ socket messages
  - Output: Sets termination event

**Input/Output**:
- **Input**:
  - `--save_path`: Save path (required)
  - ZMQ messages: `quit`
- **Output**: 
  - Saves video files
  - Sends status messages to server

**Execution**:
```bash
python src/capture/camera/video_client.py --save_path <save_path>
```
---

### 6. `video_main.py` 
**Purpose**: Manages multiple PC video capture clients and performs synchronized video recording

**Main Functions**:
- **`listen_socket(pc_name, socket)`**: Monitors each PC's socket
  - Input: PC name, ZMQ socket
  - Output: Status updates
- **`wait_for_camera_ready()`**: Waits until all cameras are ready

**Input/Output**:
- **Input**:
  - `--save_path`: Save path (required)
  - Keyboard input: `q` (quit)
  - PC configuration file: `config/environment/pc.json`
- **Output**: 
  - Saves video files on each PC
  - Synchronized video recording

**Execution**:
```bash
python src/capture/camera/video_main.py --save_path <save_path>
```

**Operation Process**:
1. Load PC configuration file
2. Execute Git pull and client scripts on all PCs
3. Set up ZMQ connections with all PCs
4. Wait for all cameras to be ready
5. Start simultaneous video recording on all PCs
6. Send quit commands to all PCs when `q` key is pressed

---

# src/capture/robot/

## 1. teaching.py 

**Purpose**: A tool for collecting pose data by allowing users to manually move the XArm robot in manual mode.

### Input/Output
- **Input**: 
  - Keyboard input ('c': capture, 'q': quit)
  - Robot IP address (hardcoded: "192.168.1.221")
- **Output**: 
  - Numpy files in `demo_asdf/` folder (0.npy, 1.npy, ...)
  - Each file contains 6DOF pose data (x, y, z, rx, ry, rz)

### Main Functions

#### `listen_keyboard()`
- **Purpose**: Detects keyboard input in background
- **Input**: None
- **Output**: Manipulates threading.Event objects

#### Main Loop
- **Purpose**: Monitor robot state and save pose data
- **Input**: save_event, stop_event
- **Output**: .npy files

---

## 2. teleop_sim.py - Teleoperation in Simulation Environment

**Purpose**: A tool for remote control of robots in simulation environment using XSens or Oculus devices.

### Key Features
- **Multiple Input Device Support**: XSens motion capture, Oculus VR headset
- **Real-time Simulation**: Physics simulation through Isaac Sim
- **Video/State Recording**: Optional recording of simulation results

### Input/Output
- **Input**: 
  - Command line arguments (--device, --arm, --hand)
  - Teleoperation device data
- **Output**: 
  - Video files (optional): `teleop_sim/video/`
  - State files (optional): `teleop_sim/state/`

### Main Components

#### Simulator Setup
- **Simulator Class**: Isaac Sim-based physics simulation
- **Configuration Options**:
  - `view_physics`: Physics simulation visualization
  - `view_replay`: Replay mode
  - `headless`: Headless mode (no GUI)
  - `save_video`: Video recording
  - `save_state`: State recording

#### Main Loop
- **Purpose**: Real-time teleoperation and simulation step execution
- **Input**: Teleoperation device data
- **Output**: Robot actions, simulation state updates
- **Description**: 
  1. Acquire hand pose data from teleoperation device
  2. Convert to robot actions through retargeting
  3. Control robot in simulation environment

---
