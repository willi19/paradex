# src/calibration/extrinsic/

## Overview

This system provides a comprehensive pipeline for calibrating extrinsic parameters of multiple cameras using ChArUco boards. The main components include:

1. **Data Capture** (capture.py, client.py)
2. **Database Generation** (generate_database.py)
3. **Scale Adjustment** (adjust_length.py)

## Files Description

### 1. capture.py 

**Purpose**
- Collects ChArUco board detection data in real-time from cameras connected to multiple PCs
- Provides integrated UI for visualizing and controlling the capture process
- Client-server communication via ZMQ

#### Functions

**`draw_charuco_corners_custom(image, corners, color, radius, thickness, ids)`**
- **Purpose**: Visualizes ChArUco corner points on images
- **Input**: 
  - `image`: Target image
  - `corners`: Detected corner coordinates
  - `color`: Drawing color
  - `radius`, `thickness`: Circle size and thickness
  - `ids`: Corner point IDs (optional)
- **Output**: Modified image (in-place)

**`listen_socket(pc_name, socket)`**
- **Purpose**: Asynchronously processes messages from each PC
- **Input**: PC name, ZMQ socket
- **Message Types**:
  - `"terminated"`: Client shutdown
  - `"camera_ready"`: Camera ready
  - `"camera_error"`: Camera error
  - `"save_finish"`: Save completed
  - ChArUco detection result JSON

**`main_ui_loop()`**
- **Purpose**: Provides real-time grid viewer and handles user input
- **Input**:
  - `'q'`: Quit
  - `'c'`: Execute capture
- **Output**: Real-time feed from all cameras displayed in grid format

#### Execution
```bash
python src/calibration/extrinsic/capture.py
```

#### Configuration Requirements
- `config/environment/pc.json`: PC and camera configuration
- `config/environment/charuco_info.json`: ChArUco board information

---

### 2. client.py 

**Purpose**
- Runs on individual PCs to capture camera data
- Real-time ChArUco board detection
- Saves data according to server commands

#### Functions

**`listen_for_commands()`**
- **Purpose**: Asynchronously receives commands from server
- **Commands**:
  - `"quit"`: Terminate client
  - `"capture:index"`: Capture data with specified index

**Main Loop**
- **Purpose**: 
  - Continuously processes camera frames
  - Detects ChArUco boards
  - Sends detection results to server
  - Saves data to disk when capture command is received

#### Execution
```bash
python src/calibration/extrinsic/client.py --save_path TIMESTAMP
```

#### Input/Output
- **Input**: 
  - Camera stream
  - Server commands (ZMQ)
- **Output**: 
  - `{serial_num}_cor.npy`: Detected corner coordinates
  - `{serial_num}_id.npy`: Corner point IDs
  - Detection results sent to server (JSON)

---

### 3. generate_database.py 

**Purpose**
- Converts captured ChArUco data into COLMAP format database
- Performs keypoint matching between cameras
- Performance optimization through multiprocessing

#### Functions

**`load_keypoint(root_dir)`**
- **Purpose**: Loads captured keypoint data
- **Input**: Root directory path
- **Output**: Dictionary of keypoints organized by index and camera

**`process_match(args)`**
- **Purpose**: Processes matching between camera pairs and calculates two-view geometry
- **Input**: Matching information and camera parameters
- **Output**: Two-view geometry results

**`parallel_processing(tot_kypt_matches, tot_kypt_dict, cam_keys)`**
- **Purpose**: Accelerates matching processing through multiprocessing
- **Input**: Total matching information, keypoint dictionary, camera keys
- **Output**: Adds two-view geometry to COLMAP database

**`draw_keypoint(image, corners, color, radius, thickness, ids)`**
- **Purpose**: Draws keypoints on images for visualization
- **Input**: Image, corner points, style parameters
- **Output**: Modified image

#### Execution
```bash
# Use latest data
python src/calibration/extrinsic/generate_database.py --latest

# Use specific dataset
python src/calibration/extrinsic/generate_database.py --name DATASET_NAME
```

#### Input/Output
- **Input**: 
  - Captured ChArUco data (`{serial_num}_cor.npy`, `{serial_num}_id.npy`)
  - Camera intrinsic parameters
- **Output**: 
  - COLMAP database (`database.db`)
  - Visualization images (`images/{serial_num}.jpg`)
  - COLMAP result(`points3D.bin`, `images.bin`, `cameras.bin`) (commented out)

---

### 4. adjust_length.py 

**Purpose**
- Adjusts the scale of 3D reconstruction results to actual physical dimensions
- Calculates scale factor based on known ChArUco board dimensions
- Saves adjusted extrinsic parameters

#### Processing Steps

**Keypoint Undistortion and Normalization**
- Removes camera distortion and converts to normalized coordinate system

**3D Triangulation**
- Uses `ransac_triangulation` for robust 3D point estimation

**Scale Calculation**
- Measures distances between adjacent points on ChArUco board
- Compares with actual size (0.025m) to calculate scale factor

**Projection Error Calculation**
- Measures reprojection error for each camera

#### Execution
```bash
# Use latest data
python src/calibration/extrinsic/adjust_length.py --latest

# Use specific dataset
python src/calibration/extrinsic/adjust_length.py --name DATASET_NAME
```

#### Input/Output
- **Input**: 
  - Captured ChArUco data
  - COLMAP calibration results
- **Output**: 
  - Adjusted extrinsic parameters (`extrinsics.json`)
  - Intrinsic parameters (`intrinsics.json`)
  - Scale statistics output

## Complete Workflow

1. **Setup Preparation**
   - `config/environment/pc.json`: PC and camera configuration
   - `config/environment/charuco_info.json`: ChArUco board information
   - `config/camera/camera_index.json`: Camera index mapping

2. **Data Capture**
   ```bash
   python src/calibration/extrinsic/capture.py
   ```
   - Run server and automatically execute clients on each PC
   - Press 'c' key to capture data

3. **Database Generation**
   ```bash
   python src/calibration/extrinsic/generate_database.py --latest
   ```

4. **COLMAP Execution** (Manual)
   - Use COLMAP GUI or CLI to perform Structure from Motion

5. **Scale Adjustment**
   ```bash
   python src/calibration/extrinsic/adjust_length.py --latest
   ```

## File Structure

```
shared_dir/
├── extrinsic/
│   └── TIMESTAMP/
│       ├── 0/
│       │   ├── {serial_num}_cor.npy
│       │   ├── {serial_num}_id.npy
│       │   ├── images/
│       │   └── colmap/
│       ├── 1/
│       └── ...
└── cam_param/
    └── TIMESTAMP/
        ├── extrinsics.json
        └── intrinsics.json
```



# src/calibration/intrinsic/

## Overview
A system consisting of 3 scripts for camera intrinsic parameter calibration. Uses ChArUco boards to calculate camera intrinsic matrix and distortion coefficients.

## Files Description

### 1. `capture.py` 

**Purpose**
- Main server that controls camera calibration process on remote PCs
- Communicates with clients via ZMQ and visualizes real-time ChArUco detection results

#### Functions

**`get_pc_info(serial_num)`**
- **Purpose**: Finds PC information where the camera with given serial number is connected from configuration file
- **Input**: `serial_num` (str) - Camera serial number
- **Output**: `pc_name` (str), `ip` (str) - PC name and IP address

**`wait_for_keypress(socket)`**
- **Purpose**: Waits for 'q' key input to send program termination signal
- **Input**: `socket` (zmq.Socket) - ZMQ socket
- **Output**: None

**`draw_charuco_corners_custom(image, corners, color, radius, thickness, ids)`**
- **Purpose**: Visualizes ChArUco corners on the image
- **Input**: 
  - `image` (np.array) - Image to draw on
  - `corners` (np.array) - Corner coordinates
  - `color` (tuple) - Color
  - `radius`, `thickness` (int) - Circle radius and thickness
  - `ids` (np.array, optional) - Corner IDs
- **Output**: None (draws directly on image)


#### Usage
```bash
python src/calibration/intrinsic/capture.py --serial <camera_serial_number>
```

#### Input/Output
- **Input**: Camera serial number
- **Output**: Real-time ChArUco detection result visualization window

---

### 2. `client.py` 

**Purpose**
- Runs on remote PC to capture images from camera and detect ChArUco patterns
- Sends detected corner data to server and saves valid data

#### Functions

**`listen_for_commands()`**
- **Purpose**: Receives commands from server to handle registration/termination
- **Input**: None (uses global variables)
- **Output**: None


**`should_save(result)`**
- **Purpose**: Determines if detected corners are worth saving
  - Checks if all 70 corners are detected
  - Ensures data is not too similar to previously saved data
- **Input**: `result` (dict) - ChArUco detection result
- **Output**: `bool` - Whether to save or not


#### Usage
```bash
python src/calibration/intrinsic/client.py --serial <camera_serial_number>
```

#### Input/Output
- **Input**: Camera serial number
- **Output**: 
  - Real-time detection results sent via ZMQ
  - Corner data saved as `shared_dir/intrinsic/{serial}/keypoint/{datetime}.npy`

---

### 3. `calculate.py` 

**Purpose**
- Calculates camera intrinsic parameters using saved ChArUco corner data
- Uses OpenCV's `calibrateCamera` function to compute intrinsic matrix and distortion coefficients

#### Logic
- ChArUco board configuration: 10x7 inner corners, 2.5cm square size
- Image size: 2048x1536
- Calibration through 3D object point and 2D image point matching

#### Usage
```bash
# Specify serial number and date
python src/calibration/intrinsic/calculate.py --serial <camera_serial> --date <date_file.npy>

# Specify serial number only (uses latest data automatically)
python src/calibration/intrinsic/calculate.py --serial <camera_serial>

# Process all serial numbers
python src/calibration/intrinsic/calculate.py
```

#### Input/Output
- **Input**: 
  - `shared_dir/intrinsic/{serial}/keypoint/{date}.npy` - Saved corner data
- **Output**: 
  - `shared_dir/intrinsic/{serial}/param/{date}.json` - Calibration results
  - Console output of RMS error, intrinsic matrix, and distortion coefficients

#### Output JSON Format
```json
{
    "RMS_error": 0.234,
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "distortion": [k1, k2, p1, p2, k3],
    "width": 2048,
    "height": 1536
}
```

## Usage Workflow

1. **Data Collection**: 
   - Run `capture.py` on server
   - `client.py` automatically runs on remote PC
   - Capture ChArUco board from various angles and positions

2. **Calibration Calculation**:
   - Run `calculate.py` to compute camera parameters from collected data


# src/calibration/handeyecalibration/

## Overview
This system performs hand-eye calibration for a robotic arm with multiple cameras. It captures images at different robot poses and calculates the transformation between the camera coordinate system and the robot coordinate system.


## Files Description

### 1. `capture.py`

**Purpose**: Controls the robot arm and coordinates image capture across multiple camera clients.

#### Classes and Functions

##### `DexArmControl`
- **Purpose**: Controls XArm6 robotic arm
- **Input**: 
  - `xarm_ip_address` (str): IP address of the XArm controller (default: "192.168.1.221")
- **Methods**:
  - `move_arm(target_action)`: Moves arm to target pose (axis-angle representation)
  - `get_joint_values()`: Returns current joint angles
  - `reset()`: Resets arm state and clears errors
  - `quit()`: Safely disconnects from arm

**`copy_calib_files(save_path)`**
- **Purpose**: Copies latest camera calibration files to save directory
- **Input**: `save_path` (str) - destination directory
- **Output**: Copies cam_param folder to save_path

##### Communication Functions
- `listen_socket(pc_name, socket)`: Handles incoming messages from clients
- `wait_for_camera_ready()`: Waits for all cameras to be ready
- `wait_for_capture()`: Waits for all captures to complete

#### Execution Flow
1. Initialize robot arm
2. Start camera clients on remote PCs
3. Register all clients via ZMQ
4. For each calibration pose (6 poses):
   - Move robot to target position
   - Trigger image capture on all cameras
   - Save robot joint angles
5. Copy calibration files and cleanup

#### How to Run
```bash
python src/calibration/handeyecalibration/capture.py
```

**Prerequisites**:
- Robot poses must be pre-saved in `hecalib/` directory as `.npy` files (0.npy to 5.npy)
- Camera configuration in `config/environment/pc.json`

---

### 2. `client.py` 

**Purpose**: Runs on each PC with cameras, handles image capture commands from server.

#### Functions

**`listen_for_commands()`**
- **Purpose**: Listens for commands from capture server
- **Commands Handled**:
  - `quit`: Terminate client
  - `capture:index`: Start image capture for given index

#### Execution Flow
1. Connect to server via ZMQ ROUTER socket
2. Register with server
3. Initialize camera manager
4. Wait for capture commands
5. Capture and save images when commanded
6. Notify server when capture complete

#### How to Run
```bash
python src/calibration/handeyecalibration/client.py
```

---

### 3. `preprocess.py` 

**Purpose**: Processes captured images to detect ArUco markers and compute 3D positions.

#### Functions

##### Main Processing Pipeline
1. **Robot Forward Kinematics**: Computes link6 pose from joint angles
2. **Image Undistortion**: Removes camera distortion from captured images
3. **ArUco Detection**: Detects ArUco markers in undistorted images
4. **3D Triangulation**: Computes 3D positions of markers from multiple camera views

#### Input/Output
- **Input**: 
  - Raw images in `handeye_calibration/{name}/{index}/image/`
  - Robot joint angles in `robot.npy`
  - Camera calibration parameters
- **Output**:
  - `link6.npy`: 4x4 transformation matrix of robot end-effector
  - `marker_3d.npy`: 3D positions of detected ArUco markers
  - `undist_image/`: Undistorted images
  - `debug/`: Debug images with marker annotations (if debug=True)

#### How to Run
```bash
python src/calibration/handeyecalibration/preprocess.py --name <calibration_name>
```

**Arguments**:
- `--name`: Name of calibration directory (optional, uses latest if not specified)

---

### 4. `get_C2R.py` 

**Purpose**: Solves the hand-eye calibration problem using the AX=XB formulation.

#### Functions

##### Main Algorithm
1. **Pose Difference Calculation**: Computes relative poses between consecutive robot positions
2. **Marker Registration**: Matches 3D marker positions between different poses
3. **Tsai-Lenz Algorithm**: Solves AX=XB equation for camera-to-robot transformation
4. **Marker Position Estimation**: Computes consistent 3D marker positions in robot frame

#### Input/Output
- **Input**:
  - `link6.npy`: Robot end-effector poses
  - `marker_3d.npy`: 3D marker positions in camera frame
- **Output**:
  - `C2R.npy`: 4x4 camera-to-robot transformation matrix
  - `marker_pos.npy`: Marker positions in robot coordinate system

#### How to Run
```bash
python src/calibration/handeyecalibration/get_C2R.py --name <calibration_name>
```

**Arguments**:
- `--name`: Name of calibration directory (optional, uses latest if not specified)

---

## Usage Workflow

### Complete Calibration Process

1. **Data Capture**:
   ```bash
   python src/calibration/handeyecalibration/capture.py
   ```

2. **Data Processing**:
   ```bash
   python src/calibration/handeyecalibration/preprocess.py
   ```

3. **Calibration Computation**:
   ```bash
   python src/calibration/handeyecalibration/get_C2R.py
   ```

### File Structure
```
shared_data/handeye_calibration/{timestamp}/
├── 0/
│   ├── image/           # Raw captured images
│   ├── undist_image/    # Undistorted images
│   ├── debug/           # Debug visualizations
│   ├── robot.npy        # Joint angles
│   ├── link6.npy        # End-effector pose
│   ├── marker_3d.npy    # 3D marker positions
│   ├── C2R.npy          # Camera-to-robot transformation
│   ├── marker_pos.npy   # Marker positions in robot frame
│   └── cam_param/       # Camera calibration parameters
├── 1/ ... 5/            # Additional poses
```


# src/calibration/eef

## Overview
This system performs Hand-Eye calibration using XArm robot and Allegro Hand. 

## File Structure

### 1. `capture.py`
**Purpose**: Server that captures calibration data using multi-camera system

#### Main Classes/Functions:

**`DexArmControl`**
- **Purpose**: Control XArm and Allegro Hand
- **Initialization Parameters**: 
  - `xarm_ip_address` (default: "192.168.1.221")
- **Methods**:
  - `move_arm(target_action)`: Move XArm to specified pose
  - `move_hand(allegro_angles)`: Move Allegro Hand to specified angles
  - `get_joint_values()`: Return current joint angles of robot and hand
  - `reset()`: Initialize robot state
  - `quit()`: Close connection

**`copy_calib_files(save_path)`**
- **Purpose**: Copy latest camera parameter files to calibration directory
- **Input**: `save_path` - destination path
- **Output**: None

#### Execution:
```bash
python src/calibration/eef/capture.py
```

#### Input/Output:
- **Input**: 
  - `hecalib/{i}.npy`: Robot target pose files (i=0~5)
  - `data/calibration_pose/hand_{i}.npy`: Hand pose files
- **Output**: 
  - `{shared_dir}/eef/{timestamp}/{i}/robot.npy`: Robot joint values
  - `{shared_dir}/eef/{timestamp}/{i}/image/`: Captured images
  - `{shared_dir}/eef/{timestamp}/0/cam_param/`: Camera parameters

---

### 2. `client.py`
**Purpose**: Client that runs on each PC to control cameras and capture images

#### Main Functions:

**`listen_for_commands()`**
- **Purpose**: Receive and process commands from server
- **Handled Commands**:
  - `quit`: Terminate client
  - `capture:{index}`: Capture images with specified index

#### Execution:
```bash
python src/calibration/eef/client.py
```

#### Input/Output:
- **Input**: Server commands via ZMQ
- **Output**: Captured images saved to specified directory

---

### 3. `preprocess.py`
**Purpose**: Detect ArUco markers in captured images and calculate 3D coordinates

#### Main Features:
- Image undistortion
- ArUco marker detection
- 3D triangulation using multiple cameras
- Robot forward kinematics calculation

#### Execution:
```bash
python src/calibration/eef/preprocess.py [--name CALIBRATION_NAME]
```

#### Arguments:
- `--name`: Calibration directory name (uses latest directory if omitted)

#### Input/Output:
- **Input**: 
  - `{he_calib_path}/{idx}/robot.npy`: Robot joint values
  - `{he_calib_path}/{idx}/image/`: Original images
  - `{he_calib_path}/0/cam_param/`: Camera parameters
- **Output**: 
  - `{he_calib_path}/{idx}/link6.npy`: Robot end-effector pose
  - `{he_calib_path}/{idx}/undist_image/`: Undistorted images
  - `{he_calib_path}/{idx}/marker_3d.npy`: 3D coordinates of detected markers
  - `{he_calib_path}/{idx}/debug/`: Debug images (marker detection visualization)

---

### 4. `get_C2R.py`
**Purpose**: Perform Hand-Eye calibration to calculate transformation matrix between camera and robot

#### Main Functions:

**`logR(T)`**
- **Purpose**: Calculate logarithmic map of rotation matrix
- **Input**: 4x4 transformation matrix
- **Output**: 3D rotation vector

**`Calibrate(A, B)`**
- **Purpose**: Solve AX=XB problem to perform Hand-Eye calibration
- **Input**: 
  - `A`: List of transformations in camera coordinate system
  - `B`: List of transformations in robot coordinate system
- **Output**: 
  - `theta`: Rotation matrix
  - `b_x`: Translation vector

#### Execution:
```bash
python src/calibration/eef/get_C2R.py [--name CALIBRATION_NAME]
```

#### Arguments:
- `--name`: Calibration directory name (uses latest directory if omitted)

#### Input/Output:
- **Input**: 
  - `{he_calib_path}/{idx}/robot.npy`: Robot joint values
  - `{he_calib_path}/{idx}/marker_3d.npy`: Marker 3D coordinates
  - `{he_calib_path}/0/`: C2R transformation file
- **Output**: Hand-Eye calibration results (console output)

---

## Complete Execution Sequence

1. **Data Capture**:
   ```bash
   python src/calibration/eef/capture.py
   ```

2. **Preprocessing**:
   ```bash
   python src/calibration/eef/preprocess.py
   ```

3. **Hand-Eye Calibration**:
   ```bash
   python src/calibration/eef/get_C2R.py
   ```

