# paradex/utils

## 1. file_io.py

### Overview
A comprehensive file I/O utility module for handling camera calibration, video processing, robot trajectory data, and various other data types in the Paradex project.

### Path Configurations
```python
rsc_path = "../../rsc"
home_path = "~"
shared_dir = "~/shared_data"
calib_path_list = ["~/captures1/calibration", "~/captures2/calibration"]
capture_path_list = ["~/captures1", "~/captures2"]
download_dir = "~/download"
cam_param_dir = "~/shared_data/cam_param"
handeye_calib_path = "~/shared_data/handeye_calibration"
config_dir = "../../config"
```

### Functions

#### Directory Management

**`get_video_list(video_dir)`**
- **Purpose**: Retrieves a list of video files from the specified directory
- **Input**: `video_dir` (str) - Directory path containing video files
- **Output**: `video_list` (list) - List of tuples (video_file_path, timestamp_file_path)
- **Supported formats**: `.avi`, `.mp4`

**`find_latest_directory(directory)`**
- **Purpose**: Finds the latest directory within the specified directory
- **Input**: `directory` (str) - Directory path to search
- **Output**: `latest_dir` (str) - Name of the latest directory

**`find_latest_index(directory)`**
- **Purpose**: Finds the directory with the highest index number
- **Input**: `directory` (str) - Directory path to search
- **Output**: `latest_dir` (int) - Highest index number

#### Camera Parameter Loading

**`load_cam_param(name=None)`**
- **Purpose**: Loads camera intrinsic and extrinsic parameters (current version)
- **Input**: `name` (str, optional) - Specific calibration set name (uses latest if None)
- **Output**: `(intrinsic, extrinsic)` - Dictionary containing intrinsic and extrinsic parameters

**`load_cam_param_prev(name=None)`** *(deprecated)*
- **Purpose**: Loads camera parameters using previous version format
- **Input/Output**: Same as `load_cam_param()`

**`load_cam_param_temp(name=None)`**
- **Purpose**: Loads camera parameters using temporary extrinsic file
- **Input/Output**: Same as `load_cam_param()` but uses `extrinsics_temp.json`

**`load_intrinsic()`**
- **Purpose**: Loads intrinsic parameters for all cameras from shared directory
- **Input**: None
- **Output**: `intrinsics` (dict) - Camera-specific intrinsic parameters

**`load_camparam(demo_path)`**
- **Purpose**: Loads camera parameters from demo data path
- **Input**: `demo_path` (str) - Demo data directory path
- **Output**: `(intrinsic, extrinsic)` - Intrinsic and extrinsic parameters

**`load_colmap_camparam(path)`**
- **Purpose**: Loads camera parameters from COLMAP reconstruction results
- **Input**: `path` (str) - COLMAP reconstruction path
- **Output**: `(intrinsics, extrinsics)` - Camera parameters
- **Dependencies**: Requires `pycolmap` library

#### Robot Data Loading

**`load_robot_traj(demo_path)`**
- **Purpose**: Loads robot trajectory data (arm + hand)
- **Input**: `demo_path` (str) - Demo data directory path
- **Output**: `robot_traj` (numpy.ndarray) - Concatenated robot trajectory

**`load_robot_traj_prev(demo_path)`**
- **Purpose**: Loads robot trajectory using previous version format
- **Input**: `demo_path` (str) - Demo data directory path
- **Output**: `arm_traj` (numpy.ndarray) - Arm trajectory only

**`load_robot_target_traj(demo_path)`**
- **Purpose**: Loads robot target trajectory (action) data
- **Input**: `demo_path` (str) - Demo data directory path
- **Output**: `robot_traj` (numpy.ndarray) - Target trajectory

**`load_contact_value(demo_path)`**
- **Purpose**: Loads contact value data
- **Input**: `demo_path` (str) - Demo data directory path
- **Output**: `contact_value` (numpy.ndarray) - Contact data

#### Other Data Loading

**`load_obj_traj(demo_path)`**
- **Purpose**: Loads object trajectory data
- **Input**: `demo_path` (str) - Demo data directory path
- **Output**: `obj_traj` - Object trajectory loaded from pickle file

**`load_mesh(obj_name)`**
- **Purpose**: Loads 3D mesh file
- **Input**: `obj_name` (str) - Object name
- **Output**: `mesh` (open3d.geometry.TriangleMesh) - 3D mesh object
- **Dependencies**: Requires `open3d` library

**`load_c2r(demo_path)`**
- **Purpose**: Loads camera-to-robot transformation matrix
- **Input**: `demo_path` (str) - Demo data directory path
- **Output**: `C2R` (numpy.ndarray) - Transformation matrix

#### Image Processing

**`is_image_file(file)`**
- **Purpose**: Checks if a file is an image file
- **Input**: `file` (str) - Filename
- **Output**: `bool` - Whether the file is an image
- **Supported formats**: `.png`, `.jpg`, `.jpeg`

**`load_images(dir)`**
- **Purpose**: Loads all image file paths from a directory
- **Input**: `dir` (str) - Image directory path
- **Output**: `list` - List of image file paths

---

## 2. keyboard_listener.py

**Purpose**: A simple listener module for asynchronous keyboard input handling using threading.

### Functions

**`run(event_dict)`**
- **Purpose**: Continuously monitors keyboard input and triggers corresponding events
- **Input**: `event_dict` (dict) - Mapping of keys to `threading.Event` objects
- **Output**: None (infinite loop)
- **Behavior**: Sets the corresponding event when an input key exists in `event_dict`

**`listen_keyboard(event_dict)`**
- **Purpose**: Starts the keyboard listener in a background thread
- **Input**: `event_dict` (dict) - Mapping of keys to `threading.Event` objects
- **Output**: None
- **Features**: Runs as a daemon thread, automatically terminates when main program exits