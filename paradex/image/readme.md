
# aruco.py
### ArUco / ChArUco Detection Utilities

Utility functions and helper data structures for detecting **ArUco markers** and **ChArUco boards** with OpenCV `cv2.aruco`.

---

#### ✨ What’s Inside

| Item | Description |
|------|-------------|
| `aruco_type` | List of all supported predefined ArUco dictionary names. |
| `aruco_dict` | Mapping from dictionary name → `cv2.aruco_Dictionary` instance. |
| `arucoDetector_dict` | Mapping from dictionary name → `cv2.aruco_ArucoDetector` instance. |
| `detect_aruco(img, dict_type='6X6_1000')` | Detects ArUco markers in an image; returns `(corners, IDs)`. |
| `check_boardinfo_valid(boardinfo)` | Validates that a ChArUco board descriptor contains all required keys. |
| `detect_charuco(img, boardinfo)` | Detects ChArUco corners for **multiple boards**; returns a per‑board detection dict. |
| `merge_charuco_detection(detection_list, boardinfo)` | Merges per‑board ChArUco detections into global ID space. |

