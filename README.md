0. run autoforce_ip.py whenever you restart the capturePC

Adjust Exposure in config/lens_info.json

1. calibration

Capture images from 16 cameras and save it under directory {root_dir} / frames

python -m src.calibration.calibration.py --root_dir {root_dir}

2. capture
    capture single image
    python -m src.capture.image.py --save_folder {folder name}

    capture synchronized video
    video is saved to ~/captures1

    python -m src.capture.video 

3. realtime hand detection
    python -m src.process.keypoint_detector
