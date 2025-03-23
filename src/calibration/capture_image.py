import argparse
import os
import datetime
import paradex
from paradex.camera.camera_loader import CameraManager
from paradex.utils.io import home_dir, calib_path_list

def capture_images_from_all_cameras(save_path, lens_info_path, camera_config_path):
    """
    Captures a single image from all connected cameras and saves them.
    """
    # Iterate through interfaces and cameras
    camera_config = json.load(open(camera_config_path, "r"))
    lens_info = json.load(open(lens_info_path, "r"))

    # Initialize the Spinnaker system
    system = ps.System.GetInstance()

    ret = {}
    camera_list = system.GetCameras()
    for pCam in camera_list:
        cam = Camera(pCam, camera_config, lens_info, save_path, False)
        for frame_num in range(1):
            for _ in range(10):
                pImg, retcode = cam.get_capture(200) # 200 ms timeout
                if retcode:
                    cvImg = spin2cv(pImg, 1536, 2048)  # Adjust resolution as needed
                    image_save_path = os.path.join(save_path , f"{cam.serialnum}.png") 
                    cv2.imwrite(str(image_save_path), cvImg)
                    print(f"Image saved at: {image_save_path}")
                    cam.stop_camera()
                    ret[cam.serialnum] = cvImg
                    break

                else:
                    print(f"Failed to capture image from camera {cam.serialnum}")

    for pCam in camera_list:
        del pCam

    camera_list.Clear()

    system.ReleaseInstance()

    print("Image capture completed for all cameras.")
    return ret

def find_latest_directory(is_init):
    if is_init:
        now = datetime.datetime.now()
        dir_name = now.strftime("%Y%m%d%H%M%S")
        
        return dir_name, 0
    else:
        # Get list of directories in the current working directory
        dirs = [d for d in os.listdir(calib_path_list[0])] 
        print(os.listdir(calib_path_list[0]), dirs)   
        if not dirs:
            print("No valid directories found.")
            return
        
        # Sort directories based on name (assuming names are time-based)
        latest_dir = max(dirs, key=str)
        index_list = [int(d) for d in os.listdir(os.path.join(calib_path_list[0], latest_dir))]

        return latest_dir, max(index_list)+1 if index_list else 0

def main():
    parser = argparse.ArgumentParser(description="Manage timestamped directories.")
    parser.add_argument("--init", action="store_true", help="Initialize a new directory with current timestamp.")
    
    args = parser.parse_args()
    is_init = args.init
    name, index = find_latest_directory(is_init)

    for calib_path in calib_path_list:
        os.makedirs(os.path.join(calib_path, str(name), str(index), "images","00001"), exist_ok=True)
        
    save_path = f"calibration/{name}/{index}/images/00001"
    camera_config = "config/lens.json"
    lens_info = "config/camera.json"
    capture_images_from_all_cameras(save_path, 4, lens_info, camera_config)


if __name__ == "__main__":
    main()
