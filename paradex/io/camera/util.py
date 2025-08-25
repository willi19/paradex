import os

def get_image():
    image_path = f'shared_data/inference/obj_6D/image'
    scene_path = f'{shared_dir}/inference/obj_6D/'
    if os.path.exists(os.path.join(shared_dir, "inference", "obj_6D")):
        shutil.rmtree(os.path.join(shared_dir, "inference", "obj_6D"))
        os.makedirs(os.path.join(shared_dir, "inference", "obj_6D", "image"))
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())

    run_script(f"python src/capture/camera/image_client.py", pc_list)

    camera_loader = RemoteCameraController("image", None)
    camera_loader.start(image_path)
    camera_loader.end()
    camera_loader.quit()
    image_list = os.listdir(os.path.join(home_path, image_path))

    img_dict = {}
    for img_name in image_list:
        img_dict[img_name.split(".")[0]] = cv2.imread(os.path.join(home_path, image_path, img_name))
    
    return img_dict