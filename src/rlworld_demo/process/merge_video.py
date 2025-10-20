import os
from paradex.image.merge import merge_image

for index in range(2):
    video_dir = os.path.join("RLWRLD_DEMO_FINAL", str(index), "multiview")
    video_list = os.listdir(video_dir)
    video_dict = {}
    for video_name in video_list:
        if video_name == "cam_param":
            continue
        video_path = os.path.join(video_dir, video_name)
        video_dict[video_name] = video_path

    
        save_path = os.path.join("RLWRLD_DEMO_FINAL", str(index), f"merged_{video_name[:-8]}.mp4")
        merge_image(video_path, save_path, 3, 1, fps=30)
    