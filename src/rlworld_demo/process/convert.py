import os

from paradex.video.convert_codec import change_to_h264

for index in ["1", "0"]:
    video_dir = os.path.join("RLWRLD_DEMO_FINAL", str(index), "multiview")
    video_list = os.listdir(video_dir)

    for video_name in ["iphone1_tmp.mp4"]:#, "grid_view_tmp.avi", "iphone1_tmp.mp4"]:
        video_path = os.path.join("RLWRLD_DEMO_FINAL", str(index), video_name)
        save_path = os.path.join("RLWRLD_DEMO_FINAL", str(index), f"{video_name[:-8]}.mp4")
        change_to_h264(video_path, save_path)

    # for video_name in video_list:
    #     video_path = os.path.join(video_dir, video_name)
    #     save_path = os.path.join("RLWRLD_DEMO_FINAL", str(index), "multiview", f"{video_name[:-8]}.mp4")
    #     change_to_h264(video_path, save_path)
    ffmpeg -i input.mp4 -filter:v "setpts=0.5*PTS" -an output_2x.mp4

# 10배속
ffmpeg -i RLWRLD_DEMO_FINAL/1/iphone1.mp4 -filter:v "setpts=0.5*PTS" -an RLWRLD_DEMO_FINAL/1/iphone1_2x.mp4