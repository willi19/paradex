import os

from paradex.utils.file_io import shared_dir
from paradex.utils.upload_file import copy_file
from paradex.video.convert_codec import change_to_h264

for obj_name in os.listdir("capture/lookup"):
    for index in os.listdir(f"capture/lookup/{obj_name}"):
        for type in ["pick", "place"]:
            change_to_h264(f"capture/lookup/{obj_name}/{index}/{type}/default.mp4", f"capture/lookup/{obj_name}/{index}/{type}.mp4")
            copy_file(f"capture/lookup/{obj_name}/{index}/{type}.mp4", os.path.join(shared_dir, f"capture/lookup/{obj_name}/{index}/{type}.mp4"))