ffmpeg -i ik_vis/0/default.mp4 -vcodec libx264 -crf 23 -preset fast -acodec aac -b:a 192k 0.mp4
ffmpeg -i ik_vis/2/default.mp4 -vcodec libx264 -crf 23 -preset fast -acodec aac -b:a 192k 2.mp4
ffmpeg -i ik_vis/3/default.mp4 -vcodec libx264 -crf 23 -preset fast -acodec aac -b:a 192k 3.mp4
ffmpeg -i ik_vis/4/default.mp4 -vcodec libx264 -crf 23 -preset fast -acodec aac -b:a 192k 4.mp4