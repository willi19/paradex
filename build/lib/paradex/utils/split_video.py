import os
import subprocess

def split_video_to_images(video_path: str, output_dir: str, image_format: str = "jpg", frame_rate: int = 1):
    """
    Splits a video into images using ffmpeg.

    :param video_path: Path to the input video file.
    :param output_dir: Directory where extracted frames will be saved.
    :param image_format: Image format (e.g., jpg, png, bmp).
    :param frame_rate: Number of frames per second to extract.
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the ffmpeg command
    output_pattern = os.path.join(output_dir, f"frame_%04d.{image_format}")
    command = [
        "ffmpeg", "-i", video_path, "-vf", f"fps={frame_rate}", output_pattern
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Frames extracted successfully.")
    except subprocess.CalledProcessError as e:
        print("Error occurred while extracting frames:", e.stderr.decode())

# Example usage
if __name__ == "__main__":
    video_file = "/home/capture18/captures1/calib_0227_9/23180202_20250227_160225-0000.avi"  # Change this to your video file path
    output_directory = "output_frames"  # Change this to your desired output directory
    split_video_to_images(video_file, output_directory, image_format="png", frame_rate=1)
