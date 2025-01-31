import cv2
import os
import glob
import numpy as np

def play_videos_in_grid(directory, grid_size=(4, 4)):
    # Get all AVI files in the directory
    video_files = glob.glob(os.path.join(directory, "*.avi"))
    
    if not video_files:
        print("No .avi files found in the directory.")
        return
    
    # Limit videos to the number of grid cells
    max_videos = grid_size[0] * grid_size[1]
    video_files = video_files[:max_videos]

    # Open video captures for each file
    captures = [cv2.VideoCapture(file) for file in video_files]

    # Set up window size for grid display
    window_width, window_height = 1280, 960  # Set the grid window size
    cell_width = window_width // grid_size[1]
    cell_height = window_height // grid_size[0]

    while True:
        grid_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        frames = []

        for i, cap in enumerate(captures):
            ret, frame = cap.read()
            if ret:
                # Resize frame to fit in grid cell
                frame = cv2.resize(frame, (cell_width, cell_height))
            else:
                # Release capture if video ends
                frame = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            frames.append(frame)

        # Add frames to grid
        for idx, frame in enumerate(frames):
            row, col = divmod(idx, grid_size[1])
            y1, y2 = row * cell_height, (row + 1) * cell_height
            x1, x2 = col * cell_width, (col + 1) * cell_width
            grid_frame[y1:y2, x1:x2] = frame

        # Display the grid
        cv2.imshow("Video Grid", grid_frame)

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all captures and close windows
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    directory = "/home/capture18/captures1/hyunsoo"
    play_videos_in_grid(directory)
