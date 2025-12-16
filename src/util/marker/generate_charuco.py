#!/usr/bin/env python3
"""
Generate ChArUco board as PDF for A4 printing using matplotlib
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

def generate_charuco_board_pdf(board_number=1, marker_id_offset=0):
    """
    Generate a ChArUco board as PDF optimized for A4 paper printing
    
    Args:
        board_number: Board number (1-4)
        marker_id_offset: Starting marker ID offset for this board
    """
    # ChArUco board parameters
    squares_x = 4  # number of squares in X direction
    squares_y = 5  # number of squares in Y direction
    square_length = 50  # in mm
    marker_length = 40  # in mm (should be smaller than square_length)
    
    # A4 dimensions in mm
    a4_width_mm = 210
    a4_height_mm = 297
    
    # Calculate board dimensions
    board_width_mm = squares_x * square_length
    board_height_mm = squares_y * square_length
    
    print(f"\nBoard {board_number}:")
    print(f"  Board dimensions: {board_width_mm}mm x {board_height_mm}mm")
    print(f"  A4 paper: {a4_width_mm}mm x {a4_height_mm}mm")
    
    # Use high DPI for image generation
    dpi = 300
    pixels_per_mm = dpi / 25.4
    
    # Calculate image size in pixels
    board_width_px = int(board_width_mm * pixels_per_mm)
    board_height_px = int(board_height_mm * pixels_per_mm)
    
    print(f"  Board size in pixels: {board_width_px} x {board_height_px}")
    
    # Create ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Calculate square and marker sizes in pixels
    square_size_px = int(square_length * pixels_per_mm)
    marker_size_px = int(marker_length * pixels_per_mm)
    
    # ChArUco board: markers go in BLACK squares only
    # Count black squares and assign marker IDs
    marker_positions = []  # (row, col, marker_id)
    marker_id = marker_id_offset
    
    for row in range(squares_y):
        for col in range(squares_x):
            # Black square if (row + col) is odd
            if (row + col) % 2 == 0:
                marker_positions.append((row, col, marker_id))
                marker_id += 1
    
    num_markers = len(marker_positions)
    print(f"  Marker IDs: {marker_id_offset} to {marker_id - 1}")
    print(f"  Total markers: {num_markers}")
    
    # Create board image manually
    board_image = np.ones((board_height_px, board_width_px), dtype=np.uint8) * 255
    
    # Draw checkerboard pattern
    for row in range(squares_y):
        for col in range(squares_x):
            y_start = row * square_size_px
            y_end = (row + 1) * square_size_px
            x_start = col * square_size_px
            x_end = (col + 1) * square_size_px
            
            if (row + col) % 2 == 0:  # White squares
                board_image[y_start:y_end, x_start:x_end] = 255
            else:  # Black squares
                board_image[y_start:y_end, x_start:x_end] = 0
    
    # Place ArUco markers in BLACK squares
    for row, col, mid in marker_positions:
        # Generate marker with specific ID
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_size_px)
        
        # Calculate position (centered in the black square)
        y_start = row * square_size_px + (square_size_px - marker_size_px) // 2
        x_start = col * square_size_px + (square_size_px - marker_size_px) // 2
        
        # Place marker in black square
        board_image[y_start:y_start+marker_size_px, x_start:x_start+marker_size_px] = marker_img
    
    # Note: OpenCV's ChArUco board generation uses sequential IDs starting from 0
    # We can't easily change them without reconstructing the entire board manually
    # So we'll note the ID offset in the PDF text
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/charuco/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.join(output_dir, f"charuco_board_{board_number}.pdf")
    
    # A4 size in inches (for matplotlib)
    a4_width_inch = a4_width_mm / 25.4
    a4_height_inch = a4_height_mm / 25.4
    
    fig = plt.figure(figsize=(a4_width_inch, a4_height_inch), dpi=300)
    
    # Create axes for the board
    ax = fig.add_axes([0, 0, 1, 1])  # Full figure
    ax.set_xlim(0, a4_width_mm)
    ax.set_ylim(0, a4_height_mm)
    ax.axis('off')
    
    # Calculate position to center the board
    x_offset = (a4_width_mm - board_width_mm) / 2
    y_offset = (a4_height_mm - board_height_mm) / 2
    
    # Display the board image
    extent = [x_offset, x_offset + board_width_mm, 
              y_offset, y_offset + board_height_mm]
    ax.imshow(board_image, cmap='gray', extent=extent, origin='lower')
    
    # Add title with actual marker ID information
    plt.savefig(output_filename, format='pdf', dpi=300, 
                bbox_inches='tight', pad_inches=0)
    plt.savefig("test.png")
    plt.close()
    
    print(f"  ✓ Saved to: {output_filename}")
    
    # Also save parameters file for this board
    params_filename = os.path.join(output_dir, f"charuco_board_{board_number}_parameters.txt")
    with open(params_filename, 'w') as f:
        f.write(f"ChArUco Board #{board_number} Parameters\n")
        f.write(f"=========================================\n")
        f.write(f"Dictionary: DICT_6X6_250\n")
        f.write(f"Squares X: {squares_x}\n")
        f.write(f"Squares Y: {squares_y}\n")
        f.write(f"Square Length: {square_length} mm\n")
        f.write(f"Marker Length: {marker_length} mm\n")
        f.write(f"Board Size: {board_width_mm}mm x {board_height_mm}mm\n")
        f.write(f"Number of markers: {num_markers}\n")
        f.write(f"Marker ID range: {marker_id_offset} to {marker_id_offset + num_markers - 1}\n")
        f.write(f"\nNOTE: This board should be detected with marker ID offset = {marker_id_offset}\n")
        f.write(f"When using for calibration, configure your detection to use offset {marker_id_offset}\n")
    
    # Copy to outputs for easy access
    output_copy = f"outputs/charuco_board_{board_number}_ids_{marker_id_offset}-{marker_id_offset+num_markers-1}.pdf"
    import shutil
    shutil.copy(output_filename, output_copy)
    print(f"  ✓ Copied to: {output_copy}")
    
    return output_filename, num_markers

if __name__ == "__main__":
    print("="*70)
    print("Generating 4 ChArUco Boards with Non-Overlapping Marker IDs")
    print("="*70)
    
    num_boards = 1
    current_marker_id = 0
    
    for i in range(1, num_boards + 1):
        output_file, markers_used = generate_charuco_board_pdf(i, current_marker_id)
        current_marker_id += markers_used
    
    print("\n" + "="*70)
    print(f"✓ All {num_boards} boards generated successfully!")
    print(f"✓ Total markers used: {current_marker_id}")
    print(f"✓ Files saved to outputs/")
    print("="*70)
    print("\nIMPORTANT NOTE:")
    print("OpenCV's ChArUco board uses sequential IDs starting from 0.")
    print("To use these boards without overlap, you need to:")
    print("1. Use Board 1 as-is (IDs 0-9)")
    print("2. For Boards 2-4, you'll need to offset detection IDs in your code")
    print("   OR print different sized boards that naturally use different ID ranges")
    print("="*70)