#!/usr/bin/env python3
"""
Generate ChArUco board as PDF for A4 printing using matplotlib
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import os
import datetime

def generate_charuco_board_pdf(board_number=1, marker_id_offset=0):
    """
    Generate a ChArUco board as PDF optimized for A4 paper printing
    
    Args:
        board_number: Board number (1-4)
        marker_id_offset: Starting marker ID for this board
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
    
    # CRITICAL: Manually create board with specific marker IDs
    # For 4x5 board, we have (4-1)*(5-1) = 12 markers
    num_markers = (squares_x - 1) * (squares_y - 1)
    
    # Create custom marker IDs for this board
    marker_ids = np.arange(marker_id_offset, marker_id_offset + num_markers, dtype=np.int32)
    
    print(f"  Marker IDs: {marker_id_offset} to {marker_id_offset + num_markers - 1}")
    
    # Generate each marker separately and compose the board
    square_size_px = int(square_length * pixels_per_mm)
    marker_size_px = int(marker_length * pixels_per_mm)
    margin_px = (square_size_px - marker_size_px) // 2
    
    # Create board image manually
    board_image = np.ones((board_height_px, board_width_px), dtype=np.uint8) * 255
    
    # Draw checkerboard pattern
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:  # White squares
                y_start = row * square_size_px
                y_end = (row + 1) * square_size_px
                x_start = col * square_size_px
                x_end = (col + 1) * square_size_px
                board_image[y_start:y_end, x_start:x_end] = 255
            else:  # Black squares
                y_start = row * square_size_px
                y_end = (row + 1) * square_size_px
                x_start = col * square_size_px
                x_end = (col + 1) * square_size_px
                board_image[y_start:y_end, x_start:x_end] = 0
    
    # Place ArUco markers at intersections
    marker_idx = 0
    for row in range(1, squares_y):
        for col in range(1, squares_x):
            # Generate marker with specific ID
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_ids[marker_idx], marker_size_px)
            
            # Calculate position (at intersection of squares)
            y_center = row * square_size_px
            x_center = col * square_size_px
            y_start = y_center - marker_size_px // 2
            x_start = x_center - marker_size_px // 2
            
            # Place marker
            board_image[y_start:y_start+marker_size_px, x_start:x_start+marker_size_px] = marker_img
            
            marker_idx += 1
    
    # Create output directory
    output_dir = f'outputs/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
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
    
    # Add title
    title_text = f"ChArUco Board #{board_number} | DICT_6X6_250 | IDs: {marker_id_offset}-{marker_id_offset+num_markers-1}"
    ax.text(a4_width_mm/2, a4_height_mm - 10, title_text,
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    plt.savefig(output_filename, format='pdf', dpi=300, 
                bbox_inches='tight', pad_inches=0)
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
        f.write(f"Marker IDs: {marker_id_offset} to {marker_id_offset + num_markers - 1}\n")
    
    return output_filename, num_markers

if __name__ == "__main__":
    print("="*70)
    print("Generating 4 ChArUco Boards with Non-Overlapping Marker IDs")
    print("="*70)
    
    num_boards = 4
    current_marker_id = 0
    
    for i in range(1, num_boards + 1):
        output_file, markers_used = generate_charuco_board_pdf(i, current_marker_id)
        current_marker_id += markers_used
    
    print("\n" + "="*70)
    print(f"✓ All {num_boards} boards generated successfully!")
    print(f"✓ Total markers used: {current_marker_id}")
    print(f"✓ Check the output directory for PDF files")
    print("="*70)