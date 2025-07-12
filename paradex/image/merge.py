def merge_image():
    for idx, serial_num in enumerate(serial_list):
                img = saved_corner_img[serial_num].copy()
                corners, ids, frame = cur_state[serial_num]
                if corners.shape[0] > 0:
                    draw_charuco(img, corners, BOARD_COLORS[1], 5, -1, ids)
                img = cv2.putText(img, f"{serial_num} {frame}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 0), 12)

                resized_img = cv2.resize(img, (new_W, new_H))
                
                r_idx = idx // grid_cols
                c_idx = idx % grid_cols

                r_start = r_idx * (new_H + border_px)
                c_start = c_idx * (new_W + border_px)
                grid_image[r_start:r_start+resized_img.shape[0], c_start:c_start+resized_img.shape[1]] = resized_img
