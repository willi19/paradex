import math
import subprocess
import cv2
import numpy as np

_screen_size = None


def get_screen_size(default=(1920, 1080)):
    """Return ``(width, height)`` of the primary display, cached after the first call.

    Tries Tk (stdlib, works under X11 and Wayland's Xwayland), then ``xrandr``,
    then falls back to ``default`` on a headless box.
    """
    global _screen_size
    if _screen_size is not None:
        return _screen_size

    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        if size[0] > 0 and size[1] > 0:
            _screen_size = size
            return _screen_size
    except Exception:
        pass

    try:
        out = subprocess.run(["xrandr"], capture_output=True, text=True, timeout=2).stdout
        for line in out.splitlines():
            if " connected" in line and "x" in line:
                for tok in line.split():
                    if "x" in tok and tok.split("x")[0].isdigit():
                        w, h = tok.split("+")[0].split("x")
                        _screen_size = (int(w), int(h))
                        return _screen_size
    except Exception:
        pass

    _screen_size = default
    return _screen_size


def fit_to_screen(image, margin=0.95, screen=None):
    """Scale ``image`` to the largest size that still fits the screen, aspect kept.

    The limiting dimension wins (letterboxed, never stretched), so a wide grid is
    bounded by screen width and a tall one by screen height.
    """
    sw, sh = screen or get_screen_size()
    h, w = image.shape[:2]
    scale = min(sw * margin / w, sh * margin / h)
    if abs(scale - 1.0) < 0.02:
        return image
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interp)

def get_optimal_font_scale(text, target_width, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2):
    """텍스트가 target_width에 맞는 최적의 font scale 찾기.

    Small tiles could not fit their caption even at the old 0.3 floor, so the text
    ran off the edge. Go down to 0.12 before giving up.
    """
    for scale in np.arange(0.3, 3.0, 0.1):
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        if text_size[0] > target_width:
            fitted = max(0.12, scale - 0.1)
            # Shrink past the floor when even the smallest step overflows.
            while fitted > 0.12 and cv2.getTextSize(text, font, fitted, thickness)[0][0] > target_width:
                fitted -= 0.04
            return round(fitted, 2)
    return 3.0

def merge_image(image_dict, image_text={}, put_text=True, canvas_width=2048):
    """Tile a multi-camera image dict into one labeled near-square grid.

    Cameras are laid out in sorted-serial order; each tile is captioned with its
    serial (plus any extra text). Assumes ~2048x1536 source frames (the canvas size
    is hardcoded); intended for live previews and sync checks, not archival output.

    Parameters
    ----------
    image_dict : dict
        ``{serial: image}``.
    image_text : dict, optional
        ``{serial: str}`` extra caption appended after the serial.
    put_text : bool
        Draw the serial/caption overlay when True.
    canvas_width : int
        Target width of the whole grid in pixels. Tiles are never upscaled past
        their source, so raising this only helps when the incoming previews are
        themselves bigger (see the stream downscale factor on the capture-PC side).

    Returns
    -------
    numpy.ndarray
        The composited grid image.
    """
    name_list = sorted(list(image_dict.keys()))
    num_images = len(name_list)

    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)
    border_px = 10

    # Keep the source aspect ratio: derive the tile height from the tile width using
    # the actual frame shape, instead of dividing W and H independently (which squashed
    # a 4:3 frame into a near-square tile). Total canvas width stays ~2048.
    src_h, src_w = image_dict[name_list[0]].shape[:2]
    # Don't upscale past the source: preview streams arrive already downscaled (/8),
    # and blowing them back up to a fixed 2048/cols only adds blur.
    new_W = min(canvas_width // grid_cols, src_w)
    new_H = max(1, round(new_W * src_h / src_w))

    canvas_h = new_H * grid_rows + border_px * (grid_rows - 1)
    canvas_w = new_W * grid_cols + border_px * (grid_cols - 1)
    grid_image = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for idx, img_name in enumerate(name_list):
        img = image_dict[img_name].copy()

        if put_text:
            # Build the WHOLE caption before sizing it. Fitting the scale to the
            # serial alone and then appending the frame counter is what pushed the
            # text past the tile edge.
            txt = f"{img_name}"
            if img_name in image_text:
                txt += f" {image_text[img_name]}"
            target_width = int(img.shape[1] * 0.92) - 20
            thickness = max(1, img.shape[1] // 500)
            font_scale = get_optimal_font_scale(txt, target_width, thickness=thickness)
            text_x = 10
            text_y = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                     font_scale, thickness)[0][1] + 10

            # Outlined text (dark border + bright fill) instead of a filled black box:
            # readable over any background without covering the image behind it.
            cv2.putText(img, txt, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(img, txt, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 0), thickness, cv2.LINE_AA)

        resized_img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)

        r_idx = idx // grid_cols
        c_idx = idx % grid_cols
        r_start = r_idx * (new_H + border_px)
        c_start = c_idx * (new_W + border_px)
        grid_image[r_start:r_start+new_H, c_start:c_start+new_W] = resized_img

    return grid_image