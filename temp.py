import os
from PIL import Image

# =========================
# Config
# =========================
base_path = "/home/temp_id/shared_data/capture/hri_inspire_left/"
subdir = os.path.join("video_extracted", "23280286")

target_frames = ["00270.jpg","00180.jpg", "00120.jpg", "00090.jpg", "00060.jpg"]  # 우선순위

exclude_objects = {
    "sync", "sync2",
    "hovering", "hovering2", "hovering3", "hovering4", "hovering5",
    "hovering0123",
    "0121_test", "0121_hovering", "mimic_calib"
}

grid_cols = 7
padding = 10
bg_color = (255, 255, 255)

# =========================
# Collect images (ONLY folder "0")
# =========================
image_paths = []

for obj_name in sorted(os.listdir(base_path)):
    obj_path = os.path.join(base_path, obj_name)
    if not os.path.isdir(obj_path):
        continue
    if obj_name in exclude_objects:
        continue

    num_path = os.path.join(obj_path, "2")
    if not os.path.isdir(num_path):
        continue

    found = False
    for frame in target_frames:
        img_path = os.path.join(num_path, subdir, frame)
        if os.path.isfile(img_path):
            image_paths.append(img_path)
            found = True
            break  # 180 있으면 120 안 봄

    if not found:
        print(f"[WARN] No frame for {obj_name}")

print(f"Found {len(image_paths)} images")

if len(image_paths) == 0:
    raise RuntimeError("No images found")

# =========================
# Load images
# =========================
images = [Image.open(p).convert("RGB") for p in image_paths]

# Resize all to same size
w, h = images[0].size
images = [img.resize((w, h), Image.BILINEAR) for img in images]

# =========================
# Create grid
# =========================
rows = (len(images) + grid_cols - 1) // grid_cols

grid_w = grid_cols * w + (grid_cols - 1) * padding
grid_h = rows * h + (rows - 1) * padding

grid_img = Image.new("RGB", (grid_w, grid_h), bg_color)

for idx, img in enumerate(images):
    r = idx // grid_cols
    c = idx % grid_cols

    x = c * (w + padding)
    y = r * (h + padding)

    grid_img.paste(img, (x, y))

# =========================
# Save
# =========================
out_path = os.path.join(base_path, "grid_180_or_120.jpg")
grid_img.save(out_path, quality=95)

print(f"Saved grid image to: {out_path}")