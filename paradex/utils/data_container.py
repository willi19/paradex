

def is_image_file(file):
    return file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")

def load_images(dir):
    return [os.path.join(dir, f) for f in os.listdir(dir) if is_image_file(f)]
