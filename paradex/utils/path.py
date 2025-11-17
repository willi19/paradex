import os

home_path = os.path.expanduser("~")
pc_name = os.path.basename(home_path)
shared_dir = os.path.join(home_path, "shared_data")
capture_path_list = [os.path.join(home_path, f"captures{i}") for i in range(1,3)]

download_dir = os.path.join(home_path, "download")
model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "model")

rsc_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "rsc",
)