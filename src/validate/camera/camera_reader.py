import cv2

from paradex.io.camera_system.camera_reader import CameraReader, MultiCameraReader
from paradex.utils.env import get_serial_list

serial_list = get_serial_list()
for serial in serial_list:
    cr = CameraReader(serial)
    frame = cr.read_frame()
    cr.release()

mcr = MultiCameraReader(serial_list)
frames = mcr.read_frames()
mcr.release()    