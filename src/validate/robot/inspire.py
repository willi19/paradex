import time
import numpy as np
import matplotlib.pyplot as plt

from paradex.utils.system import network_info
from paradex.io.robot_controller.inspire_controller import InspireController, SENSOR_LAYOUT

ic = InspireController(**network_info['inspire_usb']['param'])
ic.start("inspire_motion")
# start_time = time.time()    
# while time.time() - start_time < 10.0:
#     ic.move(np.zeros(6) + 500 + 100 * np.sin((time.time() - start_time) * 2 * np.pi * 0.2))
#     time.sleep(0.01)
# ic.end()

ic.end()

# plt.ion()  # 인터랙티브 모드
# fig, axs = plt.subplots(5, 4, figsize=(16, 20))
# axs = axs.flatten()
# im_list = []

# # 초기 히트맵 생성
# for i, name in enumerate(SENSOR_LAYOUT.keys()):
#     dummy = np.zeros((SENSOR_LAYOUT[name]["rows"], SENSOR_LAYOUT[name]["cols"]))
#     im = axs[i].imshow(dummy, cmap='viridis', vmin=0, vmax=4095)
#     axs[i].set_title(name)
#     plt.colorbar(im, ax=axs[i], shrink=0.8)
#     im_list.append(im)

# plt.tight_layout()
# while True:
#     data = ic.get_data()
#     tactile_data = data['tactile_data']
    
#     for i, name in enumerate(SENSOR_LAYOUT.keys()):
#         sensor_values = tactile_data[name]
#         rows = SENSOR_LAYOUT[name]["rows"]
#         cols = SENSOR_LAYOUT[name]["cols"]
#         reshaped_data = sensor_values.reshape((rows, cols))
#         im_list[i].set_data(reshaped_data)
#         im_list[i].set_clim(vmin=np.min(sensor_values), vmax=np.max(sensor_values))
    
#     plt.pause(0.1)