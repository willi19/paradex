from paradex.io.capture_pc.data_sender import DataCollector
from paradex.utils.env import get_pcinfo
import time

dc = DataCollector(
    pc_info=get_pcinfo(),
    port=5500
)

dc.start()

while True:
    data = dc.get_data()
    print(data)
    time.sleep(1)
