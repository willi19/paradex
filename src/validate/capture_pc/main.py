from paradex.io.capture_pc.data_sender import DataCollector
from paradex.utils.env import get_pcinfo
import time

dc = DataCollector(
    pc_info=get_pcinfo(),
    port=5500
)

dc.start()

while True:
    data_dict = dc.get_data()
    for pc_id, data in data_dict.items():
        print(f"PC ID: {pc_id}, Data: {data}")
    time.sleep(1)
