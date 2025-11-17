import time

from paradex.io.capture_pc.data_sender import DataPublisher

dp = DataPublisher(name="TestPublisher")

start_time = time.time()
while True:
    dp.send_data({"value": time.time() - start_time})
    time.sleep(0.5)