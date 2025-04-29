import datetime
import time
from dex_robot.io.contact.receiver import SerialReader
from dex_robot.io.contact.process import process_contact

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reader = SerialReader(save_path="test_capture/mingi")

    start_time = time.time()
    while time.time() - start_time < 15:
        print(time.time() - start_time)
        time.sleep(1)  # Keep main process alive

    
    print("\n[INFO] Exiting...")
    reader.quit()  # Stop the serial reader process safely

    # data_path = "/home/temp_id/shared_data/capture/pringles/0/contact"
    # value = np.load(f"{data_path}/data.npy")
    # value = process_contact(value)
    # timestamp = np.load(f"{data_path}/timestamp.npy")
    # timestamp = timestamp - timestamp[0]

    data_path = "test_capture/mingi/contact"
    value = np.load(f"{data_path}/data.npy")
    # value = value - value[0]
    # value = process_contact(value)
    timestamp = np.load(f"{data_path}/timestamp.npy")
    timestamp = timestamp - timestamp[0]

    sensor_value = np.array([1242.82021467, 3066.5509839,  2484.28175313, 2916.64847943, 1233.42218247,
 2101.0411449,  2689.30858676, 1165.40429338, 2014.72093023, 2702.45706619,
 1261.82110912, 2031.990161,   4613.33005367, 7319.88461538, 4359.25313059])

    value = value - sensor_value
    # for i in range(15):
    #     plt.plot(timestamp, value[:,i:i+1])# - value[0])
    #     plt.legend(["sensor" + str(j) for j in range(i,i+1)])
    #     plt.show()
    # value = np.clip(value, 0, 30)
    # plt.plot(timestamp[1000:2000], value[1000:2000])
    # plt.legend(["sensor" + str(i) for i in range(15)])
    plt.plot(timestamp, value)
    plt.legend(["sensor" + str(i) for i in range(15)])
    plt.show()

    # print(np.std(value, axis=0))
    print(np.mean(value, axis=0))

    print(np.max(value, axis=0))
    print(np.min(value, axis=0))
    print(np.std(value, axis=0))
    print(value[0])