import serial
import time
from multiprocessing import shared_memory, Lock, Value, Event, Process
import numpy as np
import os
import datetime

class SerialReader():
    def __init__(self, save_path, port='/dev/ttyACM0', baudrate=115200, timeout=1, freq=500):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.freq = freq  # Frequency for reading (100 Hz means 1/100s delay)
        
        self.arduino = None


        self.data = np.zeros((60000, 15), dtype=np.float64)
        self.timestamp = np.zeros((60000, 1), dtype=np.float64)
        self.log = []

        self.cnt = 0
        
        self.exit = Event()
        self.capture_path = os.path.join(save_path,"contact")

        self.recv_process = Process(target=self.run)
        self.recv_process.start()
        
    def run(self):
        """Runs the serial reader process."""
        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.arduino.reset_input_buffer()
            
            # Wait for stable readings
            tmp = 0

            while not self.exit.is_set():
                if self.arduino.readable():
                    raw_data = self.arduino.read_until(b'\n')
                    recv_time = time.time()
                    if tmp < 500:
                        tmp += 1
                        continue
                    
                    try:
                        decoded_data = raw_data.decode().strip()
                        # print(np.array([float(x) for x in decoded_data.split(" ")]))
                        data = np.array([float(x) for x in decoded_data.split(" ")])
                        if len(data) != 15:
                            self.log.append(f"[WARNING] Invalid data length: {len(data)}")
                            continue

                        self.data[self.cnt] = np.array([float(x) for x in decoded_data.split(" ")])
                        self.timestamp[self.cnt] = recv_time
                        self.cnt += 1
                        
                    except Exception as e:
                        self.log.append(f"[ERROR] Data decoding error: {e}")

                time.sleep(1 / self.freq)  # Control read rate

        except Exception as e:
            self.log.append(f"[ERROR] Serial error: {e}")
        finally:
            if self.arduino:
                self.arduino.close()
            print("[INFO] Serial reader stopped.")

        os.makedirs(self.capture_path, exist_ok=True)
        print(f"[INFO] Saving data to {self.capture_path}...")
        np.save(os.path.join(self.capture_path, f"data.npy"), self.data[:self.cnt])
        np.save(os.path.join(self.capture_path, f"timestamp.npy"), self.timestamp[:self.cnt])
        
        with open(os.path.join(self.capture_path, f"log.txt"), "w") as f:
            for line in self.log:
                f.write(line + "\n")

    def quit(self):
        """Stops the serial reader process."""
        self.exit.set()
        self.recv_process.join()


# ======= Main Process =======
if __name__ == "__main__":
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reader = SerialReader(capture_path="test_capture/mingi",date_str=date_str)

    start_time = time.time()
    while time.time() - start_time < 1:
        print(time.time() - start_time)
        time.sleep(0.01)  # Keep main process alive

    
    print("\n[INFO] Exiting...")
    reader.quit()  # Stop the serial reader process safely
