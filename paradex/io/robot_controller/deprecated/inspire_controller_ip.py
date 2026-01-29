import time
from pymodbus.client import ModbusTcpClient  # pip3 install pymodbus==2.5.3
import time

import numpy as np
from threading import Thread, Event, Lock
#  interface Modbus TCP + CAN2.0 or Modbus TCP + RS485
import os
import json
from paradex.utils.system import config_dir

action_dof = 6
regdict = {
    'ID': 1000,
    'baudrate': 1002,
    'clearErr': 1004,
    'save': 1005,
    'forceClb': 1009,
    'posSet': 1474,
    'angleSet': 1486,
    'forceSet': 1498,
    'speedSet': 1522,
    'angleAct': 1546,
    'forceAct': 1582,
    'errCode': 1606,
    'statusCode': 1612,
    'temp': 1618,
    'actionSeq': 2320,
    'actionRun': 2322
}

tactile = {
    'little':3000,
    'ring': 3370,
    'middle':3740,
    'fore': 4110,
    'thumb': 4480,
    'palm': 4900
}

TACTILE_LAYOUT = {
    "little_tip":    (3000, 3, 3),
    "little_nail":   (3018, 12, 8),
    "little_pad":    (3210, 10, 8),
    "ring_tip":      (3370, 3, 3),
    "ring_nail":     (3388, 12, 8),
    "ring_pad":      (3580, 10, 8),
    "middle_tip":    (3740, 3, 3),
    "middle_nail":   (3758, 12, 8),
    "middle_pad":    (3950, 10, 8),
    "index_tip":     (4110, 3, 3),
    "index_nail":    (4128, 12, 8),
    "index_pad":     (4320, 10, 8),
    "thumb_tip":     (4480, 3, 3),
    "thumb_nail":    (4498, 12, 8),
    "thumb_middle":  (4690, 3, 3),
    "thumb_pad":     (4708, 12, 8),
    "palm":          (4900, 8, 14),
}

class InspireControllerIP:
    def __init__(self, ip, port, tactile=False):
        network_config = json.load(open(os.path.join(config_dir, "network.json"), "r"))
        # self.ip = network_config["inspire_ip"]["param"]["ip"]
        # self.port = network_config["inspire_ip"]["param"]["port"]
        
        self.save_event = Event()
        self.exit_event = Event()
        self.connection_event = Event()
        
        self.ip = ip
        self.port = port

        self.capture_path = None
        self.tactile = tactile
        self.tactile_index, self.tactile_dim = self.build_tactile_index()
        self.latest_tactile = None
        self.latest_tactile_time = None
        
        self.home_pose = np.zeros(action_dof)+800
        
        self.lock = Lock()
        self.target_action = np.zeros(action_dof)+800
        
        self.open_modbus()
        
        self.write6('speedSet', [1000, 1000, 1000, 1000, 1000, 1000])
        self.write6('forceSet', [400, 400, 400, 400, 400, 400])
        self.write6('angleSet', [1000, 1000, 1000, 1000, 1000, 1000])
        
        self.thread = Thread(target=self.move_hand)
        self.thread.daemon = True
        self.thread.start()

    def build_tactile_index(self):
        index = {}
        offset = 0
        for name, (addr, rows, cols) in TACTILE_LAYOUT.items():
            size = rows * cols
            index[name] = (offset, rows, cols)
            offset += size
        return index, offset

    def start(self, save_path=None):
        with self.lock:
            self.capture_path=save_path
            self.data = {
                "time":[],#np.zeros((T,1), dtype=np.float64),
                "position":[],#np.zeros((T, action_dof), dtype=np.float64),
                "action":[],#np.zeros((T, action_dof), dtype=np.float64),
                "force":[],#np.zeros((T, action_dof), dtype=np.float64)
                "tactile":[]
            }
            self.save_event.set()
            
    
                    
    def set_homepose(self, home_pose):
        assert home_pose.shape == (action_dof,)
        self.home_pose = home_pose.copy()

    def home_robot(self, home_pose=None):
        if home_pose is None:
            self.target_action = self.home_pose.copy()
        else:
            self.home_pose = home_pose.copy()
            self.target_action = home_pose.copy()
        
    def open_modbus(self):
        client = ModbusTcpClient(self.ip, self.port)
        client.connect()
        self.inspire_node = client
        return 
        
    def write_register(self, address, values):
        # Write to Modbus registers: provide the address and a list of values
        self.inspire_node.write_registers(address, values)

    def read_register(self, address, count):
        # Read from Modbus registers
        response = self.inspire_node.read_holding_registers(address, count)
        return response.registers if response.isError() is False else []

    def read_tactile(self, name: str):
        """
        name: 'index_tip', 'palm', ...
        """
        if name not in TACTILE_LAYOUT:
            raise ValueError(f"Unknown tactile sensor: {name}")

        addr, rows, cols = TACTILE_LAYOUT[name]
        raw = self.read_register(addr, rows * cols)
        if len(raw) < rows * cols:
            raise RuntimeError(f"Tactile read failed for {name}")

        arr = np.asarray(raw, dtype=np.int32)
        if name == "palm":
            return arr.reshape(cols, rows).T
        return arr.reshape(rows, cols)

    def read_all_tactile(self):
        tactile = {}
        for name in TACTILE_LAYOUT:
            tactile[name] = self.read_tactile(name)
        return tactile

    def read_all_tactile_raw(self):
        buf = np.zeros(self.tactile_dim, dtype=np.int16)
        offset = 0

        for name, (addr, rows, cols) in TACTILE_LAYOUT.items():
            raw = self.read_register(addr, rows * cols)
            if len(raw) < rows * cols:
                raise RuntimeError(f"Tactile read failed for {name}")
            raw_arr = np.asarray(raw, dtype=np.int16)
            buf[offset:offset + raw_arr.size] = raw_arr
            offset += raw_arr.size

        return buf

    def write6(self, reg_name, val):
        if reg_name in ['angleSet', 'forceSet', 'speedSet']:
            val_reg = []
            for i in range(6):
                val_reg.append(val[i] & 0xFFFF)  # Take the lower 16 bits
            self.write_register(regdict[reg_name], val_reg)
        else:
            print("Incorrect function call. Usage: reg_name should be 'angleSet', 'forceSet', or 'speedSet', and val should be a list of 6 integers (0~1000). Use -1 as placeholder if needed.")

    def read6(self, reg_name):
        if reg_name in ['angleSet', 'forceSet', 'speedSet', 'angleAct', 'forceAct']:
            val = self.read_register(regdict[reg_name], 6)
            if len(val) < 6:
                print("No data received.")
                return
            return val
        
        elif reg_name in ['errCode', 'statusCode', 'temp']:
            val_act = self.read_register(regdict[reg_name], 3)
            if len(val_act) < 3:
                print("No data received.")
                return

            results = []
            for i in range(len(val_act)):
                low_byte = val_act[i] & 0xFF
                high_byte = (val_act[i] >> 8) & 0xFF
                results.append(low_byte)
                results.append(high_byte)

            return results
        
        else:
            print("Incorrect function call. Usage: reg_name should be one of 'angleSet', 'forceSet', 'speedSet', 'angleAct', 'forceAct', 'errCode', 'statusCode', or 'temp'.")

    def get_qpos(self):
        with self.lock:
            current_hand_angles = np.asarray(self.read6('angleAct'))            
            return current_hand_angles            
    
    def get_force(self):
        with self.lock:
            current_force = np.asarray(self.read6('forceAct'))
            return current_force

    def get_tactile(self):
        if not self.tactile:
            raise RuntimeError("Tactile mode is disabled")
        with self.lock:
            if self.latest_tactile is not None:
                return self.latest_tactile.copy()
            return self.read_all_tactile()
        
    def move_hand(self):
        self.fps = 100
        # self.open_modbus()
        self.hand_lock = Lock()

        # self.write6('speedSet', [1000, 1000, 1000, 1000, 1000, 1000])
        # self.write6('forceSet', [500, 500, 500, 500, 500, 500])
        # self.write6('angleSet', [1000, 1000, 1000, 1000, 1000, 1000])
        
        # self.write_register(1009, 1)
        # while True:
        #     v = self.read_register(1009, 1)
        #     print(v)
        #     if v[0] == 255:
        #         break
        #     time.sleep(1)
                
        while not self.exit_event.is_set():
            start_time = time.time()
            with self.lock:
                action = self.target_action.copy().astype(np.int32)
            
            self.write6('angleSet', action)

            current_hand_angles = np.asarray(self.read6('angleAct'))
            current_force = np.asarray(self.read6('forceAct'))
            if self.tactile:
                current_tactile = np.asarray(self.read_all_tactile())
            # current_action = np.asarray(self.read6('angleSet'))
            with self.lock:
                if self.capture_path is not None:
                    self.data["time"].append(time.time())
                    self.data["position"].append(current_hand_angles.copy())
                    # self.data["time"].append(start_time)
                    self.data["action"].append(action.copy())
                    self.data["force"].append(current_force.copy())
                    if self.tactile:
                        self.data["tactile"].append(current_tactile.copy())

                # if self.tactile:
                #     self.latest_tactile = current_tactile
                #     self.latest_tactile_time = time.time()
            
            end_time = time.time()
            time.sleep(max(0, 1 / self.fps - (end_time - start_time)))
            # print("current loop took: ", end_time - start_time, "sleep time: ", max(0, 1 / self.fps - (end_time - start_time)))

    def move(self, action):
        with self.lock:
            self.target_action = action.copy()
    
    def save(self):
        with self.lock:
            if self.capture_path is not None:       
                os.makedirs(os.path.join(self.capture_path), exist_ok=True)
                for name, value in self.data.items():                     
                    np.save(os.path.join(self.capture_path, f"{name}.npy"), np.array(value))
                    # self.data[name] = value
            self.capture_path = None
                                    
    def end(self):
        self.exit_event.set()
        self.thread.join()
        
        if self.save_event.is_set():
            self.stop()
        
        print("Inspire Exiting...")


    def stop(self):
        self.save_event.clear()
        self.save()
