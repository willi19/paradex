import time
from pymodbus.client import ModbusTcpClient  # pip3 install pymodbus==2.5.3
import time

import numpy as np
from multiprocessing import Process, shared_memory, Event, Lock

import os

action_dof = 6
regdict = {
    'ID': 1000,
    'baudrate': 1001,
    'clearErr': 1004,
    'forceClb': 1009,
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


class InspireController:
    def __init__(self, save_path=None):

        self.home_pose = None
        self.ip = '192.168.11.210'
        self.port = 6000
        
        self.capture_path = save_path
        if save_path is not None:
            os.makedirs(self.capture_path, exist_ok=True)

        self.hand_state_hist = np.zeros((60000, action_dof), dtype=np.float64)
        self.hand_timestamp = np.zeros((60000, 1), dtype=np.float64)        
        self.hand_action_hist = np.zeros((60000, action_dof), dtype=np.float64)

        self.exit = Event()

        self.shm = {}
        self.create_shared_memory("hand_target_action", action_dof * np.dtype(np.float32).itemsize)
        self.hand_target_action_array = np.ndarray((action_dof,), dtype=np.float32, buffer=self.shm["hand_target_action"].buf)

        self.hand_process = Process(target=self.move_hand)
        self.hand_process.start()

    def set_homepose(self, home_pose):
        assert home_pose.shape == (action_dof,)
        self.home_pose = home_pose.copy()
    
    def create_shared_memory(self, name, size):
        try:
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_shm.close()
            existing_shm.unlink()
        except FileNotFoundError:
            pass
        self.shm[name] = shared_memory.SharedMemory(create=True, name=name, size=size)

    def home_robot(self):
        self.hand_target_action_array[:] = self.home_pose.copy()

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

    def move_hand(self):
        fps = 100
        self.hand_cnt = 0
        self.open_modbus()
        self.hand_lock = Lock()

        self.write6('speedSet', [1000, 1000, 1000, 1000, 1000, 1000])


        while not self.exit.is_set():
            start_time = time.time()
            action = self.hand_target_action_array.copy().astype(np.int32)

            current_hand_angles = np.asarray(self.read6('angleAct'))
            self.write6('angleSet', action)
                
            self.hand_state_hist[self.hand_cnt] = current_hand_angles.copy()
            self.hand_timestamp[self.hand_cnt] = start_time
            self.hand_action_hist[self.hand_cnt] = self.hand_target_action_array.copy()
            self.hand_cnt += 1
                
            
            end_time = time.time()
            time.sleep(max(0, 1 / fps - (end_time - start_time)))
        
        if self.capture_path is not None:    
            os.makedirs(os.path.join(self.capture_path, "hand"), exist_ok=True)
            np.save(os.path.join(self.capture_path, "hand", f"state.npy"), self.hand_state_hist[:self.hand_cnt])
            np.save(os.path.join(self.capture_path, "hand", f"timestamp.npy"), self.hand_timestamp[:self.hand_cnt])
            np.save(os.path.join(self.capture_path, "hand", f"action.npy"), self.hand_action_hist[:self.hand_cnt])
        

    def set_target_action(self, action):
        self.hand_target_action_array[:] = action.copy()
        
    def quit(self):
        self.exit.set()
        self.hand_process.join()

        for key in self.shm.keys():
            self.shm[key].close()
            self.shm[key].unlink()

        print("Exiting...")
