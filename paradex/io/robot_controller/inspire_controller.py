import time
from pymodbus.client import ModbusTcpClient  # pip3 install pymodbus==2.5.3
import time

import numpy as np
from threading import Thread, Event, Lock

import os
import json
from paradex.utils.file_io import config_dir

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
        network_config = json.load(open(os.path.join(config_dir, "environment/network.json"), "r"))
        self.ip = network_config["inspire"]["ip"]
        self.port = network_config["inspire"]["port"]
        
        self.home_pose = np.zeros(action_dof)+500
        
        self.capture_path = save_path
        if save_path is not None:
            os.makedirs(self.capture_path, exist_ok=True)
        
        self.cnt = 0
        
        T = 60000
        self.data = {
            "time":np.zeros((T,1), dtype=np.float64),
            "position":np.zeros((T, action_dof), dtype=np.float64),
            "action":np.zeros((T, action_dof), dtype=np.float64),
            "force":np.zeros((T, action_dof), dtype=np.float64)
        }
        
        self.exit = Event()
        self.lock = Lock()
        self.target_action = np.zeros(action_dof)+500
        
        self.thread = Thread(target=self.move_hand)
        self.thread.daemon = True
        self.thread.start()

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
        self.fps = 100
        self.open_modbus()
        self.hand_lock = Lock()

        self.write6('speedSet', [1000, 1000, 1000, 1000, 1000, 1000])
        self.write6('forceSet', [500, 500, 500, 500, 500, 500])
        self.write6('angleSet', [1000, 1000, 1000, 1000, 1000, 1000])
        time.sleep(1)
        
        # self.write_register(1009, 1)
        # while True:
        #     v = self.read_register(1009, 1)
        #     print(v)
        #     if v[0] == 255:
        #         break
        #     time.sleep(1)
                
        while not self.exit.is_set():
            start_time = time.time()
            with self.lock:
                action = self.target_action.copy().astype(np.int32)
            
            self.write6('angleSet', action)

            current_hand_angles = np.asarray(self.read6('angleAct'))
            current_force = np.asarray(self.read6('forceAct'))
            # current_action = np.asarray(self.read6('angleSet'))
            
            self.data["position"][self.cnt] = current_hand_angles.copy()
            self.data["time"][self.cnt] = start_time
            self.data["action"][self.cnt] = action.copy()
            self.data["force"][self.cnt] = current_force.copy()
            
            self.cnt += 1
                
            
            end_time = time.time()
            time.sleep(max(0, 1 / self.fps - (end_time - start_time)))

    def set_target_action(self, action):
        with self.lock:
            self.target_action = action.copy()
    
    def save(self):
        with self.lock:
            if self.capture_path is not None:       
                os.makedirs(os.path.join(self.capture_path, "inspire"), exist_ok=True)
                for name, value in self.data.items():                     
                    np.save(os.path.join(self.capture_path, "inspire", f"{name}.npy"), value[:self.cnt])
                                    
    def quit(self):
        self.exit.set()
        self.save()
        self.thread.join()
        print("Inspire Exiting...")
