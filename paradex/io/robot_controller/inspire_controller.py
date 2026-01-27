import time
import serial
from threading import Thread, Event, Lock
import numpy as np
import os
import json


action_dof = 6
hand_id = 1
BAUDRATE_REG_ADDR = 1002
command = {
    'setpos':[0xEB, 0x90, hand_id, 0x0F, 0x12, 0xC2, 0x05],
    'setangle':[0xEB, 0x90, hand_id, 0x0F, 0x12, 0xCE, 0x05],
    'setpower':[0xEB, 0x90, hand_id, 0x0F, 0x12, 0xDA, 0x05],
    'setspeed':[0xEB, 0x90, hand_id, 0x0F, 0x12, 0xF2, 0x05],
    'getsetspeed':[0xEB, 0x90, hand_id, 0x04, 0x11, 0xC2, 0x05, 0x0C],
    'getsetangle':[0xEB, 0x90, hand_id, 0x04, 0x11, 0xCE, 0x05, 0x0C],
    'getsetpower':[0xEB, 0x90, hand_id, 0x04, 0x11, 0xDA, 0x05, 0x0C],
    'getactpos':[0xEB, 0x90, hand_id, 0x04, 0x11, 0xFE, 0x05, 0x0C],
    'getactangle':[0xEB, 0x90, hand_id, 0x04, 0x11, 0x0A, 0x06, 0x0C],
    'getactforce':[0xEB, 0x90, hand_id, 0x04, 0x11, 0x2E, 0x06, 0x0C], 
    'setbaudrate':[0xEB, 0x90, hand_id, 0x04, 0x12, 0xEA, 0x03],
}

TOUCH_SENSOR_BASE_ADDR = 3000
TOTAL_REG_COUNT = 2124
# TACTILE_LAYOUT = {
#     "little_tip":    {"addr": 3000, "rows": 3, "cols": 3},
#     "little_nail":   {"addr": 3018, "rows": 12, "cols": 8},
#     "little_pad":    {"addr": 3210, "rows": 10, "cols": 8},
#     "ring_tip":      {"addr": 3370, "rows": 3, "cols": 3},
#     "ring_nail":     {"addr": 3388, "rows": 12, "cols": 8},
#     "ring_pad":      {"addr": 3580, "rows": 10, "cols": 8},
#     "middle_tip":    {"addr": 3740, "rows": 3, "cols": 3},
#     "middle_nail":   {"addr": 3758, "rows": 12, "cols": 8},
#     "middle_pad":    {"addr": 3950, "rows": 10, "cols": 8},
#     "index_tip":     {"addr": 4110, "rows": 3, "cols": 3},
#     "index_nail":    {"addr": 4128, "rows": 12, "cols": 8},
#     "index_pad":     {"addr": 4320, "rows": 10, "cols": 8},
#     "thumb_tip":     {"addr": 4480, "rows": 3, "cols": 3},
#     "thumb_nail":    {"addr": 4498, "rows": 12, "cols": 8},
#     "thumb_middle":  {"addr": 4690, "rows": 3, "cols": 3},
#     "thumb_pad":     {"addr": 4708, "rows": 12, "cols": 8},
#     "palm":          {"addr": 4900, "rows": 8, "cols": 14},
# }


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

def data2bytes(data):
    data = int(data)
    rdata = [0xff]*2
    if data == -1:
        rdata[0] = 0xff
        rdata[1] = 0xff
    else:
        rdata[0] = data&0xff
        rdata[1] = (data>>8)&(0xff)
    return rdata

def num2str(num):
    str = hex(num)
    str = str[2:4]
    if(len(str) == 1):
        str = '0'+ str
    str = bytes.fromhex(str)    
    return str

def checknum(data):
    result = 0
    for v in data[2:-1]:
        result += v
    result = result&0xff
    return result

def data2str(data):
    ret = b''
    for v in data:
        ret = ret + num2str(v)
    return ret

def crc16(data: bytes):
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


class InspireController:
    def __init__(self, addr, tactile = False, baud_rate = 115200):
        self.home_pose = np.zeros(action_dof) + 800
        
        self.save_event = Event()
        self.exit_event = Event()
        self.connection_event = Event()
        
        self.lock = Lock()
        self.serial_lock = Lock()
        
        self.action = np.zeros(action_dof) + 800
        self.joint_value = np.zeros(action_dof) + 800
        self.addr = addr
        self.tactile = tactile
        self.default_baud = 115200
        self.desired_baud = baud_rate
        
        self.tactile_index, self.tactile_dim = self.build_tactile_index()

        
        self.open_serial(baud=self.default_baud)
        if self.desired_baud != self.default_baud:
            success = self.set_baud_rate(self.desired_baud)
            if not success:
                print(f"[Inspire] Baud change to {self.desired_baud} failed, reverting to {self.default_baud}")
                self.desired_baud = self.default_baud
        
        self.write6('setspeed', [1000, 1000, 1000, 1000, 1000, 1000])
        self.write6('setpower', [400, 400, 400, 400, 400, 400])
        self.write6('setangle', [1000, 1000, 1000, 1000, 1000, 1000])
        
        self.connection_event.set()
        
        self.latest_tactile = None
        self.latest_tactile_time = None
        
        self.control_thread = Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        
        # self.tactile_thread = Thread(target=self.tactile_loop)
        # self.tactile_thread.daemon = True
        # self.tactile_thread.start()
        
        
        self.connection_event.wait()
        
    def build_tactile_index(self):
        index = {}
        offset = 0
        for name, (addr, rows, cols) in TACTILE_LAYOUT.items():
            size = rows * cols
            index[name] = (offset, rows, cols)
            offset += size
        return index, offset
        
    def serial_read_registers(self, start_addr: int, count: int):
        """
        RS485 register block read (Modbus RTU style)
        """
        with self.serial_lock:
            device_id = hand_id
            function = 0x03  # Read Holding Registers

            frame = bytearray([
                device_id,
                function,
                (start_addr >> 8) & 0xFF,
                start_addr & 0xFF,
                (count >> 8) & 0xFF,
                count & 0xFF,
            ])

            crc = crc16(frame)
            frame += crc.to_bytes(2, byteorder="little")

            self.ser.write(frame)

            # response = id(1) + func(1) + byte_cnt(1) + data + crc(2)
            expected = 5 + count * 2
            resp = self.ser.read(expected)

            if len(resp) != expected:
                raise RuntimeError("Tactile serial read failed")

            data = resp[3:-2]
            regs = []
            for i in range(0, len(data), 2):
                # Device returns high byte first (Modbus RTU convention). Keep this
                # low-first decoding for legacy tactile parsing; use the _be variant
                # when you need the canonical big-endian interpretation.
                regs.append(data[i] | (data[i + 1] << 8))

            return np.array(regs, dtype=np.int32)

    def serial_read_registers_be(self, start_addr: int, count: int):
        """
        Big-endian variant (hi byte first -> canonical Modbus value).
        """
        raw = self.serial_read_registers(start_addr, count)
        return np.array([(v >> 8) | ((v & 0xFF) << 8) for v in raw], dtype=np.int32)
    
    

    def control_loop(self):
        self.fps = 30
        
        while not self.exit_event.is_set():
            start_time = time.time()
            
            with self.lock:
                action = self.action.copy().astype(np.int32)
            
            self.write6('setangle', action)

            try:
                current_hand_angles = np.asarray(self.read6('getactangle'))

            except Exception as exc:
                print(f"[Inspire] Control loop read failed: {exc}")
                time.sleep(0.05)
                continue
      
            if self.save_event.is_set():
                self.data["time"].append(time.time())

            try:
                
                current_hand_angles = np.asarray(self.read6('getactangle'))
                current_force = np.asarray(self.read6('getactforce'))
                
                if self.tactile:
                    current_tactile = np.asarray(self.read_all_tactile())
            except Exception as exc:
                print(f"[Inspire] Control loop read failed: {exc}")
                time.sleep(0.05)
                continue
            
            with self.lock:
                self.joint_value = current_hand_angles.copy()
                
                if self.tactile:
                    self.latest_tactile = current_tactile
                    self.latest_tactile_time = start_time
            
            if self.save_event.is_set():
                self.data["position"].append(current_hand_angles.copy())

                self.data["action"].append(action.copy())
                self.data["force"].append(current_force.copy())
                if self.tactile:
                    self.data["tactile"].append(current_tactile.copy())
            
            end_time = time.time()
            time.sleep(max(0, 1 / self.fps - (end_time - start_time)))
            # print("current loop took: ", end_time - start_time, "sleep time: ", max(0, 1 / self.fps - (end_time - start_time)))


    def tactile_loop(self):
        tactile_fps = 1.0
        interval = 1.0 / tactile_fps

        while not self.exit_event.is_set():
            t0 = time.time()

            current_tactile = np.asarray(self.read_all_tactile_raw())

            with self.lock:
                self.latest_tactile = current_tactile
                self.latest_tactile_time = t0

                if self.save_event.is_set():
                    self.data["tactile"].append(current_tactile)

            time.sleep(max(0, interval - (time.time() - t0)))

    def start(self, save_path):
        self.capture_path = save_path
        self.data = {
            "time": [],
            "position": [],
            "action": [],
            "force": [],
            "tactile": [],
        }
        self.save_event.set()

    def stop(self):
        self.save_event.clear()
        
        os.makedirs(self.capture_path, exist_ok=True)
        for name, value in self.data.items():                     
            np.save(os.path.join(self.capture_path, f"{name}.npy"), np.array(value))
                    
        self.capture_path = None
        
    def end(self):
        self.exit_event.set()
        
        self.control_thread.join()
        # self.tactile_thread.join()
        
        if self.save_event.is_set():
            self.stop()
        
        print("Inspire Exiting...")
                    
    def open_serial(self, baud=115200):
        if hasattr(self, "ser") and self.ser is not None and self.ser.is_open:
            self.ser.close()
        self.ser = serial.Serial(self.addr, baud, timeout=0.1)
        self.current_baud = baud
        # Clear any stale buffers after open.
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass
        return

    def _baud_to_reg_value(self, baud_rate: int) -> int:
        if baud_rate == 115200:
            return 0
        if baud_rate == 921600:
            return 3
        raise ValueError(f"Unsupported baud rate {baud_rate}. Only 115200 or 921600 are supported.")

    def set_baud_rate(self, baud_rate: int):
        """
        Update the Inspire hand baud rate using the native (0xEB 0x90) protocol,
        then reopen the serial port at the requested baud.
        """
        baud_value = self._baud_to_reg_value(baud_rate)
        try:
            self._write_baud_native(baud_value)
        except Exception as exc:
            print(f"[Inspire] Failed to send baud change (native): {exc}")
            return False

        # Give the device time to switch its UART clock.
        time.sleep(1.0)
        self.open_serial(baud_rate)
        time.sleep(0.2)
        # Verify the new baud by attempting a read; if it fails, revert.
        try:
            self.read6('getactangle')
        except Exception as exc:
            self.open_serial(self.default_baud)
            time.sleep(0.2)
            try:
                self.read6('getactangle')
            except Exception as exc_fallback:
                print(f"[Inspire] Baud verify failed (new and default): {exc} / {exc_fallback}")
                return False
            return False
        return True

    def _write_baud_native(self, baud_value: int):
        """
        Native protocol write to baud register (addr 1002) with 1-byte payload.
        Packet: EB 90 | ID | len(0x04) | flag(0x12) | addrL | addrH | data | checksum
        """
        addr = BAUDRATE_REG_ADDR
        packet = [
            0xEB, 0x90, hand_id,
            0x04,  # length: flag(1) + addr(2) + data(1)
            0x12,  # write register flag per Inspire protocol
            addr & 0xFF, (addr >> 8) & 0xFF,
            baud_value,
            0x00,  # checksum placeholder
        ]
        packet[-1] = checknum(packet)
        frame = data2str(packet)

        with self.serial_lock:
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception:
                pass
            self.ser.write(frame)
            resp = self.ser.read(9)

        if len(resp) == 0:
            raise RuntimeError("No response to baud change command")
        return resp

    def serial_write_registers(self, start_addr: int, values):
        """
        RS485 multi-register write (Modbus RTU style, function code 0x10)
        """
        with self.serial_lock:
            device_id = hand_id
            function = 0x10  # Write Multiple Registers
            count = len(values)
            byte_count = count * 2
            frame = bytearray([
                device_id,
                function,
                (start_addr >> 8) & 0xFF,
                start_addr & 0xFF,
                (count >> 8) & 0xFF,
                count & 0xFF,
                byte_count
            ])
            for v in values:
                frame.append((v >> 8) & 0xFF)
                frame.append(v & 0xFF)

            crc = crc16(frame)
            frame += crc.to_bytes(2, byteorder="little")

            self.ser.write(frame)

            expected = 8  # Echo of the request header with CRC
            resp = self.ser.read(expected)

            if len(resp) != expected:
                raise RuntimeError(f"Serial multi-write failed for addr {start_addr}")
    
    def write6(self, command_name, value):
        with self.serial_lock:
            datanum = command[command_name][3]
            len_command = len(command[command_name])
            
            b = [0] * (datanum + 5)
            
            for i, v in enumerate(command[command_name]):
                b[i] = v
            
            for i in range(6):
                b[len_command + 2 * i] = data2bytes(value[i])[0]
                b[len_command + 2 * i + 1] = data2bytes(value[i])[1]
            
            b[-1] = checknum(b)
            putdata = data2str(b)
            self.ser.write(putdata)
            getdata = self.ser.read(9)
        
    def read6(self, command_name):
        with self.serial_lock:
            datanum = command[command_name][3]
            len_command = len(command[command_name])
            
            b = [0] * (datanum+5)
            for i, v in enumerate(command[command_name]):
                b[i] = v
            b[-1] = checknum(b)
            putdata = data2str(b)
            self.ser.write(putdata)
            
            getdata = self.ser.read(20)
            expected_len = 19  # minimum bytes needed for parsing indices up to 18
            if len(getdata) < expected_len:
                raise RuntimeError(f"read6 {command_name} returned {len(getdata)} bytes, expected at least {expected_len}")
            # print("getdata:", getdata)
            ret = np.zeros(6)
            
            for i in range(6):
                if getdata[i*2+7] == 0xff and getdata[i*2+8] == 0xff:
                    ret[i] = -1
                else:
                    ret[i] = getdata[i*2+7] + (getdata[i*2+8]<<8)
            return ret

    def serial_write_register(self, address: int, value: int):
        """
        RS485 single-register write (Modbus RTU style, function code 0x06)
        """
        with self.serial_lock:
            device_id = hand_id
            function = 0x06  # Write Single Register
            frame = bytearray([
                device_id,
                function,
                (address >> 8) & 0xFF,
                address & 0xFF,
                (value >> 8) & 0xFF,
                value & 0xFF,
            ])

            crc = crc16(frame)
            frame += crc.to_bytes(2, byteorder="little")

            self.ser.write(frame)

            expected = 8  # Echo of the request with CRC
            resp = self.ser.read(expected)

            if len(resp) != expected:
                raise RuntimeError(f"Serial write failed for addr {address}")
    
    def read_tactile(self, name: str):
        """
        name: 'index_tip', 'palm', ...
        """
        if name not in TACTILE_LAYOUT:
            raise ValueError(f"Unknown tactile sensor: {name}")

        addr, rows, cols = TACTILE_LAYOUT[name]
        raw = self.serial_read_registers(addr, rows * cols)

        if name == "palm":
            # palm은 column-first → transpose
            return raw.reshape(cols, rows).T
        else:
            return raw.reshape(rows, cols)
        
        
    def read_all_tactile(self):
        tactile = {}
        for name in TACTILE_LAYOUT:
            tactile[name] = self.read_tactile(name)
        return tactile
    
    def read_all_tactile_raw(self):
        buf = np.zeros(self.tactile_dim, dtype=np.int16)
        offset = 0

        for name, (addr, rows, cols) in TACTILE_LAYOUT.items():
            raw = self.serial_read_registers(addr, rows * cols)
            buf[offset:offset + raw.size] = raw
            offset += raw.size

        return buf
    
    def get_data(self):
        ret = {}
        with self.lock:
            ret["action"] = self.action.copy()
            ret["joint_value"] = self.joint_value.copy()
            ret["time"] = time.time()
            
        return ret
    
    def get_force(self):
        with self.lock:
            current_force = np.asarray(self.read6('getactforce'))
            return current_force
        
    def move(self, action):
        with self.lock:
            self.action = action.copy()
    
    def is_error(self):
        return False
