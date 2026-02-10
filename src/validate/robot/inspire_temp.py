import serial
import time

PORT = "/dev/ttyUSB0"        # 왼손 포트
BAUD = 115200        # 필요하면 921600도 시도
HAND_ID = 1          # 혹시 안 되면 255도 시도

def checksum(body):
    return sum(body) & 0xFF

def read_reg(addr, nbytes):
    body = bytearray([
        HAND_ID & 0xFF,
        0x04,
        0x11,
        addr & 0xFF,
        (addr >> 8) & 0xFF,
        nbytes & 0xFF,
    ])
    frame = bytes([0xEB, 0x90]) + body + bytes([checksum(body)])
    ser.reset_input_buffer()
    ser.write(frame)
    time.sleep(0.05)
    return ser.read(64)

def write_reg(addr, payload):
    body = bytearray([
        HAND_ID & 0xFF,
        len(payload) + 3,
        0x12,
        addr & 0xFF,
        (addr >> 8) & 0xFF,
    ]) + payload
    frame = bytes([0xEB, 0x90]) + body + bytes([checksum(body)])
    ser.reset_input_buffer()
    ser.write(frame)
    time.sleep(0.05)
    return ser.read(64)

ser = serial.Serial(PORT, BAUD, timeout=0.5)
time.sleep(0.5)

print("=== TRY READ ACTUAL ANGLE ===")
print(read_reg(0x0428, 12))   # ACTUAL_ANGLE

print("=== TRY READ STATUS ===")
print(read_reg(0x0436, 12))   # STATUS / ERROR 근처

print("=== TRY CLEAR ERROR ===")
print(write_reg(0x03EC, bytes([1])))

print("=== READ AGAIN ===")
print(read_reg(0x0428, 12))

ser.close()
