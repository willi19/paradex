import time
import threading
import numpy as np
from pymodbus.client import ModbusTcpClient
from queue import Queue
import matplotlib.pyplot as plt
# ===============================
# :압정: Modbus TCP 기본 설정
# ===============================
MODBUS_IP = "192.168.11.210"
MODBUS_PORT = 6000
TOUCH_SENSOR_BASE_ADDR = 3000
TOTAL_REG_COUNT = 2124  # 5123 - 3000 + 1
# ===============================
# :압정: 전역 변수
# ===============================
data_queue = Queue()
# ===============================
# :압정: 센서 레이아웃 정의
# ===============================
SENSOR_LAYOUT = {
    "little_tip":    {"addr": 3000, "rows": 3, "cols": 3},
    "little_nail":   {"addr": 3018, "rows": 12, "cols": 8},
    "little_pad":    {"addr": 3210, "rows": 10, "cols": 8},
    "ring_tip":      {"addr": 3370, "rows": 3, "cols": 3},
    "ring_nail":     {"addr": 3388, "rows": 12, "cols": 8},
    "ring_pad":      {"addr": 3580, "rows": 10, "cols": 8},
    "middle_tip":    {"addr": 3740, "rows": 3, "cols": 3},
    "middle_nail":   {"addr": 3758, "rows": 12, "cols": 8},
    "middle_pad":    {"addr": 3950, "rows": 10, "cols": 8},
    "index_tip":     {"addr": 4110, "rows": 3, "cols": 3},
    "index_nail":    {"addr": 4128, "rows": 12, "cols": 8},
    "index_pad":     {"addr": 4320, "rows": 10, "cols": 8},
    "thumb_tip":     {"addr": 4480, "rows": 3, "cols": 3},
    "thumb_nail":    {"addr": 4498, "rows": 12, "cols": 8},
    "thumb_middle":  {"addr": 4690, "rows": 3, "cols": 3},
    "thumb_pad":     {"addr": 4708, "rows": 12, "cols": 8},
    "palm":          {"addr": 4900, "rows": 8, "cols": 14},
}
# ===============================
# :압정: 데이터 파싱 함수
# ===============================
def parse_tactile_data(registers):
    parsed = {}
    for name, meta in SENSOR_LAYOUT.items():
        start = meta["addr"] - TOUCH_SENSOR_BASE_ADDR
        rows, cols = meta["rows"], meta["cols"]
        length = rows * cols
        raw = registers[start: start + length]
        array = np.array(raw, dtype=np.uint16).reshape(rows, cols)
        if name == "palm":
            array = array[::-1].T  # 전치 후 세로 방향 반전 (문서 기준)
        parsed[name] = array
    return parsed
# ===============================
# :압정: 히트맵 시각화 함수
# ===============================
def show_tactile_heatmap(data_dict):
    fig, axs = plt.subplots(5, 4, figsize=(16, 20))
    axs = axs.flatten()
    for i, (name, data) in enumerate(data_dict.items()):
        ax = axs[i]
        im = ax.imshow(data, cmap='viridis', vmin=0, vmax=4095)
        ax.set_title(name)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()
# ===============================
# :압정: Modbus 데이터 읽기 쓰레드
# ===============================
def read_modbus_data(client):
    last_time = time.time()
    read_count = 0
    all_registers = []
    all_registers.clear()
    for addr in range(TOUCH_SENSOR_BASE_ADDR, 5124, 120):
        count = min(120, 5124 - addr)
        response = client.read_holding_registers(addr, count)
        if response.isError():
            print(f":x: Failed to read registers {addr} ~ {addr + count - 1}")
            continue
        all_registers.extend(response.registers)
        print(addr)
    return all_registers
# ===============================
# :압정: 데이터 처리 및 시각화 쓰레드
# ===============================
def process_data(raw_data):
    print(f"[INFO] Received data length: {len(raw_data)}")
    if len(raw_data) != TOTAL_REG_COUNT:
        print(":느낌표: Unexpected data length:", len(raw_data))
        return
    tactile_map = parse_tactile_data(raw_data)
    print(f"[INFO] Parsed tactile data: {list(tactile_map.keys())}")
    show_tactile_heatmap(tactile_map)
# ===============================
# :압정: 메인 함수
# ===============================
def main():
    client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT)
    client.connect()
    plt.ion()  # 인터랙티브 모드
    fig, axs = plt.subplots(5, 4, figsize=(16, 20))
    axs = axs.flatten()
    im_list = []
    # 초기 히트맵 생성
    for i, name in enumerate(SENSOR_LAYOUT.keys()):
        dummy = np.zeros((SENSOR_LAYOUT[name]["rows"], SENSOR_LAYOUT[name]["cols"]))
        im = axs[i].imshow(dummy, cmap='viridis', vmin=0, vmax=4095)
        axs[i].set_title(name)
        plt.colorbar(im, ax=axs[i], shrink=0.8)
        im_list.append(im)
    plt.tight_layout()
    while True:
        all_register = read_modbus_data(client)
        if len(all_register) != TOTAL_REG_COUNT:
            print(":느낌표: Unexpected data length")
            continue
        tactile_map = parse_tactile_data(all_register)
        for i, (name, data) in enumerate(tactile_map.items()):
            im_list[i].set_data(data)
        fig.canvas.draw()
        plt.pause(0.01)
if __name__ == "__main__":
    main()