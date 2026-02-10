import pysoem
import time
import sys
import ctypes

import time
import numpy as np
import sys
# 위에서 만든 파일 import (파일명이 inspire_f1_controller.py 라고 가정)
from paradex.io.robot_controller.inspire_f1_controller_ethercat import InspireF1Controller

def main():
    # ★ 중요: 포트 이름을 선생님 PC의 이더넷 인터페이스 이름으로 변경
    port_name = "enp7s0f3" 
    
    print("컨트롤러 시작...")
    # tactile=True로 해도 EtherCAT 데이터 구조 차이로 완벽하지 않을 수 있음
    ic = InspireF1Controller(port=port_name, tactile=True) 

    try:
        start_time = time.time()
        print("\n--- 5초간 손을 쥐었다 폅니다 ---")
        
        while time.time() - start_time < 10.0:
            t = time.time() - start_time
            
            # 0~1000 사이 값 생성 (1000=Open, 0=Close)
            # 사인파로 부드럽게 움직이기
            val = 500 + 500 * np.sin(t * 2.0) 
            
            # 모든 손가락에 같은 값 적용
            action = np.array([val] * 6)
            ic.move(action)
            
            # 상태 출력
            data = ic.get_data()
            print(data)
            if data['qpos'] is not None:
                # 라디안 -> 각도 변환해서 보기 편하게 출력
                deg = np.rad2deg(data['qpos'])
                print(f"\r[t={t:.1f}] Cmd:{val:.0f} | Pos(deg): {deg[0]:.1f} {deg[1]:.1f} ...", end="")
            
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\n중단됨!")
    finally:
        ic.end()

if __name__ == "__main__":
    main()