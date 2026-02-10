import time
import struct
import numpy as np
import pysoem
from threading import Thread, Event

ACTION_DOF = 6


class InspireF1Controller:
    def __init__(self, ifname: str, fps: int = 200):
        self.ifname = ifname
        self.fps = fps
        self.exit_event = Event()

        self.master = pysoem.Master()
        self.slave = None

        self.latest_values = None

        print(f"[ECAT] open {ifname}")
        self.master.open(ifname)

        if self.master.config_init() <= 0:
            raise RuntimeError("No slave found")

        self.slave = self.master.slaves[0]
        self.master.config_map()

        # DC sync (중요)
        self.slave.dc_sync(1, int(1e9 / fps))

        # SAFEOP
        self.slave.state = pysoem.SAFEOP_STATE
        self.master.write_state()
        self.master.state_check(pysoem.SAFEOP_STATE, 50000)

        # OP
        self.slave.state = pysoem.OP_STATE
        self.master.write_state()
        self.master.state_check(pysoem.OP_STATE, 50000)

        print("[ECAT] OP entered")

        # 🔥 WARM-UP LOOP (핵심)
        print("[ECAT] warm-up PDO exchange...")
        for i in range(200):
            self._send_dummy()
            self.master.send_processdata()
            wkc = self.master.receive_processdata(2000)
            if wkc > 0 and self.slave.input:
                print(f"[ECAT] input alive at cycle {i}")
                break
            time.sleep(0.002)
        else:
            print("[ECAT] ❌ input never became valid")

        # start loop
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _send_dummy(self):
        controlword = 0x0F
        targets = [0] * ACTION_DOF
        force = [0] * ACTION_DOF
        speed = [200] * ACTION_DOF
        tx = [controlword] + targets + force + speed
        self.slave.output = struct.pack("<19h", *tx)

    def _loop(self):
        period = 1.0 / self.fps
        while not self.exit_event.is_set():
            t0 = time.time()

            self._send_dummy()
            self.master.send_processdata()
            wkc = self.master.receive_processdata(2000)

            if wkc > 0 and self.slave.input:
                data = bytes(self.slave.input)
                values = struct.unpack_from("<76h", data, 0)
                self.latest_values = values

                # 🔥 이거 한 줄이라도 찍혀야 정상
                print(
                    f"[RX] angle0={values[6]} force0={values[12]} current0={values[18]}"
                )

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def close(self):
        self.exit_event.set()
        self.thread.join()
        self.master.close()
        print("[ECAT] closed")
