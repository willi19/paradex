"""In-process smoke test for the refactored remote_camera_controller.

No hardware: spins a mock daemon (ping REP 5480, command REP 5482, health PUB 5481)
on localhost and drives rcc through connect→register→start→health→stop→end,
asserting no deadlock and that health arrives via the PUB channel.
"""
import sys, time, threading, json
import zmq

import paradex.io.camera_system.remote_camera_controller as rccmod

# ── point the controller at localhost ────────────────────────────────────────
rccmod.get_pc_list = lambda: ["mock"]
rccmod.get_pc_ip = lambda pc: "127.0.0.1"

PING, MON, CMD = 5480, 5481, 5482
stop = threading.Event()
state = {"controller": None, "running": False, "fid": 0, "hb": 0, "started": False}


def ping_thread():
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REP); s.bind(f"tcp://*:{PING}")
    while not stop.is_set():
        try:
            if s.poll(100):
                s.recv_string(); s.send_string("pong")
        except zmq.ZMQError:
            break


def cmd_thread():
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REP); s.bind(f"tcp://*:{CMD}")
    while not stop.is_set():
        if not s.poll(100):
            continue
        cmd = s.recv_json()
        a = cmd.get("action")
        if a == "register":
            prev = state["controller"]
            if prev is not None and prev != cmd.get("controller_name") and not cmd.get("force"):
                s.send_json({"status": "error", "msg": f"locked by {prev}"})
            else:
                state["controller"] = cmd.get("controller_name")
                s.send_json({"status": "ok", "msg": "registered", "version": cmd.get("version")})
        elif a == "start":
            state["running"] = cmd.get("mode") in ("video", "stream", "full", "acquire")
            state["started"] = True
            s.send_json({"status": "ok", "msg": "started"})
        elif a == "stop":
            state["running"] = False
            s.send_json({"status": "ok", "msg": "stopped"})
        elif a == "heartbeat":
            state["hb"] += 1
            s.send_json({"status": "ok", "msg": "heartbeat", "running": state["running"]})
        elif a == "end":
            state["controller"] = None
            s.send_json({"status": "ok", "msg": "ended"})
        elif a == "reload":
            s.send_json({"status": "ok", "msg": "reloaded"})
        elif a == "sink":
            for k in ("video", "stream", "snapshot", "save_path"):
                if k in cmd:
                    state.setdefault("sinks", {})[k] = cmd[k]
            s.send_json({"status": "ok", "msg": "sink set", "running": state["running"]})
        else:
            s.send_json({"status": "error", "msg": "unknown"})


def pub_thread():
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PUB); s.bind(f"tcp://*:{MON}")
    while not stop.is_set():
        if state["running"]:
            state["fid"] += 1
        summary = {
            "expected_camera_count": 1,
            "detected_camera_count": 1,
            "camera_names": ["cam0"],
            "frame_ids": {"cam0": state["fid"]},
            "states": {"cam0": "CAPTURING" if state["running"] else "READY"},
            "errors": {},
        }
        s.send_json({"cameras": [], "controller": state["controller"] or "None",
                     "running": state["running"], "summary": summary})
        time.sleep(0.1)


def main():
    for t in (ping_thread, cmd_thread, pub_thread):
        threading.Thread(target=t, daemon=True).start()
    time.sleep(0.3)

    rcc = rccmod.remote_camera_controller("test", pc_list=["mock"])
    time.sleep(0.6)
    assert rcc.init_error is None, f"init failed: {rcc.init_error}"
    assert state["controller"] and state["controller"].startswith("test_"), "register did not land"
    print("[ok] connected + registered; controller =", state["controller"])

    rcc.arm(syncMode=False, fps=30)
    rcc.set_stream(True)
    time.sleep(0.8)
    st = rcc.get_status()
    assert state["started"], "start command never reached daemon"
    assert not st["error"], f"unexpected error: {st}"
    pc = st["pc"]["mock"]
    assert pc.get("frame_ids", {}).get("cam0", 0) > 0, f"no frame_ids from PUB: {pc}"
    assert pc.get("states", {}).get("cam0") == "CAPTURING", f"state not from PUB: {pc}"
    print(f"[ok] start; health via PUB: fid={pc['frame_ids']['cam0']} state={pc['states']['cam0']} "
          f"heartbeats={state['hb']}")

    rcc.stop()
    time.sleep(0.3)
    assert not rcc.get_status()["error"], "error after stop"
    print("[ok] stop")

    # ── decoupled-sink API: arm (no sink) then toggle sinks live ──────────────
    state["sinks"] = {}
    rcc.arm(syncMode=False, fps=30)
    time.sleep(0.4)
    assert state["started"] and state["running"], "arm did not start a running capture"
    rcc.set_stream(True); time.sleep(0.2)
    rcc.set_record("dataset/001", True); time.sleep(0.2)
    rcc.snapshot("dataset/001", 3); time.sleep(0.2)
    rcc.set_record(on=False); time.sleep(0.2)
    sk = state.get("sinks", {})
    assert sk.get("stream") is True, f"stream sink not received: {sk}"
    assert sk.get("save_path") == "dataset/001", f"record save_path not received: {sk}"
    assert sk.get("snapshot") == [ "dataset/001", 3 ], f"snapshot not received: {sk}"
    assert sk.get("video") is False, f"final video-off not received: {sk}"
    print(f"[ok] arm + live sink toggles reached daemon: {sk}")
    rcc.stop(); time.sleep(0.2)
    print("[ok] stop after arm")

    # ── orphaned daemon (restart / dead-man) → rcc auto re-registers ──────────
    me = rcc.name
    state["controller"] = None                 # simulate a daemon restart
    time.sleep(3.5)
    assert state["controller"] == me, \
        f"rcc did not re-claim orphaned daemon (controller={state['controller']})"
    print("[ok] orphaned daemon auto re-registered")

    # ── another controller holds the lock → rcc must NOT fight it ─────────────
    state["controller"] = "intruder_ctrl"
    time.sleep(3.0)
    assert state["controller"] == "intruder_ctrl", \
        f"rcc stole the lock from another controller: {state['controller']}"
    print("[ok] rcc did not fight a real takeover")
    state["controller"] = me                   # restore for clean end()

    # ── daemon dies mid-capture → sticky 'capture_interrupted' ────────────────
    rcc.arm(syncMode=False, fps=30)                      # a recording session
    rcc.set_record("dataset/002", on=True)
    time.sleep(0.5)
    assert not rcc.capture_interrupted(), "false interrupt before any failure"
    state["controller"] = None                 # daemon restarted mid-recording
    time.sleep(1.5)
    assert rcc.capture_interrupted(), "mid-capture daemon death not detected"
    assert rcc.is_error(), "is_error should be True after interruption"
    st = rcc.get_status()
    assert st["capture_interrupted"] and st["interrupt_msg"], f"status missing interrupt: {st}"
    print(f"[ok] mid-capture death detected: {st['interrupt_msg']}")
    time.sleep(2.5)                            # daemon recovers + rcc re-registers
    assert state["controller"] == me, "should have re-registered after recovery"
    assert rcc.capture_interrupted(), "interrupt flag must stay sticky after recovery"
    print("[ok] interrupt sticky through daemon recovery + re-register")
    rcc.arm(syncMode=False, fps=30)                      # new session clears it
    rcc.set_record("dataset/003", on=True)
    time.sleep(0.3)
    assert not rcc.capture_interrupted(), "new start() should clear interrupt flag"
    print("[ok] new start() cleared the interrupt flag")
    rcc.stop(); time.sleep(0.2)

    t0 = time.time()
    rcc.end()                       # must not hang
    assert time.time() - t0 < 8, "end() hung"
    assert not rcc.run_thread.is_alive(), "run thread did not exit"
    print(f"[ok] end clean in {time.time()-t0:.2f}s; total heartbeats={state['hb']}")

    stop.set()
    time.sleep(0.2)
    print("\nALL PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e); sys.exit(1)
    except Exception as e:
        import traceback; traceback.print_exc(); sys.exit(2)
