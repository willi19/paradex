import threading
import zmq
import time
import traceback

from paradex.io.camera_system.camera_loader import CameraLoader

class camera_server_daemon:
    def __init__(self):
        self.camera_loader = CameraLoader()
        
        self.ping_port = 5480
        self.monitor_port = 5481
        self.command_port = 5482
        self.connection_port = 5483

        self.ctx = zmq.Context()

        self.current_controller = None
        
        self.state = "idle"

        threading.Thread(target=self.pingpong_thread, daemon=True).start()
        threading.Thread(target=self.monitor_thread, daemon=True).start()
        threading.Thread(target=self.command_thread, daemon=True).start()

    def reload_cameras(self):
        self.camera_loader.end()
        time.sleep(1)
        
        self.camera_loader = CameraLoader()
        print("[Info] Camera loader reloaded.")
        
    def pingpong_thread(self):
        self.ping_socket = self.ctx.socket(zmq.REP)
        self.ping_socket.setsockopt(zmq.LINGER, 0)
        self.ping_socket.bind(f"tcp://*:{self.ping_port}")

        while True:
            try:
                _ = self.ping_socket.recv_string(flags=zmq.NOBLOCK)
                self.ping_socket.send_string("pong")
            except zmq.ZMQError:
                time.sleep(0.1)

    def connection_thread(self):
        self.connection_socket = self.ctx.socket(zmq.REP)
        self.connection_socket.bind(f"tcp://*:{self.connection_port}")

        while True:
            try:
                _ = self.connection_socket.recv_string(flags=zmq.NOBLOCK)
                self.connection_socket.send_string("connected")
            except zmq.ZMQError:
                time.sleep(0.1)

    def monitor_thread(self):
        monitor_socket = self.ctx.socket(zmq.PUB)
        monitor_socket.bind(f"tcp://*:{self.monitor_port}")
        
        while True:
            status = {
                'cameras': self.camera_loader.get_status_list(),
                'controller': self.current_controller if self.current_controller else 'None'
            }
            monitor_socket.send_json(status)
            time.sleep(0.1)

    def execute_command(self, cmd):
        action = cmd.get('action')
        controller_name = cmd.get('controller_name')

        # Recording is a side-channel: it only opens/closes the per-camera
        # VideoWriter on already-running cameras and never touches the
        # acquisition lifecycle or shared memory. So it intentionally bypasses
        # the single-owner lock — a record-only controller can toggle .avi
        # while a separate "stream owner" controller holds the lock and keeps
        # the daemon alive via heartbeat.
        if action == "record_start":
            try:
                self.camera_loader.record_start(cmd.get('save_path'), cmd.get('fps', 30))
                return {"status": "ok", "msg": "recording started"}
            except Exception as e:
                return {"status": "error", "msg": f"record_start failed: {e}"}

        if action == "record_stop":
            try:
                self.camera_loader.record_stop()
                return {"status": "ok", "msg": "recording stopped"}
            except Exception as e:
                return {"status": "error", "msg": f"record_stop failed: {e}"}

        if controller_name != self.current_controller and self.current_controller is not None:
            print(f"[Warning] {controller_name} tried to access, but locked by {self.current_controller}")
            return {"status":"error", "msg":f"locked by {self.current_controller}"}
        
        if action == "register":
            self.current_controller = controller_name
            return {"status":"ok", "msg":"registered"}

        if self.current_controller is None:
            return {"status":"error", "msg":"no active controller"}
        
        if action == "start":
            try:
                self.camera_loader.start(
                            cmd.get('mode'),
                            cmd.get('syncMode'),
                            cmd.get('save_path'),
                            cmd.get('fps', 30)
                        )
                
                return {"status":"ok", "msg":"started"}

            except Exception as e:
                tb = traceback.format_exc()
                print("[ERROR] camera_loader.start failed:")
                print(tb)
                return {"status":"error", "msg":f"start failed: {e}"}

        if action == "stop":
            try:
                self.camera_loader.stop()
                return {"status":"ok", "msg":"stopped"}
            except Exception as e:
                tb = traceback.format_exc()
                print("[ERROR] camera_loader.stop failed:")
                print(tb)
                return {"status":"error", "msg":f"stop failed: {e}"}

        if action == "end":
            try:
                self.current_controller = None
                return {"status":"ok", "msg":"ended"}
            except:
                return {"status":"error", "msg":"end failed"}

        if action == "heartbeat":
            if len(self.camera_loader.get_all_errors()) == 0:
                return {"status":"ok", "msg":"heartbeat received"}
            else:
                return {"status":"error", "msg":"camera errors detected"}

        if action == "reload":
            try:
                self.reload_cameras()
                return {"status":"ok", "msg":"cameras reloaded"}
            except:
                return {"status":"error", "msg":"reload failed"}
            
        return {"status":"error", "msg":"unknown action"}


    def command_thread(self):
        self.command_socket = self.ctx.socket(zmq.REP)
        self.command_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.command_socket.bind(f"tcp://*:{self.command_port}")
        
        while True:
            try:
                cmd = self.command_socket.recv_json()
                response = self.execute_command(cmd)
                    
                self.command_socket.send_json(response)
                
            except zmq.Again:
                if self.current_controller is not None:
                    self.camera_loader.stop()
                    self.current_controller = None
                    print("[Error] Command socket timeout. Camera loader stopped and controller released.")
                    
            except Exception as e:
                self.camera_loader.stop()
                self.current_controller = None
                
                traceback.print_exc()
                self.command_socket.send_json({
                    'status': 'error', 
                    'msg': f'{type(e).__name__}: {str(e)} traceback : {traceback.format_exc()}'
                })
                print("[Error] Exception in command thread. Camera loader stopped and controller released.")
                print("response: ", response)