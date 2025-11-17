import threading
import zmq
import time
import traceback

from paradex.io.camera_system.camera_loader import CameraLoader

class camera_server_daemon:
    def __init__(self):
        self.camera_loader = CameraLoader()
        self.camera_loader.load_pyspin_camera()
        
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

            except:
                return {"status":"error", "msg":"start failed"}

        if action == "stop":
            try:
                self.camera_loader.stop()
                return {"status":"ok", "msg":"stopped"}
            except:
                return {"status":"error", "msg":"stop failed"}

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

        return {"status":"error", "msg":"unknown action"}


    def command_thread(self):
        self.command_socket = self.ctx.socket(zmq.REP)
        self.command_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.command_socket.bind(f"tcp://*:{self.command_port}")
        
        while True:
            try:
                cmd = self.command_socket.recv_json()
                print("cmd:", cmd)
                response = self.execute_command(cmd)
                    
                self.command_socket.send_json(response)
                print("response:", response)

            except zmq.Again:
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
                print("response": response)