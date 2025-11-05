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

        self.ctx = zmq.Context()

        self.current_controller = None

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

    def command_thread(self):
        self.command_socket = self.ctx.socket(zmq.REP)
        self.command_socket.bind(f"tcp://*:{self.command_port}")
        
        while True:
            try:
                cmd = self.command_socket.recv_json()
                action = cmd.get('action')
                controller_name = cmd.get('controller_name')
                
                if self.current_controller is None:
                    if action == 'start':
                        self.current_controller = controller_name
                        print(f"[Controller] {controller_name} connected")
                        print(f"[Current Controller] {self.current_controller}")
                        
                        self.camera_loader.start(
                            cmd.get('mode'),
                            cmd.get('syncMode'),
                            cmd.get('save_path'),
                            cmd.get('fps', 30)
                        )
                        self.command_socket.send_json({'status': 'ok', 'msg': 'started'})
                    else:
                        self.command_socket.send_json({'status': 'error', 'msg': 'no active controller'})
                
                elif controller_name == self.current_controller:
                    print(f"[Command] {action} from {controller_name}")
                    
                    if action == 'start':
                        self.camera_loader.start(
                            cmd.get('mode'),
                            cmd.get('syncMode'),
                            cmd.get('save_path'),
                            cmd.get('fps', 30)
                        )
                        self.command_socket.send_json({'status': 'ok', 'msg': 'started'})
                    
                    elif action == 'stop':
                        self.camera_loader.stop()
                        self.command_socket.send_json({'status': 'ok', 'msg': 'stopped'})
                    
                    elif action == 'exit':
                        print(f"[Controller] {controller_name} disconnected")
                        self.current_controller = None
                        print(f"[Current Controller] None")
                        self.command_socket.send_json({'status': 'ok', 'msg': 'exited'})
                    
                    else:
                        self.command_socket.send_json({'status': 'error', 'msg': 'unknown action'})
                
                else:
                    print(f"[Warning] {controller_name} tried to access, but locked by {self.current_controller}")
                    self.command_socket.send_json({
                        'status': 'error', 
                        'msg': f'controller locked by {self.current_controller}'
                    })

            except Exception as e:
                print(f"="*60)
                print(f"[ERROR] Command thread exception occurred:")
                print(f"Exception Type: {type(e).__name__}")
                print(f"Exception Message: {str(e)}")
                print(f"-"*60)
                print("Traceback:")
                traceback.print_exc()
                print(f"="*60)
                
                try:
                    self.command_socket.send_json({
                        'status': 'error', 
                        'msg': f'{type(e).__name__}: {str(e)}'
                    })
                except:
                    print("[ERROR] Failed to send error response to client")
                
                self.current_controller = None