import zmq
import threading
from typing import Dict, Optional, Any, Callable
from datetime import datetime

from paradex.utils.system import get_pc_list, get_pc_ip

class CommandSender:
    """
    Send commands to multiple PCs using REQ-REP pattern.
    
    Usage:
        sender = CommandSender(
            pc_info={"pc1": {"ip": "192.168.1.10"}},
            port=5482
        )
        
        sender.send_command({'action': 'start'})
    """
    
    def __init__(
        self,
        pc_list: Optional[list] = None,
        port: int = 6890,
        timeout: int = 60000,
    ):
        self.pc_list = pc_list or get_pc_list()
        self.port = port
        
        # ZMQ setup
        self.context = zmq.Context()
        self.sockets = {}
        
        # Create sockets
        for pc_name in self.pc_list:
            ip = get_pc_ip(pc_name)
            socket = self.context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, timeout)
            socket.setsockopt(zmq.SNDTIMEO, timeout)
            socket.connect(f"tcp://{ip}:{self.port}")
            
            self.sockets[pc_name] = socket
            print(f"[{pc_name}] Connected to {ip}:{self.port}")

    def _send_to_pc(self, pc_name: str, cmd: str, wait=False, cmd_info={}) -> None:
        """Send command to single PC."""
        socket = self.sockets[pc_name]
        cmd_dict = {
            "command": cmd,
            "is_wait": wait,
            "info": cmd_info,
        }
        try:
            socket.send_json(cmd_dict)
            response = socket.recv_json()
            
            if response.get("state") == "error":
                print(f"[{pc_name}] Error: {response.get('message')}")
            else:
                print(f"[{pc_name}] {response.get('message')}")
                
        except zmq.ZMQError as e:
            print(f"[{pc_name}] Error: {e}")
    
    def send_command(self, cmd: str, wait=False, cmd_info = {}) -> None:
        """Send command to all PCs and wait for responses."""
        
        threads = []
        for pc_name in self.sockets.keys():
            thread = threading.Thread(
                target=self._send_to_pc,
                args=(pc_name, cmd, wait, cmd_info),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
    
    def end(self):
        self.send_command('exit')
        
        for socket in self.sockets.values():
            socket.close()
        self.context.term()
        
class CommandReceiver:
    def __init__(
        self,
        event_dict: Optional[Dict[str, threading.Event]] = None,
        port: int = 6890,
    ):
        self.port = port
        self.event_dict = event_dict or {}
        self.event_info = {}
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        
        # Threading
        self.thread = None
        self.exit_event = threading.Event()
        
        print(f"[CommandReceiver] Listening on port {self.port}")
        
        self.start()
    
    def _recv_loop(self) -> None:
        """Main loop for receiving commands."""
        while not self.exit_event.is_set():
            try:
                # Receive command
                cmd_dict = self.socket.recv_json(flags=zmq.NOBLOCK)
                
                is_wait = cmd_dict.get("is_wait", False)
                cmd = cmd_dict.get("command", "")
                cmd_info = cmd_dict.get("info", {})

                # Execute callback
                if cmd in self.event_dict:
                    try:
                        self.event_dict[cmd].set()
                        self.event_info[cmd] = cmd_info
                        if is_wait:
                            self.event_dict[cmd].wait()
                        response = {"state": "success", "message": f"Command '{cmd}' executed"}
                    except Exception as e:
                        response = {"state": "error", "message": f"Callback error: {str(e)}"}
                else:
                    response = {"state": "error", "message": f"Unknown command: {cmd}"}
                
                # Send response
                self.socket.send_json(response)
                
            except zmq.Again:
                # No message available
                threading.Event().wait(0.01)  # Small sleep
                
            except Exception as e:
                print(f"[CommandReceiver] Error: {e}")
                try:
                    self.socket.send_json({"state": "error", "message": str(e)})
                except:
                    pass
    
    def start(self) -> None:
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()
        print("[CommandReceiver] Started")
        
    def end(self) -> None:
        """Clean up resources."""
        self.thread.join(timeout=2)
        self.socket.close()
        self.context.term()
        print("[CommandReceiver] Closed")