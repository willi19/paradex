"""
Simplified data publishing system using PUB-SUB pattern.

Much simpler than ROUTER-DEALER:
- No registration handshake needed
- Just publish and subscribe
- Fire and forget
"""

import zmq
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable

from paradex.utils.system import get_pc_list

class DataPublisher():
    def __init__(self, port: int=1234, name: Optional[str] = None):
        """
        Initialize data publisher.
        
        Args:
            port: Port number to bind the PUB socket
            name: Optional identifier for this publisher
        """
        self.port = port
        self.name = name or f"publisher_{port}"
        
        # ZMQ PUB socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")
        
        # Publishing control
        self.publishing = False
        self.publish_thread = None
        
        print(f"[{self.name}] Publisher started on port {self.port}")
        
        # Give subscribers time to connect
        time.sleep(0.1)
    
    def send_data(self, data: Dict[str, Any]) -> None:
        """
        Publish data.
        
        Args:
            data: Data dictionary to send
        """
        message = {
            'timestamp': datetime.now().isoformat(),
            'publisher': self.name,
            'data': data
        }
        self.socket.send_json(message)
    
    def close(self) -> None:
        """Clean up resources."""
        time.sleep(0.1)  # Let message send
        self.socket.close()
        self.context.term()
        print(f"[{self.name}] Closed")

class DataCollector:
    
    def __init__(
        self,
        pc_list: Dict[str, Dict[str, str]] = None,
        port: int = 1234
    ):
        """
        Initialize data collector.
        
        Args:
            port: Port to connect to on each client PC
        """
        self.pc_list = pc_list or get_pc_list()
        self.port = port
        
        # ZMQ setup
        self.context = zmq.Context()
        self.sockets = {}
        self.poller = zmq.Poller()
        
        # Data storage
        self.latest_data = {pc: None for pc in self.pc_list}
        
        # Collection control
        self.collecting = False
        self.collection_thread = None
        
        # Initialize sockets
        self._init_sockets()
    
    def _init_sockets(self) -> None:
        """Initialize SUB sockets to all client PCs."""
        for pc_name in self.pc_list:
            ip = get_pc_ip(pc_name)
            
            socket = self.context.socket(zmq.SUB)
            socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all messages
            socket.connect(f"tcp://{ip}:{self.port}")
            
            self.sockets[pc_name] = socket
            self.poller.register(socket, zmq.POLLIN)
            
            print(f"[Collector] Subscribed to {pc_name} at {ip}:{self.port}")
    
    def _collection_loop(self) -> None:
        """Main loop for collecting data from all PCs."""
        while self.collecting:
            # Poll all sockets with timeout
            socks = dict(self.poller.poll(timeout=100))  # 100ms timeout
            
            for pc_name, socket in self.sockets.items():
                if socket in socks:
                    try:
                        data = socket.recv_json(flags=zmq.NOBLOCK)                        
                        self.latest_data[pc_name] = data
                        
                    except zmq.Again:
                        pass
                    except json.JSONDecodeError as e:
                        print(f"[Collector] JSON error from {pc_name}: {e}")
                    except Exception as e:
                        print(f"[Collector] Error from {pc_name}: {e}")
    
    def start(self) -> None:
        """Start collecting data from all PCs."""
        if self.collecting:
            print("[Collector] Already collecting")
            return
        
        self.collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        print("[Collector] Started")
    
    def stop(self) -> None:
        """Stop collecting data."""
        if not self.collecting:
            return
        
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2)
    
    def get_data(self, pc_name: Optional[str] = None) -> Any:
        """
        Get the latest received data.
        
        Args:
            pc_name: Specific PC to get data from, or None for all PCs
            
        Returns:
            Latest data from specified PC, or dict of all latest data
        """
        if pc_name:
            return self.latest_data.get(pc_name)
        return self.latest_data.copy()
    
    def end(self) -> None:
        """Clean up resources."""
        self.stop()
        
        for socket in self.sockets.values():
            socket.close()
        
        self.context.term()
        print("[Collector] Closed")