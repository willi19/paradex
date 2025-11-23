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
from typing import Dict, Any, Optional, Callable, List

from paradex.utils.system import get_pc_list, get_pc_ip

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

    def send_data(self, metadata: List[Dict[str, Any]], data: List[bytes]) -> None:
        """
        Publish data using multipart messages for efficient binary transmission.
        
        Args:
            metadata: Metadata dictionary for each camera (frame_id, shape, image_index)
            data: List of binary image data (JPEG bytes)
        """
        
        message = {
            'timestamp': datetime.now().isoformat(),
            'publisher': self.name,
            'items': metadata
        }
        
        # Send as multipart: [topic, metadata_json, image1_bytes, image2_bytes, ...]
        message_parts = [b'data', json.dumps(message).encode('utf-8')]
        message_parts.extend(data)
        
        self.socket.send_multipart(message_parts)
    
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
        self.latest_data = {}
        
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
        while self.collecting:
            socks = dict(self.poller.poll(timeout=100))
            
            for pc_name, socket in self.sockets.items():
                if socket in socks:
                    try:
                        parts = socket.recv_multipart(flags=zmq.NOBLOCK)
                        
                        if len(parts) < 2:
                            continue
                        
                        metadata = json.loads(parts[1].decode('utf-8'))
                        
                        # 각 item을 name 기준으로 저장
                        items = metadata.get('items', [])
                        for item in items:
                            if 'data_index' in item:
                                idx = item.pop('data_index')
                                if idx + 2 < len(parts):
                                    item['data'] = parts[idx + 2]
                            
                            # name을 key로 저장!
                            item_name = item.get('name')
                            if item_name:
                                item['pc'] = pc_name  # 어느 PC에서 왔는지 기록
                                item['timestamp'] = metadata.get('timestamp')
                                item['publisher'] = metadata.get('publisher')
                                
                                self.latest_data[item_name] = item
                        
                    except zmq.Again:
                        pass
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