"""
Modbus RTU communication protocol for the Inspire Hand.
"""

import serial
import time
from enum import IntEnum
from typing import List, Optional, Union, Tuple
from exceptions import ConnectionError, CommandError


class ModbusFunction(IntEnum):
    """Modbus function codes"""
    READ_HOLDING_REGISTERS = 0x03
    WRITE_SINGLE_REGISTER = 0x06
    WRITE_MULTIPLE_REGISTERS = 0x10


class ModbusClient:
    """Low-level Modbus RTU client for communication with the Inspire Hand."""
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200, slave_id: int = 1):
        """
        Initialize the Modbus RTU client.
        
        Args:
            port: Serial port path
            baudrate: Serial baudrate
            slave_id: Modbus slave ID of the device
        """
        self.port = port
        self.baudrate = baudrate
        self.slave_id = slave_id
        self.ser = None
        self.debug = False
    
    def connect(self) -> bool:
        """
        Connect to the device.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            ConnectionError: If there's an issue connecting to the device
        """
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            
            if self.ser.is_open:
                if self.debug:
                    print(f"Connected to device on {self.port}")
                return True
            return False
        except Exception as e:
            raise ConnectionError(f"Failed to connect: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from the device."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            if self.debug:
                print("Disconnected from device")
    
    def _calculate_crc(self, data: bytearray) -> bytes:
        """
        Calculate Modbus CRC16. Checking error
        
        Args:
            data: Data to calculate CRC for
            
        Returns:
            CRC as bytes (little-endian)
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc = crc >> 1
        return crc.to_bytes(2, byteorder='little')
    
    def read_holding_registers(self, address: int, num_registers: int = 1) -> Optional[List[int]]:
        """
        Read holding registers (Modbus function 0x03).
        
        Args:
            address: Starting register address
            num_registers: Number of registers to read
            
        Returns:
            List of register values if successful, None otherwise
            
        Raises:
            ConnectionError: If not connected to device
            CommandError: If command fails
        """
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("Not connected to device")
        
        # Create Modbus RTU frame for reading registers
        packet = bytearray([
            self.slave_id,                  # Slave ID
            ModbusFunction.READ_HOLDING_REGISTERS,  # Function code
            (address >> 8) & 0xFF,          # Address high byte
            address & 0xFF,                 # Address low byte
            (num_registers >> 8) & 0xFF,    # Number of registers high byte
            num_registers & 0xFF            # Number of registers low byte
        ])
        
        # Add CRC
        packet.extend(self._calculate_crc(packet))
        
        # Send packet
        if self.debug:
            print(f"Sending read request: {' '.join([f'0x{b:02X}' for b in packet])}")
        
        try:
            self.ser.write(packet)
            time.sleep(0.2)  # Wait for response
            
            # Check for response
            if self.ser.in_waiting:
                response = self.ser.read(self.ser.in_waiting)
                if self.debug:
                    print(f"Response: {' '.join([f'0x{b:02X}' for b in response])}")
                
                # Parse the response
                if len(response) >= 5 and response[0] == self.slave_id and response[1] == ModbusFunction.READ_HOLDING_REGISTERS:
                    byte_count = response[2]
                    values = []
                    
                    for i in range(0, byte_count, 2):
                        if i + 3 < len(response):
                            value = (response[i+3] << 8) | response[i+4]
                            values.append(value)
                            
                    return values
                else:
                    raise CommandError("Invalid response")
            else:
                raise CommandError("No response received")
        except Exception as e:
            raise CommandError(f"Error reading registers: {e}")
    
    def write_single_register(self, address: int, value: int) -> bool:
        """
        Write to a single holding register (Modbus function 0x06).
        
        Args:
            address: Register address
            value: Value to write (0-65535)
            
        Returns:
            True if successful
            
        Raises:
            ConnectionError: If not connected to device
            CommandError: If command fails
        """
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("Not connected to device")
        
        # Create Modbus RTU frame for writing a register
        packet = bytearray([
            self.slave_id,                  # Slave ID
            ModbusFunction.WRITE_SINGLE_REGISTER,  # Function code
            (address >> 8) & 0xFF,          # Address high byte
            address & 0xFF,                 # Address low byte
            (value >> 8) & 0xFF,            # Value high byte
            value & 0xFF                    # Value low byte
        ])
        
        # Add CRC
        packet.extend(self._calculate_crc(packet))
        
        # Send packet
        if self.debug:
            print(f"Sending write request: {' '.join([f'0x{b:02X}' for b in packet])}")
        
        try:
            self.ser.write(packet)
            time.sleep(0.2)  # Wait for response
            
            # Check for response
            if self.ser.in_waiting:
                response = self.ser.read(self.ser.in_waiting)
                if self.debug:
                    print(f"Response: {' '.join([f'0x{b:02X}' for b in response])}")
                
                # Check if response is valid
                if len(response) >= 8 and response[0] == self.slave_id and response[1] == ModbusFunction.WRITE_SINGLE_REGISTER:
                    return True
                else:
                    raise CommandError("Invalid response")
            else:
                raise CommandError("No response received")
        except Exception as e:
            raise CommandError(f"Error writing register: {e}")
    
    def write_multiple_registers(self, address: int, values: List[int]) -> bool:
        """
        Write to multiple holding registers (Modbus function 0x10).
        
        Args:
            address: Starting register address
            values: List of values to write
            
        Returns:
            True if successful
            
        Raises:
            ConnectionError: If not connected to device
            CommandError: If command fails
        """
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("Not connected to device")
        
        # Create Modbus RTU frame for writing multiple registers
        num_registers = len(values)
        byte_count = num_registers * 2
        
        packet = bytearray([
            self.slave_id,                  # Slave ID
            ModbusFunction.WRITE_MULTIPLE_REGISTERS,  # Function code
            (address >> 8) & 0xFF,          # Address high byte
            address & 0xFF,                 # Address low byte
            (num_registers >> 8) & 0xFF,    # Number of registers high byte
            num_registers & 0xFF,           # Number of registers low byte
            byte_count                      # Byte count
        ])
        
        # Add data values
        for value in values:
            packet.append((value >> 8) & 0xFF)  # Value high byte
            packet.append(value & 0xFF)         # Value low byte
        
        # Add CRC
        packet.extend(self._calculate_crc(packet))
        
        # Send packet
        if self.debug:
            print(f"Sending write multiple request: {' '.join([f'0x{b:02X}' for b in packet])}")
        
        try:
            self.ser.write(packet)
            time.sleep(0.2)  # Wait for response
            
            # Check for response
            if self.ser.in_waiting:
                response = self.ser.read(self.ser.in_waiting)
                if self.debug:
                    print(f"Response: {' '.join([f'0x{b:02X}' for b in response])}")
                
                # Check if response is valid
                if len(response) >= 8 and response[0] == self.slave_id and response[1] == ModbusFunction.WRITE_MULTIPLE_REGISTERS:
                    return True
                else:
                    raise CommandError("Invalid response")
            else:
                raise CommandError("No response received")
        except Exception as e:
            raise CommandError(f"Error writing registers: {e}") 