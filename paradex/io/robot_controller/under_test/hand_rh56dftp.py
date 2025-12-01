"""
Main module for controlling the Inspire Hand RH56DFTP.
Based on the RH56DFTP User Manual V1.0.0.
"""

from enum import IntEnum
from typing import List, Optional, Dict, Tuple
import time
from contextlib import contextmanager

from .modbus import ModbusClient
from .exceptions import InspireHandError, ConnectionError, CommandError


class RegisterRH56DFTP:
    """
    Register addresses for the Inspire Hand RH56DFTP.
    
    Note: These addresses should be verified against the RH56DFTP User Manual V1.0.0.
    Update these values based on the actual register map from the manual.
    """
    # System information registers
    HAND_ID = 1000        # Dexterous Hand ID
    CLEAR_ERROR = 1004    # Error clearance
    SAVE = 1005           # Saving data in Flash
    RESET_PARA = 1006     # Restoring factory defaults
    FORCE_SENSOR_CALIB = 1009  # Force sensor calibration
    
    # Control registers - Pose/Position setting
    # These addresses need to be verified from the manual
    POSE_SET = 1474       # Pose/Position set values (for all joints)
    ANGLE_SET = 1486      # Angle set values (for all joints)
    FORCE_SET = 1498      # Force control threshold values
    SPEED_SET = 1522      # Speed values
    
    # Status registers - Current pose reading
    POSE_ACT = 1534       # Actual pose/position values
    ANGLE_ACT = 1546      # Actual angle values
    FORCE_ACT = 1582      # Actual force values
    CURRENT = 1594        # Actuator current values
    ERROR = 1606          # Error codes
    STATUS = 1612         # Status information
    TEMP = 1618           # Actuator temperature values
    
    # Contact information registers
    # These addresses need to be verified from the manual
    # Contact sensors typically provide information about which fingers are in contact
    CONTACT_INFO = 1624   # Contact information (which fingers are touching)
    CONTACT_FORCE = 1630  # Contact force values for each finger
    CONTACT_STATUS = 1636 # Contact status flags


class FingerID(IntEnum):
    """Finger IDs for the Inspire Hand RH56DFTP."""
    LITTLE = 0
    RING = 1
    MIDDLE = 2
    INDEX = 3
    THUMB_BEND = 4
    THUMB_ROTATE = 5
    ALL = 6  # Special ID for controlling all fingers at once


class FingerStatus(IntEnum):
    """Status codes for fingers."""
    UNCLENCHING = 0
    GRASPING = 1
    REACHED_TARGET = 2
    REACHED_FORCE = 3
    CURRENT_PROTECTION = 5
    LOCKED_ROTOR = 6
    FAULT = 7


class ContactStatus:
    """Contact status information for fingers."""
    
    def __init__(self, contact_flags: int, contact_forces: List[int]):
        """
        Initialize contact status.
        
        Args:
            contact_flags: Bit flags indicating which fingers are in contact
            contact_forces: List of contact force values for each finger
        """
        self.contact_flags = contact_flags
        self.contact_forces = contact_forces
    
    def is_contact(self, finger_id: int) -> bool:
        """
        Check if a specific finger is in contact.
        
        Args:
            finger_id: Finger ID (0-5)
            
        Returns:
            True if finger is in contact, False otherwise
        """
        if not 0 <= finger_id <= 5:
            return False
        return bool(self.contact_flags & (1 << finger_id))
    
    def get_contact_force(self, finger_id: int) -> int:
        """
        Get contact force for a specific finger.
        
        Args:
            finger_id: Finger ID (0-5)
            
        Returns:
            Contact force value for the finger
        """
        if not 0 <= finger_id <= 5:
            return 0
        return self.contact_forces[finger_id] if finger_id < len(self.contact_forces) else 0
    
    def get_all_contacts(self) -> Dict[int, bool]:
        """
        Get contact status for all fingers.
        
        Returns:
            Dictionary mapping finger ID to contact status
        """
        return {i: self.is_contact(i) for i in range(6)}
    
    def __repr__(self) -> str:
        contacts = [f"F{i}" for i in range(6) if self.is_contact(i)]
        return f"ContactStatus(contacts={contacts}, forces={self.contact_forces})"


class InspireHandRH56DFTP:
    """
    Interface for controlling the Inspire Hand RH56DFTP.
    
    This class provides functionality for:
    - Connection management
    - Reading current pose
    - Writing/executing pose
    - Reading contact information
    """
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200, 
                 slave_id: int = 1, debug: bool = False):
        """
        Initialize the Inspire Hand RH56DFTP interface.
        
        Args:
            port: Serial port path
            baudrate: Serial baudrate
            slave_id: Modbus slave ID of the hand
            debug: Whether to print debug information
        """
        self.modbus = ModbusClient(port, baudrate, slave_id)
        self.modbus.debug = debug
        self._connected = False
    
    @contextmanager
    def connect(self):
        """
        Context manager for connecting to the hand.
        
        Example:
            with hand.connect():
                pose = hand.read_pose()
                hand.write_pose([1000, 1000, 1000, 1000, 1000, 1000])
        """
        try:
            self.open()
            yield self
        finally:
            self.close()
    
    def open(self) -> None:
        """
        Connect to the hand.
        
        Raises:
            ConnectionError: If connection fails
        """
        if not self._connected:
            self.modbus.connect()
            self._connected = True
            if self.modbus.debug:
                print(f"Connected to Inspire Hand RH56DFTP on {self.modbus.port}")
    
    def close(self) -> None:
        """Disconnect from the hand."""
        if self._connected:
            self.modbus.disconnect()
            self._connected = False
            if self.modbus.debug:
                print("Disconnected from Inspire Hand RH56DFTP")
    
    @property
    def is_connected(self) -> bool:
        """Check if the hand is connected."""
        return self._connected
    
    def _check_connection(self) -> None:
        """Check if the hand is connected, raise an exception if not."""
        if not self._connected:
            raise ConnectionError("Not connected to hand. Call open() first.")
    
    # Pose reading and writing functions
    
    def read_pose(self) -> List[int]:
        """
        Read the current pose of all joints.
        
        Returns:
            List of current pose values for all joints (typically 6 values for 6 DOF)
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
        """
        self._check_connection()
        try:
            # Read actual pose values from registers
            # Adjust the number of registers based on the actual number of joints
            pose_values = self.modbus.read_holding_registers(RegisterRH56DFTP.POSE_ACT, 6)
            if pose_values is None:
                raise CommandError("Failed to read pose from hand")
            return pose_values
        except Exception as e:
            raise CommandError(f"Error reading pose: {e}")
    
    def read_angles(self) -> List[int]:
        """
        Read the current angles of all joints.
        
        Returns:
            List of current angle values for all joints (0-1000, 0=closed, 1000=open)
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
        """
        self._check_connection()
        try:
            angle_values = self.modbus.read_holding_registers(RegisterRH56DFTP.ANGLE_ACT, 6)
            if angle_values is None:
                raise CommandError("Failed to read angles from hand")
            return angle_values
        except Exception as e:
            raise CommandError(f"Error reading angles: {e}")
    
    def write_pose(self, pose: List[int]) -> None:
        """
        Write/execute a pose for all joints.
        
        Args:
            pose: List of pose values for all joints (typically 6 values)
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
            ValueError: If pose values are invalid
        """
        self._check_connection()
        
        if not isinstance(pose, list):
            raise ValueError("Pose must be a list")
        
        if len(pose) != 6:
            raise ValueError(f"Pose must contain 6 values, got {len(pose)}")
        
        # Validate pose values (typically 0-1000 range)
        for i, value in enumerate(pose):
            if not isinstance(value, int):
                raise ValueError(f"Pose value at index {i} must be an integer")
            if not 0 <= value <= 1000:
                raise ValueError(f"Pose value at index {i} must be between 0 and 1000, got {value}")
        
        try:
            # Write pose values to registers
            success = self.modbus.write_multiple_registers(RegisterRH56DFTP.POSE_SET, pose)
            if not success:
                raise CommandError("Failed to write pose to hand")
        except Exception as e:
            raise CommandError(f"Error writing pose: {e}")
    
    def write_angles(self, angles: List[int]) -> None:
        """
        Write/execute angles for all joints.
        
        Args:
            angles: List of angle values for all joints (0-1000, 0=closed, 1000=open)
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
            ValueError: If angle values are invalid
        """
        self._check_connection()
        
        if not isinstance(angles, list):
            raise ValueError("Angles must be a list")
        
        if len(angles) != 6:
            raise ValueError(f"Angles must contain 6 values, got {len(angles)}")
        
        # Validate angle values
        for i, value in enumerate(angles):
            if not isinstance(value, int):
                raise ValueError(f"Angle value at index {i} must be an integer")
            if not 0 <= value <= 1000:
                raise ValueError(f"Angle value at index {i} must be between 0 and 1000, got {value}")
        
        try:
            # Write angle values to registers
            success = self.modbus.write_multiple_registers(RegisterRH56DFTP.ANGLE_SET, angles)
            if not success:
                raise CommandError("Failed to write angles to hand")
        except Exception as e:
            raise CommandError(f"Error writing angles: {e}")
    
    def set_joint_angle(self, joint_id: int, angle: int) -> None:
        """
        Set the angle of a specific joint.
        
        Args:
            joint_id: Joint ID (0-5)
            angle: Angle value (0-1000, 0=closed, 1000=open)
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
            ValueError: If joint_id or angle is invalid
        """
        self._check_connection()
        
        if not 0 <= joint_id <= 5:
            raise ValueError(f"Invalid joint ID. Must be 0-5, got {joint_id}")
        
        if not 0 <= angle <= 1000:
            raise ValueError(f"Invalid angle. Must be 0-1000, got {angle}")
        
        try:
            register_address = RegisterRH56DFTP.ANGLE_SET + (joint_id * 2)
            success = self.modbus.write_single_register(register_address, angle)
            if not success:
                raise CommandError(f"Failed to set angle for joint {joint_id}")
        except Exception as e:
            raise CommandError(f"Error setting joint angle: {e}")
    
    # Contact information reading functions
    
    def read_contact_info(self) -> ContactStatus:
        """
        Read contact information from the hand.
        
        This function reads which fingers are in contact and their contact forces.
        
        Returns:
            ContactStatus object containing contact flags and forces
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
        """
        self._check_connection()
        
        try:
            # Read contact status flags
            contact_flags_data = self.modbus.read_holding_registers(
                RegisterRH56DFTP.CONTACT_STATUS, 1
            )
            if contact_flags_data is None:
                raise CommandError("Failed to read contact status from hand")
            
            contact_flags = contact_flags_data[0] if contact_flags_data else 0
            
            # Read contact force values
            contact_forces = self.modbus.read_holding_registers(
                RegisterRH56DFTP.CONTACT_FORCE, 6
            )
            if contact_forces is None:
                raise CommandError("Failed to read contact forces from hand")
            
            return ContactStatus(contact_flags, contact_forces)
            
        except Exception as e:
            raise CommandError(f"Error reading contact info: {e}")
    
    def is_finger_in_contact(self, finger_id: int) -> bool:
        """
        Check if a specific finger is in contact.
        
        Args:
            finger_id: Finger ID (0-5)
            
        Returns:
            True if finger is in contact, False otherwise
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
        """
        contact_info = self.read_contact_info()
        return contact_info.is_contact(finger_id)
    
    def get_contact_force(self, finger_id: int) -> int:
        """
        Get contact force for a specific finger.
        
        Args:
            finger_id: Finger ID (0-5)
            
        Returns:
            Contact force value for the finger
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
        """
        contact_info = self.read_contact_info()
        return contact_info.get_contact_force(finger_id)
    
    def get_all_contact_status(self) -> Dict[int, bool]:
        """
        Get contact status for all fingers.
        
        Returns:
            Dictionary mapping finger ID to contact status
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
        """
        contact_info = self.read_contact_info()
        return contact_info.get_all_contacts()
    
    # Additional utility functions
    
    def set_speed(self, speed: int) -> None:
        """
        Set the speed for all joints.
        
        Args:
            speed: Speed value (0-1000)
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
            ValueError: If speed is invalid
        """
        self._check_connection()
        
        if not 0 <= speed <= 1000:
            raise ValueError(f"Invalid speed. Must be 0-1000, got {speed}")
        
        try:
            values = [speed] * 6
            success = self.modbus.write_multiple_registers(RegisterRH56DFTP.SPEED_SET, values)
            if not success:
                raise CommandError("Failed to set speed")
        except Exception as e:
            raise CommandError(f"Error setting speed: {e}")
    
    def set_force_threshold(self, force: int) -> None:
        """
        Set the force threshold for all joints.
        
        Args:
            force: Force threshold value (0-1000)
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
            ValueError: If force is invalid
        """
        self._check_connection()
        
        if not 0 <= force <= 1000:
            raise ValueError(f"Invalid force. Must be 0-1000, got {force}")
        
        try:
            values = [force] * 6
            success = self.modbus.write_multiple_registers(RegisterRH56DFTP.FORCE_SET, values)
            if not success:
                raise CommandError("Failed to set force threshold")
        except Exception as e:
            raise CommandError(f"Error setting force threshold: {e}")
    
    def reset(self) -> None:
        """
        Reset the hand (clear errors).
        
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
        """
        self._check_connection()
        try:
            success = self.modbus.write_single_register(RegisterRH56DFTP.CLEAR_ERROR, 1)
            if not success:
                raise CommandError("Failed to reset hand")
        except Exception as e:
            raise CommandError(f"Error resetting hand: {e}")
    
    def get_forces(self) -> List[int]:
        """
        Read the actual forces applied by all joints.
        
        Returns:
            List of force values for all joints
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
        """
        self._check_connection()
        try:
            forces = self.modbus.read_holding_registers(RegisterRH56DFTP.FORCE_ACT, 6)
            if forces is None:
                raise CommandError("Failed to read forces from hand")
            return forces
        except Exception as e:
            raise CommandError(f"Error reading forces: {e}")
    
    def get_status(self) -> Dict[str, any]:
        """
        Get comprehensive status information from the hand.
        
        Returns:
            Dictionary containing pose, angles, forces, contact info, etc.
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
        """
        self._check_connection()
        
        try:
            status = {
                'pose': self.read_pose(),
                'angles': self.read_angles(),
                'forces': self.get_forces(),
                'contact_info': self.read_contact_info(),
            }
            return status
        except Exception as e:
            raise CommandError(f"Error getting status: {e}")

