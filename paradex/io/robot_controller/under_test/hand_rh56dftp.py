"""
Main module for controlling the Inspire Hand RH56DFTP.
Based on the RH56DFTP User Manual V1.0.0.
"""

from enum import IntEnum
from typing import List, Optional, Dict, Tuple
import time
from contextlib import contextmanager

from modbus import ModbusClient
from exceptions import InspireHandError, ConnectionError, CommandError


class RegisterRH56DFTP:
    """
    Register addresses for the Inspire Hand RH56DFTP.
    
    Note: These addresses should be verified against the RH56DFTP User Manual V1.0.0.
    Update these values based on the actual register map from the manual.
    """
    # System information registers
    HAND_ID = 1000        # Dexterous Hand ID
    BAUD_RATE = 1002      # Baud Rate
    CLEAR_ERROR = 1004    # Error clearance
    SAVE = 1005           # Saving data in Flash
    RESET_PARA = 1006     # Restoring factory defaults
    FORCE_SENSOR_CALIB = 1009  # Force sensor calibration
    
    DEFAULT_SPEED_SET = 1032 # Set value of power-on speed for each DOF
    DEFAULT_FORCE_SET = 1033 # Set value of power-on force control threshold for each DOF

    # Control registers - Pose/Position setting
    # These addresses need to be verified from the manual
    POS_SET = 1474        # Pose/Position set values (for all joints)
    ANGLE_SET = 1486      # Angle set values (for all joints)
    FORCE_SET = 1498      # Force control threshold values
    SPEED_SET = 1522      # Speed values
    POSE_MAX = 2000
    ANGLE_MAX = 1000
    SPEED_MAX = 1000
    FORCE_MAX = 3000      # 3000g

    # Status registers - Current pose reading
    POSE_ACT = 1534       # Actual pose/position values
    ANGLE_ACT = 1546      # Actual angle values
    FORCE_ACT = 1582      # Actual force values
    CURRENT = 1594        # Actuator current values
    ERROR = 1606          # Error codes
    STATUS = 1612         # Status information
    TEMP = 1618           # Actuator temperature values

    LF_TOUCH = 3000         # Little finger tactile sensor
    RF_TOUCH = 3370         # Ring finger tactile sensor
    MF_TOUCH = 3740         # Middle finger tactile sensor
    IF_TOUCH = 4110         # Index finger tactile sensor
    TF_TOUCH = 4480         # Thumb tactile sensor
    PALM_TOUCH = 4900       # Palm tactile sensor
    
    # Tactile sensor data lengths (in bytes/registers)
    # Tip: 3*3*2 = 18 bytes
    # Nail: 12*8*2 = 192 bytes
    # Pad: 10*8*2 = 160 bytes (Thumb Pad is 12*8*2 = 192 bytes)
    # Thumb Middle: 3*3*2 = 18 bytes
    # Palm: 8*14*2 = 224 bytes
    
    TACTILE_LENS = {
        'tip': 18,
        'nail': 192,
        'pad': 160,
        'thumb_mid': 18,
        'thumb_pad': 192,
        'palm': 224
    }

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


class ErrorStatus(IntEnum):
    """Error codes for the Inspire Hand RH56DFTP."""
    Locked_Rotor_Error = 0
    Over_Temperature_Error = 1
    Overcurrent_Error = 2
    Abnormal_Operation_Motor = 3
    Communication_Error = 4
    

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
                    0: 115200 (R485 Inferface)
                    0: 1000 (CAN Interface)'
            slave_id: Modbus slave ID of the hand
            debug: Whether to print debug information
        """
        self.modbus = ModbusClient(port, baudrate, slave_id)
        self.modbus.debug = debug
        self._connected = False

        with self.connect():
            self.force_calibration()

    
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


    def read_error(self) -> List[int]:
        """
        Read the current error code of the hand.
        
        Returns:
            List of error codes for all joints
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
        """
        self._check_connection()
        try:
            error_values = self.modbus.read_holding_registers(RegisterRH56DFTP.ERROR, 6)
            if error_values is None:
                raise CommandError("Failed to read error from hand")
            return error_values
        except Exception as e:
            raise CommandError(f"Error reading error: {e}")


    def read_contact(self) -> List[int]:
        """
        Read the current contact flag of the hand.
        
        Returns:
            List of contact flags for all joints
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
        """
        self._check_connection()
        try:
            contact_flags_data = self.modbus.read_holding_registers(RegisterRH56DFTP.CONTACT_FLAG, 6)
            if contact_flags_data is None:
                raise CommandError("Failed to read contact flags from hand")
            return contact_flags_data
        except Exception as e:
            raise CommandError(f"Error reading contact flags: {e}")


    def write_pose_by_id(self, id: int, pose: int) -> None:
        """
        Write/execute a pose for all joints.
        
        Args:
            id: Joint ID (0-5)
            pose: Pose value for the joint (0-2000, -1 for no action)
            order of DOF: lf, rf, mf, if, thumb bending, thumb rotating
            0 : open 2000: closed
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
            ValueError: If pose values are invalid
        """
        self._check_connection()
        
        if not 0 <= id <= 5:
            raise ValueError(f"Invalid joint ID. Must be 0-5, got {id}")
        
        if not 0 <= pose <= RegisterRH56DFTP.POSE_MAX:
            raise ValueError(f"Pose value must be between 0 and {RegisterRH56DFTP.POSE_MAX}, got {pose}")
        
        try:
            # Write pose values to registers
            success = self.modbus.write_single_register(RegisterRH56DFTP.POS_SET + id*2, pose)
            if not success:
                raise CommandError(f"Failed to write pose to joint {id}")
        except Exception as e:
            raise CommandError(f"Error writing pose to joint {id}: {e}")

    
    def write_pose(self, pose: List[int]) -> None:
        """
        Write/execute a pose for all joints.
        
        Args:
            pose: List of pose values for all joints (typically 6 values) (0-2000, -1 for no action)
            order of DOF: lf, rf, mf, if, thumb bending, thumb rotating
            
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
        
        # Validate pose values (typically 0-2000 range)
        for i, value in enumerate(pose):
            if not isinstance(value, int):
                raise ValueError(f"Pose value at index {i} must be an integer")
            if not 0 <= value <= RegisterRH56DFTP.POSE_MAX:
                raise ValueError(f"Pose value at index {i} must be between 0 and 2000, got {value}")
        
        try:
            # Write pose values to registers
            success = self.modbus.write_multiple_registers(RegisterRH56DFTP.POS_SET, pose)
            if not success:
                raise CommandError("Failed to write angle to hand")
        except Exception as e:
            raise CommandError(f"Error writing angle: {e}")


    def write_angle_by_id(self, id: int, angle: int) -> None:
        """
        Write/execute a pose for all joints.
        
        Args:
            id: Joint ID (0-5)
            angle: angle value for each joint (0-1000)
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
            ValueError: If angle values are invalid
        """
        self._check_connection()
        
        if not 0 <= id <= 5:
            raise ValueError(f"Invalid joint ID. Must be 0-5, got {id}")
        
        if not 0 <= angle <= RegisterRH56DFTP.ANGLE_MAX:
            raise ValueError(f"Angle value must be between 0 and {RegisterRH56DFTP.ANGLE_MAX}, got {angle}")
        
        try:
            success = self.modbus.write_single_register(RegisterRH56DFTP.ANGLE_SET+2*id, angle)
            if not success: 
                raise CommandError("Failed to write angles to hand")
        except Exception as e:
            raise CommandError(f"Error writing angles: {e}")
    
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
            if not 0 <= value <= RegisterRH56DFTP.ANGLE_MAX:
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
        
        if not 0 <= angle <= RegisterRH56DFTP.ANGLE_MAX:
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
        
        if not 0 <= speed <= RegisterRH56DFTP.SPEED_MAX:
            raise ValueError(f"Invalid speed. Must be 0-1000, got {speed}")
        
        try:
            values = [speed] * 6
            success = self.modbus.write_multiple_registers(RegisterRH56DFTP.SPEED_SET, values)
            if not success:
                raise CommandError("Failed to set speed")
        except Exception as e:
            raise CommandError(f"Error setting speed: {e}")
        
    def force_calibration(self):
        try:
            success = self.modbus.write_single_register(RegisterRH56DFTP.FORCE_SENSOR_CALIB, 1)
            if not success:
                raise CommandError("Failes to calibrate force sensor")
        except Exception as e:
            raise CommandError(f"Error during calibration force sensor: {e}")

    def set_force_threshold(self, force: int) -> None:
        """
        Set the force threshold for all joints.
        
        Args:
            force: Force threshold value (0-3000)
            After the user set the angle of the index finger, index finger will move towards
            until it reach the actual force in this force threshold
            
        Raises:
            ConnectionError: If not connected
            CommandError: If command fails
            ValueError: If force is invalid
        """
        self._check_connection()
        
        if not 0 <= force <= RegisterRH56DFTP.FORCE_MAX:
            raise ValueError(f"Invalid force. Must be 0-3000, got {force}")
        
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
        Range: -4000~4000, unit g

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

    def _read_tactile_section(self, start_addr: int, byte_count: int) -> List[int]:
        """
        Read a section of tactile sensor data.
        
        Args:
            start_addr: Starting register address
            byte_count: Number of bytes (registers) to read
            
        Returns:
            List of 16-bit integer values
        """
        values = []
        # Max registers per read (Modbus limit is typically around 125)
        chunk_size = 100
        
        for i in range(0, byte_count, chunk_size):
            count = min(chunk_size, byte_count - i)
            addr = start_addr + i
            
            # Read registers (assuming 1 register = 1 byte of data)
            regs = self.modbus.read_holding_registers(addr, count)
            if regs is None:
                raise CommandError(f"Failed to read tactile data at {addr}")
            
            values.extend(regs)
            
        # Combine bytes into 16-bit integers (Little-Endian)
        # Data Point = Low Byte + High Byte * 256
        # regs[0] is Low Byte, regs[1] is High Byte
        tactile_values = []
        for i in range(0, len(values), 2):
            if i + 1 < len(values):
                low_byte = values[i]
                high_byte = values[i+1]
                val = low_byte | (high_byte << 8)
                tactile_values.append(val)
                
        return tactile_values

    def read_tactile_data(self) -> Dict[str, any]:
        """
        Read all tactile sensor data.
        
        Returns:
            Dictionary containing tactile data for all fingers and palm.
            Structure:
            {
                'little': {'tip': [], 'nail': [], 'pad': []},
                'ring': {'tip': [], 'nail': [], 'pad': []},
                'middle': {'tip': [], 'nail': [], 'pad': []},
                'index': {'tip': [], 'nail': [], 'pad': []},
                'thumb': {'tip': [], 'nail': [], 'mid': [], 'pad': []},
                'palm': []
            }
        """
        self._check_connection()
        
        try:
            data = {}
            
            # Little Finger
            lf_base = RegisterRH56DFTP.LF_TOUCH
            data['little'] = {
                'tip': self._read_tactile_section(lf_base, RegisterRH56DFTP.TACTILE_LENS['tip']),
                'nail': self._read_tactile_section(lf_base + 18, RegisterRH56DFTP.TACTILE_LENS['nail']),
                'pad': self._read_tactile_section(lf_base + 18 + 192, RegisterRH56DFTP.TACTILE_LENS['pad'])
            }
            
            # Ring Finger
            rf_base = RegisterRH56DFTP.RF_TOUCH
            data['ring'] = {
                'tip': self._read_tactile_section(rf_base, RegisterRH56DFTP.TACTILE_LENS['tip']),
                'nail': self._read_tactile_section(rf_base + 18, RegisterRH56DFTP.TACTILE_LENS['nail']),
                'pad': self._read_tactile_section(rf_base + 18 + 192, RegisterRH56DFTP.TACTILE_LENS['pad'])
            }
            
            # Middle Finger
            mf_base = RegisterRH56DFTP.MF_TOUCH
            data['middle'] = {
                'tip': self._read_tactile_section(mf_base, RegisterRH56DFTP.TACTILE_LENS['tip']),
                'nail': self._read_tactile_section(mf_base + 18, RegisterRH56DFTP.TACTILE_LENS['nail']),
                'pad': self._read_tactile_section(mf_base + 18 + 192, RegisterRH56DFTP.TACTILE_LENS['pad'])
            }
            
            # Index Finger
            if_base = RegisterRH56DFTP.IF_TOUCH
            data['index'] = {
                'tip': self._read_tactile_section(if_base, RegisterRH56DFTP.TACTILE_LENS['tip']),
                'nail': self._read_tactile_section(if_base + 18, RegisterRH56DFTP.TACTILE_LENS['nail']),
                'pad': self._read_tactile_section(if_base + 18 + 192, RegisterRH56DFTP.TACTILE_LENS['pad'])
            }
            
            # Thumb
            tf_base = RegisterRH56DFTP.TF_TOUCH
            data['thumb'] = {
                'tip': self._read_tactile_section(tf_base, RegisterRH56DFTP.TACTILE_LENS['tip']),
                'nail': self._read_tactile_section(tf_base + 18, RegisterRH56DFTP.TACTILE_LENS['nail']),
                'mid': self._read_tactile_section(tf_base + 18 + 192, RegisterRH56DFTP.TACTILE_LENS['thumb_mid']),
                'pad': self._read_tactile_section(tf_base + 18 + 192 + 18, RegisterRH56DFTP.TACTILE_LENS['thumb_pad'])
            }
            
            # Palm
            palm_base = RegisterRH56DFTP.PALM_TOUCH
            data['palm'] = self._read_tactile_section(palm_base, RegisterRH56DFTP.TACTILE_LENS['palm'])
            
            return data
            
        except Exception as e:
            raise CommandError(f"Error reading tactile data: {e}")

