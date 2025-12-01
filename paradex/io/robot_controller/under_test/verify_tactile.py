
import sys
import os
from unittest.mock import MagicMock, patch

# Mock serial module before importing modbus
sys.modules['serial'] = MagicMock()

# Add the directory to the path so we can import the module
sys.path.append('/home/jisoo/data2/paradex/paradex/io/robot_controller/under_test')

from hand_rh56dftp import InspireHandRH56DFTP, RegisterRH56DFTP

def test_tactile_reading():
    # Mock the ModbusClient
    mock_modbus = MagicMock()
    
    # Setup the mock to return predictable data based on address
    def side_effect(address, count):
        # Return values equal to the address
        # This allows us to verify that we are reading from the correct location
        return list(range(address, address + count))
        
    mock_modbus.read_holding_registers.side_effect = side_effect
    mock_modbus.connect.return_value = True
    
    # Patch the ModbusClient in the module
    with patch('hand_rh56dftp.ModbusClient', return_value=mock_modbus):
        hand = InspireHandRH56DFTP()
        hand.open() # Ensure connected
        
        print("Testing read_tactile_data()...")
        data = hand.read_tactile_data()
        
        # Verify structure and values
        fingers = ['little', 'ring', 'middle', 'index', 'thumb']
        for finger in fingers:
            if finger not in data:
                print(f"FAILED: Missing {finger} in data")
                return
            
            parts = ['tip', 'nail', 'pad']
            if finger == 'thumb':
                parts = ['tip', 'nail', 'mid', 'pad']
                
            for part in parts:
                if part not in data[finger]:
                    print(f"FAILED: Missing {part} in {finger}")
                    return
                
                # Verify length
                expected_len = 0
                if part == 'tip': expected_len = 9
                elif part == 'nail': expected_len = 96
                elif part == 'pad': 
                    if finger == 'thumb': expected_len = 96
                    else: expected_len = 80
                elif part == 'mid': expected_len = 9
                
                actual_len = len(data[finger][part])
                if actual_len != expected_len:
                    print(f"FAILED: Wrong length for {finger} {part}. Expected {expected_len}, got {actual_len}")
                    return
                    
                # Verify values
                # The value should match the address
                # We need to know the expected start address for this part
                # We can get it from the class constants
                expected_start_addr = 0
                if finger == 'little': base = RegisterRH56DFTP.LF_TOUCH
                elif finger == 'ring': base = RegisterRH56DFTP.RF_TOUCH
                elif finger == 'middle': base = RegisterRH56DFTP.MF_TOUCH
                elif finger == 'index': base = RegisterRH56DFTP.IF_TOUCH
                elif finger == 'thumb': base = RegisterRH56DFTP.TF_TOUCH
                
                if part == 'tip': expected_start_addr = base
                elif part == 'nail': expected_start_addr = base + 18
                elif part == 'mid': expected_start_addr = base + 18 + 192
                elif part == 'pad':
                    if finger == 'thumb': expected_start_addr = base + 18 + 192 + 18
                    else: expected_start_addr = base + 18 + 192
                
                first_val = data[finger][part][0]
                if first_val != expected_start_addr:
                    print(f"FAILED: Wrong start value for {finger} {part}. Expected {expected_start_addr}, got {first_val}")
                    return

        if 'palm' not in data:
            print("FAILED: Missing palm in data")
            return
        
        if len(data['palm']) != 112:
             print(f"FAILED: Wrong length for palm. Expected 112, got {len(data['palm'])}")
             return
             
        if data['palm'][0] != RegisterRH56DFTP.PALM_TOUCH:
             print(f"FAILED: Wrong start value for palm. Expected {RegisterRH56DFTP.PALM_TOUCH}, got {data['palm'][0]}")
             return

        print("SUCCESS: Tactile data structure and reconstruction verified.")

if __name__ == "__main__":
    test_tactile_reading()
