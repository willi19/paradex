import os
import time

class UTGE900:
    """
    Unit-T UTG900 signal generator - Direct USBTMC control
    """
    
    def __init__(self, addr):
        # Get device path from network info
        self.device = addr
        
        # Check device exists and has permissions
        if not os.path.exists(addr):
            raise FileNotFoundError(f"Device {addr} not found")

        if not os.access(addr, os.R_OK | os.W_OK):
            raise PermissionError(f"No permission for {addr}. Run: sudo chmod 666 {addr}")
        
        self.ch = [False, False]

    def end(self):
        """Close device"""
        self.llOpen()

    # Low level communication
    def write(self, cmd):
        """Send command to device"""
        with open(self.device, 'wb') as f:
            f.write((cmd + '\n').encode('ascii'))
            f.flush()
    
    def query(self, cmd, strip=False):
        """Send command and read response"""
        with open(self.device, 'w+b', buffering=0) as f:
            f.write((cmd + '\n').encode('ascii'))
            time.sleep(0.05)
            ret = f.read(4096).decode('ascii').strip()
        
        if strip:
            ret = ret.rstrip()
        return ret

    def llOpen(self):
        self.write("System:LOCK off")
    
    def llCh(self, ch):
        self.write(f"KEY:CH{ch}")
    
    def llWave(self):
        self.write("KEY:Wave")
    
    def llUtility(self):
        self.write("KEY:Utility")
    
    def llF(self, digit):
        self.write(f"KEY:F{digit}")
    
    def llKey(self, keyStr):
        self.write(f"KEY:{keyStr}")
    
    def llUp(self):
        self.llKey("Up")
    
    def llDown(self):
        self.llKey("Down")
    
    def llLeft(self):
        self.llKey("Left")
    
    def llRight(self):
        self.llKey("Right")
    
    def llNum(self, numStr):
        def ch2cmd(ch):
            chMap = {
                "0": "NUM0",
                "1": "NUM1",
                "2": "NUM2",
                "3": "NUM3",
                "4": "NUM4",
                "5": "NUM5",
                "6": "NUM6",
                "7": "NUM7",
                "8": "NUM8",
                "9": "NUM9",
                "-": "SYMBOL",
                ".": "DOT",
                ",": "DOT",
            }
            try:
                keyName = chMap[ch]
                return keyName
            except KeyError:
                raise
        
        for ch in str(numStr):
            self.write(f"KEY:{ch2cmd(ch)}")
    
    def llFKey(self, val, keyMap):
        try:
            self.llF(keyMap[val])
        except KeyError as err:
            raise

    # IL intermediate (=action in a given mode)
    def ilFreq(self, freq, unit):
        self.llNum(str(freq))
        self.ilFreqUnit(unit)
    
    def ilAmp(self, amp, unit):
        self.llNum(str(amp))
        self.ilAmpUnit(unit)

    def ilWave1(self, wave):
        """Select wave type"""
        waveMap = {
            "sine": "1",
            "square": "2",
            "pulse": "3",
            "ramp": "4",
            "arb": "5",
            "MHz": "6",
        }
        self.llFKey(val=wave, keyMap=waveMap)

    def ilWave1Props(self, wave):
        """Wave properties, page1"""
        waveMap = {
            "Freq": "1",
            "Amp": "2",
            "Offset": "3",
            "Phase": "4",
            "Duty": "5",
            "Page Down": "6",
        }
        self.llFKey(val=wave, keyMap=waveMap)

    # Units
    def ilChooseChannel(self, ch):
        """Key sequence to bring UTG962 to display to a known state."""
        ch = int(ch)
        self.llUtility()
        self.ilUtilityCh(ch)
        self.llWave()
        self.llUtility()
        self.ilUtilityCh(ch)
        self.llWave()
        time.sleep(0.1)
    
    def ilFreqUnit(self, unit):
        freqUnit = {
            "uHz": "1",
            "mHz": "2",
            "Hz": "3",
            "kHz": "4",
            "MHz": "5",
        }
        self.llFKey(val=unit, keyMap=freqUnit)
    
    def ilAmpUnit(self, unit):
        ampUnit = {
            "mVpp": "1",
            "Vpp": "2",
            "mVrms": "3",
            "Vrms": "4",
            "Cancel": "6",
        }
        self.llFKey(val=unit, keyMap=ampUnit)
    
    def ilUtilityCh(self, ch):
        chSelect = {
            1: "1",
            2: "2",
        }
        self.llFKey(val=ch, keyMap=chSelect)

    def start(self, fps, ch=1):
        ch = int(ch)
        if self.ch[ch-1]:
            return
        
        self.generate(ch, wave="square", freq=fps, amp=10)
        
        self.ilChooseChannel(ch)
        self.llCh(ch)
        
        self.ch[ch-1] = True
        self.llOpen()
        
        time.sleep(0.1)

    def stop(self, fps, ch=1):
        ch = int(ch)
        if not self.ch[ch-1]:
            return
        
        
        self.ilChooseChannel(ch)
        self.llCh(ch)
        
        self.ch[ch-1] = False
        self.llOpen()

    def generate(self, ch=1, wave="square", freq=30, amp=10):
        """sine, square, pulse generation"""
        # Deactivate
        self.stop(ch)
        # Start config
        self.ilChooseChannel(ch)
        # At this point correct channel selected
        self.ilWave1(wave)
        # Frequency (sine, square, pulse, arb)
        
        self.llDown()
        self.ilWave1Props("Freq")
        self.ilFreq(freq, "Hz")
        
        self.ilWave1Props("Amp")
        self.ilAmp(amp, "Vpp")
    
    def getName(self):
        return self.query("*IDN?")