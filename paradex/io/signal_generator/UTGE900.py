import os
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS

import pyvisa
import re
from time import sleep

from paradex.utils.env import get_network_info

class UTGE900:
    """
    Unit-T UTG900 signal generator PYVISA control wrapper.
    """
    _rm = pyvisa.ResourceManager()
    @staticmethod
    def list_resources():
        return UTGE900._rm.list_resources()
             
    def __init__(self):
        print(self.list_resources())
        addr = get_network_info()["signal_generator"]
        self.sgen = UTGE900._rm.open_resource(addr)
        # "USB0::0x6656::0x0834::770595938::INSTR"
        self.ch = [ False, False ]
        self.llOpen()
        self.generate()

    def quit(self ):
        self.sgen.close()
        UTGE900._rm.close()
        UTGE900._rm = None

    # Low level commuincation 
    def write(self, cmd):
        self.sgen.write(cmd)
        
    def query(self, cmd, strip=False ):
        ret = self.sgen.query(cmd)
        if strip: ret = ret.rstrip()
        return( ret )

    def llOpen(self):
        self.write( "System:LOCK off")
        
    def llCh(self, ch):
        self.write( "KEY:CH{}".format(ch))
        
    def llWave(self):
        self.write( "KEY:Wave")
        
    def llUtility(self):
        self.write( "KEY:Utility")
        
    def llF(self, digit ):
        self.write( "KEY:F{}".format(digit))
    def llKey(self, keyStr):
           self.write( "KEY:{}".format(keyStr))
    def llUp(self):
        self.llKey("Up")
    def llDown(self):
        self.llKey("Down")
    def llLeft(self):
        self.llKey("Left")
    def llRight(self):
        self.llKey("Right")
           
    def llNum(self, numStr):
        def ch2cmd( ch ):
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
                return  keyName
            except KeyError:
                    logging.fatal( "Could not extract keyName for ch {} numStr {}".format( ch, numStr ))
                    raise 
        for ch in str(numStr):
            self.write( "KEY:{}".format(ch2cmd(ch)))
            
    def llFKey( self, val, keyMap):
        try:
            self.llF(keyMap[val])
        except KeyError as err:
            logging.error( "Invalid key: '{}', valid keys: {}".format( val, keyMap.keys()))
            logging.error( str(err) )
            raise

    # IL intermediate (=action in a given mode)
    def ilFreq( self, freq, unit ):
        self.llNum( str(freq))
        self.ilFreqUnit( unit )
        
    def ilAmp( self, amp, unit ):
        self.llNum( str(amp))
        self.ilAmpUnit( unit )

    def ilWave1( self, wave ):
        """Selec wave type"""
        waveMap  = {
        "sine": "1",
        "square": "2",
        "pulse":  "3",
        "ramp": "4",
        "arb": "5",
        "MHz": "6",
        }
        self.llFKey( val=wave, keyMap = waveMap )

    def ilWave1Props( self, wave ):
        """Wave properties, page1"""
        waveMap  = {
        "Freq": "1",
        "Amp": "2",
        "Offset":  "3",
        "Phase": "4",
        "Duty": "5",
        "Page Down": "6",
        }
        self.llFKey( val=wave, keyMap = waveMap )

    # Units
    def ilChooseChannel( self, ch ):
        """Key sequence to to bring UTG962 to display to a known state. 
        
        Here, invoke Utility option, use function key F1 or F2 to
        choose channel. Do it twice (and visit Wave menu in between)

        """
        ch = int(ch)
        self.llUtility()
        self.ilUtilityCh( ch )
        self.llWave()
        self.llUtility()
        self.ilUtilityCh( ch )
        self.llWave()
        sleep( 0.1)
        
    def ilFreqUnit( self, unit):
        freqUnit  = {
        "uHz": "1",
        "mHz": "2",
        "Hz":  "3",
        "kHz": "4",
        "MHz": "5",
        }
        self.llFKey( val=unit, keyMap = freqUnit)
        
    def ilAmpUnit( self, unit ):
        ampUnit  = {
        "mVpp": "1",
        "Vpp": "2",
        "mVrms":  "3",
        "Vrms": "4",
        "Cancel": "6",
        }
        self.llFKey(val=unit, keyMap = ampUnit)
        
    def ilUtilityCh( self, ch ):
        chSelect  = {
        1: "1",
        2: "2",
        }
        self.llFKey( val=ch, keyMap = chSelect )

    def on(self,ch):
        ch = int(ch)
        if self.ch[ch-1]: 
            return
        
        self.ilChooseChannel( ch )
        self.llCh(ch)
        self.ch[ch-1] = True
        self.llOpen()
        sleep( 0.1)

    def off(self,ch):
        ch = int(ch)
        if not self.ch[ch-1]: 
            return
        
        self.ilChooseChannel( ch )
        self.llCh(ch)
        self.ch[ch-1] = False
        self.llOpen()
        sleep( 0.1)
        self.generate()
        self.llOpen()

    def generate( self, ch=1, wave="square", freq=30, amp=10):
        """sine, square, pulse generation
        """
        # Deactivate
        self.off(ch)
        # Start config
        self.ilChooseChannel( ch )
        # At this point correct channel selected
        self.ilWave1( wave )
        # Frequencey (sine, square, pulse,arb)
        
        self.llDown()
        self.ilWave1Props("Freq")
        self.ilFreq(freq, "Hz")
        
        self.ilWave1Props("Amp")
        self.ilAmp(amp, "Vpp")
            
    def getName(self):
        return( self.query( "*IDN?"))