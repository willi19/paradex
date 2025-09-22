from time import sleep
from paradex.io.signal_generator.UTGE900 import UTGE900
import chime

chime.theme('pokemon')

if __name__ == "__main__":
    gSgen = UTGE900()     #"USB0::0x6656::0x0834::770595938::INSTR"

    gSgen.generate(freq=1000)
    
    gSgen.on(1)
    sleep(5)
    print("on")
    
    gSgen.off(1)
    sleep(5)
    gSgen.quit()
    
    
