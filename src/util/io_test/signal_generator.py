from time import sleep
from paradex.io.signal_generator.UTGE900 import UTGE900


if __name__ == "__main__":
    gSgen = UTGE900()
    gSgen.generate(freq=10)
    
    gSgen.on(1)
    sleep(5)
    print("on")
    
    gSgen.off(1)
    sleep(5)
    gSgen.quit()
    