import os
import time

from paradex.io.robot_controller import get_hand
from paradex.utils.path import shared_dir


hand = get_hand("inspire")

hand.start(os.path.join(shared_dir, "capture/erasethis/5", "raw", "hand"))

time.sleep(15)



hand.end()