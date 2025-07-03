from paradex.simulator.isaac import Simulator
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.retargetor.unimanual import unimanual

from threading import Event
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus'])
parser.add_argument('--arm', type=str, nargs="+", default=["franka", "xarm"])
parser.add_argument('--hand', type=str, nargs="+", default=["inspire", "allegro"])
parser.add_argument('--object', type=str, nargs="+", default=[])

args = parser.parse_args()

if args.device == 'xsens': 
    from paradex.io.xsens.receiver import XSensReceiver
    teleop_device = XSensReceiver()

if args.device =='occulus':
    from paradex.io.occulus.receiver import OculusReceiver
    teleop_device = OculusReceiver()

save_video = False
save_state = False
headless = False
num_envs = 1

sim = Simulator(
                headless, 
                num_envs
                )

for arm_name in args.arm:
    for hand_name in args.hand:
        sim.load_robot_asset(arm_name, hand_name)

obj_list = []
for obj_name in obj_list:
    sim.load_object_asset(obj_name)

if save_video:
    sim.load_camera() # load camparam if desired
    sim.set_videopath("teleop_sim/video")

if save_state:
    sim.set_savepath("teleop_sim/state")

for arm_name in args.arm:
    for hand_name in args.hand:
        env_info = {
            "robot":{"right":(arm_name, hand_name)},
            "robot_vis":{"right":(arm_name, hand_name)}, 
            "object":{},
            "object_vis":{}
        }

        sim.add_env(env_info)

stop_event = Event()
listen_keyboard({"q":stop_event})

while not stop_event.is_set():
    hand_pose = teleop_device.get_data() #{'Left':{}, 'Right':{}}
    robot_action = retargeter.retarget(hand_pose)
    
    sim.step(0, {"robot":{"right":robot_action},"robot_vis":{"right":robot_action}})
    sim.tick()
    
sim.save()
sim.terminate()