from paradex.simulator.isaac import Simulator
from paradex.utils.keyboard_listener import listen_keyboard
from threading import Event
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus'])
parser.add_argument('--arm', default=None, type=str)
parser.add_argument('--hand', default=None, type=str)
args = parser.parse_args()

if args.device == 'xsens': 
    from paradex.io.xsens.receiver import XSensReceiver
    teleop_device = XSensReceiver()

if args.device =='occulus':
    from paradex.io.occulus.receiver import OculusReceiver
    teleop_device = OculusReceiver()

view_physics = False
view_replay = False
headless = False
save_video = False
save_state = False

sim = Simulator(view_physics, 
                view_replay, 
                headless, 
                save_video, 
                save_state)

arm_name = None
hand_name = None
obj_list = []
sim.load_asset(arm_name, hand_name, obj_list)

if save_video:
    sim.load_camera() # load camparam if desired
    sim.set_videopath("teleop_sim/video")

if save_state:
    sim.set_savepath("teleop_sim/state")

env_obj_list = [[],["bottle"]]
sim.load_env(env_obj_list)

stop_event = Event()
listen_keyboard({"q":stop_event})

while not stop_event.is_set():
    hand_pose = teleop_device.get_data() #{'Left':{}, 'Right':{}}
    robot_action = retargeter.retarget(hand_pose)
    
    sim.step(0, {"robot":robot_action,"robot_viz":robot_action,"object_vis":object_pose})

sim.save()
sim.terminate()