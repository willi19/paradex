from pathlib import Path

from curobo.geom.types import WorldConfig, Cuboid, Sphere, Cylinder, Mesh
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml

# pregrasp position 
import joblib

cfg_path = str(Path().home() / "robothome" / "robothome_collision/1001_rlwrld_collision_v8_grasp2/robothome_scene_collision_rlwrld.yaml")
target_env = WorldConfig.from_dict(load_yaml(cfg_path))

ee_paths = (Path().home() / "robothome" / "robothome_collision/1001_rlwrld_collision_v8_grasp2").glob("./ramen_place_*.pkl") # where to place the object
eetargets = dict()
for ep in ee_paths:
    # print(ep)
    graspnum = int(ep.stem.split("_")[-1])
    targetdata = joblib.load(ep)
    finger_action[graspnum] = targetdata["action"][::-1]
    if graspnum in PLACE_TARGETS:
        
        eetargets[graspnum] = reformat_transf(targetdata["se3"]) # grasp position (local2robotbase)

        place_target_ee[graspnum] = reformat_transf(targetdata["se3"] @ np.linalg.inv(targetdata["se3_in_obj"]) @ ramen_local_pose) # local2robotbase @ 


        

        geom = Cylinder(
            name = f"ramen_{graspnum}",
            radius = 0.0475,#0.0475,
            height = 0.102, #0.102,
            pose = place_target_ee[graspnum]
        )
        ramen_targets[graspnum] = geom