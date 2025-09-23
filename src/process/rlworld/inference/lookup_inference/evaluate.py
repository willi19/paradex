import os
import numpy as np
import argparse

from paradex.utils.file_io import shared_dir, load_camparam, load_c2r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_name', nargs="+", type=str, default=None)
    parser.add_argument('--grasp_type', nargs="+", type=str, default=None)
    args = parser.parse_args()

    process_list = []
    
    if args.obj_name == None:
        name_list = os.listdir(os.path.join(shared_dir, 'inference_', "lookup"))
        name_list.sort()

    else:
        name_list = args.obj_name
        
    for name in name_list:
        grasp_list = os.listdir(os.path.join(shared_dir, "inference_", "lookup", name))
        if args.grasp_type is not None:
            for grasp_name in args.grasp_type:
                if grasp_name in grasp_list:
                    process_list.append((name, grasp_name))
        if args.grasp_type is None:
            for grasp_name in grasp_list:
                process_list.append((name, grasp_name))
    
    for name, grasp_name in process_list:
        demo_path = os.path.join(os.path.join(shared_dir, "inference_", "lookup", name, grasp_name))
        index_list = os.listdir(demo_path)
        for index in index_list:
            c2r = load_c2r(os.path.join(demo_path, index))
            pick_6D = np.load(os.path.join(demo_path, index, "pick_6D.npy"))
            place_6D = np.load(os.path.join(demo_path, index, "place_6D.npy"))
            
            obj_T = np.load(os.path.join(demo_path, index, "obj_T.npy"))
            for 
            import pdb; pdb.set_trace()