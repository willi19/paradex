import argparse
import os

from paradex.utils.file_io import shared_dir

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=["inference", "capture"])
    parser.add_argument('--obj', nargs="+", type=str, default=None)
    parser.add_argument('--hand', nargs="+", type=str, default=None)
    parser.add_argument('--index', nargs="+", type=str, default=None)
    
    args = parser.parse_args()
    
    return args

def get_path(args):
    return os.path.join(args.type, "lookup")

def get_process_list(args):
    process_list = []
    ret = []
    path = get_path(args)
    
    name_list = args.obj if args.obj is not None else os.listdir(os.path.join(shared_dir, path))
    for name in name_list:
        process_list += [(name, hand_name) for hand_name in os.listdir(os.path.join(shared_dir, path, name)) if (hand_name in args.hand_list or args.hand_list is None)]
    
    for process_file in process_list:
        obj = process_file[0]
        hand = process_file[1]
        
        process_path = os.path.join(shared_dir, path, obj, hand)
        index_list = os.listdir(process_path)
        
        for index in index_list:
            if index in args.index or args.index is None:
                ret.append((obj, hand, index))
                
    return ret