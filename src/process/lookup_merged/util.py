import argparse
import os

from paradex.utils.file_io import shared_dir

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=["inference", "capture"])
    parser.add_argument('--obj', nargs="+", type=str, default=None)
    parser.add_argument('--index', nargs="+", type=str, default=None)
    
    args = parser.parse_args()
    
    return args

def get_path(args):
    return os.path.join(args.type, "lookup")

def get_process_list(args):
    ret = []
    path = get_path(args)
    
    name_list = args.obj if args.obj is not None else os.listdir(os.path.join(shared_dir, path))
    
    for name in name_list:
        process_path = os.path.join(shared_dir, path, name)
        index_list = os.listdir(process_path)
        
        for index in index_list:
            if args.index is None or index in args.index:
                ret.append(os.path.join(path, name, index))
                
    return ret