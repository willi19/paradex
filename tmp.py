from paradex.visualization_.viewer import ViserViewer
import argparse
from pathlib import Path
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
        scene_path: Path = Path('/home/jisoo/teserract_nas/processed/spray/1'),
        object_nm: str='spray',
        downsample_factor=4,
    '''
    # parser.add_argument('--scene_path', type=str, required=True)
    # parser.add_argument('--object_nm', type=str, required=True)
    parser.add_argument('--obj_status_path',type=str,default=None)
    parser.add_argument('--downsample_factor', type=int, default=4)
    args = parser.parse_args()

    args.scene_path = '/home/jisoo/teserract_nas/processed/spray/1'
    object_nm = args.scene_path.split("/")[-2]

    viewer = ViserViewer(Path(args.scene_path), object_nm, args.obj_status_path, args.downsample_factor, draw_tg=['mesh', 'camera'])
    
    while True:
        viewer.update()
        time.sleep(0.5)
