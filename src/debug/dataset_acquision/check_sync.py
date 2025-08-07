import numpy as np
import os

index_list = [13, 14, 15, 16, 17]
for ind in index_list:
    npy_path = f"/home/temp_id/shared_data/debug_/aruco/{ind}/marker2D"
    npy_list = os.listdir(npy_path)
    
    for npy_file in npy_list:
        npy_file_path = os.path.join(npy_path, npy_file)
        data = np.load(npy_file_path, allow_pickle=True).item()
        
        import pdb; pdb.set_trace()