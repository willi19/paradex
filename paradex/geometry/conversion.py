import transforms3d as t3d
import numpy as np

def aa2mtx(pos_aa):
    pos_aa = np.array(pos_aa)
    T = np.eye(4)
    T[:3,3] = pos_aa[:3] / 1000
    
    angle = np.linalg.norm(pos_aa[3:])
    axis = pos_aa[3:] / angle
    T[:3,:3] = t3d.axangles.axangle2mat(axis, angle)
    
    return T

def to_homo(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1) 

def project(proj_mtx, x):
    assert len(x.shape) == 2
    
    if x.shape[-1] == 3:
        x = to_homo(x)
        
    proj_x = (proj_mtx @ x.T).T
    if proj_x.shape[-1] == 4:
        proj_x = proj_x[:, 3]
        
    proj_x = (proj_x / proj_x[:,2:])[:, :2]
    return proj_x