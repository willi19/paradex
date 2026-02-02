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

def SOLVE_XA_B(A, B):
    """
    Calculates the rigid transformation matrix (rotation + translation)
    that best aligns A to B using Singular Value Decomposition (SVD).
    
    Parameters:
        A (numpy.ndarray): Nx3 array of points.
        B (numpy.ndarray): Nx3 array of points.
    
    Returns:
        numpy.ndarray: 4x4 transformation matrix.
        T @ A = B
    """
    assert A.shape == B.shape, "Input matrices must have the same shape."

    # Compute centroids
    
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Subtract centroids
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute the covariance matrix
    H = AA.T @ BB

    # Perform SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Ensure a right-handed coordinate system (correct for reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_B - R @ centroid_A

    # Construct homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T
