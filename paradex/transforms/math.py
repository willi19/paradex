import numpy as np

def rigid_transform_3D(A, B):
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
