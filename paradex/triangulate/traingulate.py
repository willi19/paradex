import numpy as np

def triangulate(corners: np.ndarray, projections: np.ndarray):
    """
    Triangulates 3D points from multiple camera views.
    
    Args:
        corners: (N, 2) matrix where each row is a 2D image coordinate from a camera.
        projections: (N, 3, 4) matrix where each row is a camera projection matrix.
    
    Returns:
        kp3d: (1, 3) matrix containing the triangulated 3D point.
    """
    numImg = projections.shape[0]
    if numImg < 2:
        return None  # At least two views are needed for triangulation

    curX = corners[:, 0]  # x-coordinates
    curY = corners[:, 1]  # y-coordinates

    A = np.zeros((numImg * 2, 4))  # (2N, 4) matrix

    for i in range(numImg):
        A[2 * i] = curY[i] * projections[i, 2] - projections[i, 1]  # Row for Y
        A[2 * i + 1] = curX[i] * projections[i, 2] - projections[i, 0]  # Row for X

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]  # Last row of V (smallest singular value)
    return (X[:3] / X[3]).reshape(3)  # Normalize homogeneous coordinate


def ransac_triangulation(corners: np.ndarray, projections: np.ndarray, threshold=1.5, iterations=100):
    """
    RANSAC-based triangulation to filter out outliers.
    
    corners : Nx2 matrix (N cameras, 2D image coordinates)
    projections : Nx3x4 matrix (N cameras)
    threshold : Inlier threshold for reprojection error
    iterations : Number of RANSAC iterations
    
    Returns:
        best_kp3d : (4, 3) matrix with filtered 3D keypoints
    """
    best_inliers = 0
    best_kp3d = None
    
    numPts = corners.shape[1]
    numImg = projections.shape[0]
    if numImg < 2:
        return None
    # print(corners.shape, projections.shape)
    for _ in range(iterations):
        # Randomly sample a subset of cameras
        sample_idx = np.random.choice(numImg, size=max(2, numImg // 2), replace=False)
        sampled_corners = corners[sample_idx]
        sampled_projections = projections[sample_idx]
        
        # Triangulate points
        kp3d = np.array(triangulate(sampled_corners, sampled_projections))
        kp3d_h = np.hstack((kp3d, np.ones((1))))  # Convert to homogeneous coordinates
        reprojections = projections @ kp3d_h.T  # Shape: (N, 3, numPts)
        # print(kp3d_h.shape, projections.shape, reprojections.shape, corners.shape)
        reprojections = reprojections[:, :2] / reprojections[:, 2:3]  # Normalize
        
        # Compute reprojection errors
        errors = np.linalg.norm(reprojections - corners, axis=1)
        inliers = np.sum(errors < threshold, axis=0)
        
        # Update best inlier count and result
        total_inliers = np.sum(inliers)
        if total_inliers > best_inliers:
            best_inliers = total_inliers
            best_kp3d = kp3d
    return best_kp3d
