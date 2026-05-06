# ------------------------------------------------------------------------------
# Hand-Eye Calibration: Solves the AX = XB equation using a rotation-translation 
# decoupling method based on the Tsai–Lenz algorithm (1989).
#
# Reference Implementation:
#     Adapted from the blog post by ROPIENS:
#     https://ropiens.tistory.com/194
#
# Problem:
#     Given a set of homogeneous transformations {A_i}, {B_i} ∈ SE(3),
#     find the unknown rigid transform X such that:
#         A_i * X = X * B_i     for all i
#
# Method:
#     1. Estimate rotation (X_rot) using the logarithmic map of SO(3) and cross products:
#            α = log(R_A),  β = log(R_B),  then accumulate M = β αᵀ terms
#        Final rotation is estimated via matrix normalization.
#     2. Estimate translation (X_trans) by solving the linear system:
#            (I - R_A) * t_X = t_A - R_X * t_B
#        using least-squares.
#
# Inputs:
#     A, B: Lists of 4×4 transformation matrices (SE(3)), with len(A) == len(B)
#
# Returns:
#     theta (np.ndarray): Estimated 3×3 rotation matrix (X_rot)
#     b_x   (np.ndarray): Estimated 3×1 translation vector (X_trans)
# ------------------------------------------------------------------------------
import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torch.optim as optim

def logR(T):
    R = T[0:3, 0:3]
    theta = np.arccos((np.trace(R) - 1)/2)
    logr = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))
    return logr

def solve_axb_cpu(A, B):
    n_data = len(A)
    M = np.zeros((3,3))
    C = np.zeros((3*n_data, 3))
    d = np.zeros((3*n_data, 1))
    A_ = np.array([])
    for i in range(n_data-1):
        alpha = logR(A[i])
        beta = logR(B[i])
        alpha2 = logR(A[i+1])
        beta2 = logR(B[i+1])
        alpha3 = np.cross(alpha, alpha2)
        beta3  = np.cross(beta, beta2)
        M1 = np.dot(beta.reshape(3,1),alpha.reshape(3,1).T)
        M2 = np.dot(beta2.reshape(3,1),alpha2.reshape(3,1).T)
        M3 = np.dot(beta3.reshape(3,1),alpha3.reshape(3,1).T)
        M += M1+M2+M3
    theta = np.dot(sqrtm(np.linalg.inv(np.dot(M.T, M))), M.T)
    for i in range(n_data):
        rot_a = A[i][0:3, 0:3]
        rot_b = B[i][0:3, 0:3]
        trans_a = A[i][0:3, 3]
        trans_b = B[i][0:3, 3]
        C[3*i:3*i+3, :] = np.eye(3) - rot_a
        d[3*i:3*i+3, 0] = trans_a - np.dot(theta, trans_b)
    b_x  = np.dot(np.linalg.inv(np.dot(C.T, C)), np.dot(C.T, d))
    T = np.eye(4)
    T[:3, :3] = theta
    T[:3, 3] = b_x.flatten()

    return T

    
def solve_ax_xb(A_list, B_list, init_X=None, max_epochs=20000, learning_rate=0.005, verbose=False):
    """
    Solve AX = XB using PyTorch gradient descent
    
    Args:
        A_list: List of 4x4 numpy arrays (camera poses)
        B_list: List of 4x4 numpy arrays (robot poses)
        max_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimization
        verbose: Print progress
    
    Returns:
        X: 4x4 numpy array (hand-to-camera transformation)
        losses: List of loss values during training
    """
    
    class HandEyeCalibrationNet(nn.Module):
        """
        PyTorch module for hand-eye calibration using gradient descent
        Solves AX = XB where X is the hand-to-camera transformation
        """
        
        def __init__(self, init_rotation=None, init_translation=None):
            super(HandEyeCalibrationNet, self).__init__()
            
            # Initialize rotation using 6D representation (more stable than quaternions)
            if init_rotation is not None:
                # Convert rotation matrix to 6D representation
                self.rotation_6d = nn.Parameter(self._matrix_to_6d(init_rotation))
            else:
                # Random initialization
                self.rotation_6d = nn.Parameter(torch.randn(6))
            
            # Initialize translation
            if init_translation is not None:
                self.translation = nn.Parameter(torch.tensor(init_translation, dtype=torch.float64))
            else:
                self.translation = nn.Parameter(torch.zeros(3))
        
        def _matrix_to_6d(self, matrix):
            """Convert 3x3 rotation matrix to 6D representation"""
            # Take first two columns of rotation matrix
            return matrix[:, :2]
        
        def _6d_to_matrix(self, d6):
            """Convert 6D representation to 3x3 rotation matrix"""
            # Reshape to get first two columns
            a1, a2 = d6[:, 0], d6[:, 1]
            
            # Gram-Schmidt process to ensure orthogonality
            b1 = a1 / torch.norm(a1)
            b2 = a2 - torch.dot(b1, a2) * b1
            b2 = b2 / torch.norm(b2)
            b3 = torch.cross(b1, b2)
            
            return torch.stack([b1, b2, b3], dim=1)
        
        def forward(self):
            """Returns the current 4x4 transformation matrix X"""
            rotation_matrix = self._6d_to_matrix(self.rotation_6d)
            
            # Build 4x4 transformation matrix
            X = torch.eye(4,dtype=torch.float64)
            X[:3, :3] = rotation_matrix
            X[:3, 3] = self.translation
            
            return X
    
    # Convert to PyTorch tensors
    A_tensors = [torch.tensor(A, dtype=torch.float64) for A in A_list]
    B_tensors = [torch.tensor(B, dtype=torch.float64) for B in B_list]
    
    # Initialize network with Tsai-Lenz solution as starting point
    if init_X is None:
        init_X = solve_axb_cpu(np.array(A_list), np.array(B_list))
    init_R = init_X[:3, :3]
    init_t = init_X[:3, 3]
    model = HandEyeCalibrationNet(init_rotation=torch.tensor(init_R, dtype=torch.float64),
                                    init_translation=torch.tensor(init_t, dtype=torch.float64))
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.7)

    losses = []
    best_loss = float('inf')
    best_X = None
    plateau_count = 0
    plateau_tol = 1e-9
    plateau_patience = 1500

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        X = model()

        # Separate rotation and translation residuals so neither dominates
        rot_loss = 0.0
        trans_loss = 0.0
        for A, B in zip(A_tensors, B_tensors):
            AX = torch.matmul(A, X)
            XB = torch.matmul(X, B)
            diff = AX - XB
            rot_loss = rot_loss + torch.norm(diff[:3, :3], 'fro') ** 2
            trans_loss = trans_loss + torch.norm(diff[:3, 3]) ** 2

        # Translation in meters is O(1e-2) squared; rotation residual is O(1).
        # Scale translation up so it actually drives the gradient.
        total_loss = rot_loss + 1000.0 * trans_loss

        rotation_matrix = X[:3, :3]
        ortho_loss = torch.norm(torch.matmul(rotation_matrix.T, rotation_matrix) - torch.eye(3)) ** 2
        det_loss = (torch.det(rotation_matrix) - 1) ** 2
        total_loss = total_loss + 0.1 * ortho_loss + 0.1 * det_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = total_loss.item()
        losses.append(loss_val)

        if loss_val < best_loss - plateau_tol:
            best_loss = loss_val
            best_X = model().detach().clone()
            plateau_count = 0
        else:
            plateau_count += 1

        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss_val:.8f} "
                  f"(rot={rot_loss.item():.6f}, trans={trans_loss.item():.6e}, "
                  f"ortho={ortho_loss.item():.6e}, det={det_loss.item():.6e})")

        if plateau_count >= plateau_patience:
            if verbose:
                print(f"Early stop at epoch {epoch}, best loss={best_loss:.8f}")
            break
    
    # Return best transformation seen during training (numpy)
    with torch.no_grad():
        X_final = best_X.numpy() if best_X is not None else model().numpy()
    return X_final