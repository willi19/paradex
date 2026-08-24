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

    
def _handeye_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def handeye_residual_loss(A_batch, B_batch, X):
    """Return the batched equivalent of the original per-pose AX=XB loss."""
    AX = torch.matmul(A_batch, X)
    XB = torch.matmul(X, B_batch)
    residual = AX - XB
    return residual.square().sum() + torch.linalg.vector_norm(
        residual[:, :3, 3], dim=1
    ).sum()


def solve_ax_xb(
    A_list,
    B_list,
    init_X=None,
    max_epochs=3000,
    learning_rate=0.001,
    verbose=False,
    device=None,
):
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
                self.rotation_6d = nn.Parameter(
                    self._matrix_to_6d(init_rotation).detach().clone()
                )
            else:
                # Random initialization
                self.rotation_6d = nn.Parameter(
                    torch.randn((3, 2), dtype=torch.float64)
                )
            
            # Initialize translation
            if init_translation is not None:
                self.translation = nn.Parameter(
                    init_translation.detach().clone().to(dtype=torch.float64)
                )
            else:
                self.translation = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        
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
            b3 = torch.cross(b1, b2, dim=0)
            
            return torch.stack([b1, b2, b3], dim=1)
        
        def forward(self):
            """Returns the current 4x4 transformation matrix X"""
            rotation_matrix = self._6d_to_matrix(self.rotation_6d)
            
            # Build 4x4 transformation matrix
            X = torch.eye(
                4,
                dtype=self.translation.dtype,
                device=self.translation.device,
            )
            X[:3, :3] = rotation_matrix
            X[:3, 3] = self.translation
            
            return X
    
    if len(A_list) != len(B_list) or not A_list:
        raise ValueError("A_list and B_list must contain the same non-zero count")

    compute_device = _handeye_device(device)
    if compute_device.type == "cuda":
        torch.cuda.set_device(compute_device.index or 0)

    # One transfer and one batched kernel per operation replaces the per-pose loop.
    A_tensors = torch.as_tensor(
        np.asarray(A_list), dtype=torch.float64, device=compute_device
    )
    B_tensors = torch.as_tensor(
        np.asarray(B_list), dtype=torch.float64, device=compute_device
    )
    
    # Initialize network with Tsai-Lenz solution as starting point
    if init_X is None:
        init_X = solve_axb_cpu(np.array(A_list), np.array(B_list))
    init_R = init_X[:3, :3]
    init_t = init_X[:3, 3]
    model = HandEyeCalibrationNet(
        init_rotation=torch.as_tensor(init_R, dtype=torch.float64),
        init_translation=torch.as_tensor(init_t, dtype=torch.float64),
    ).to(compute_device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
    
    if verbose:
        print(f"Hand-eye optimizer device: {compute_device}")

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        
        # Get current transformation
        X = model()
        
        # Compute every pose-pair residual in one batched operation.
        total_loss = handeye_residual_loss(A_tensors, B_tensors, X)
        
        # Add regularization terms
        rotation_matrix = X[:3, :3]
        
        # Orthogonality constraint: R^T R = I
        identity = torch.eye(
            3,
            dtype=rotation_matrix.dtype,
            device=rotation_matrix.device,
        )
        ortho_loss = torch.norm(
            torch.matmul(rotation_matrix.T, rotation_matrix) - identity
        ) ** 2
        
        # Determinant constraint: det(R) = 1
        det_loss = (torch.det(rotation_matrix) - 1) ** 2
        
        # Combined loss
        total_loss = total_loss + 0.01 * ortho_loss + 0.01 * det_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.8f}, "
                  f"Ortho: {ortho_loss.item():.8f}, Det: {det_loss.item():.8f}")
    
    # Return final transformation as numpy array
    with torch.no_grad():
        X_final = model().detach().cpu().numpy()
    return X_final
