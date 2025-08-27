import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation as R

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
            self.translation = nn.Parameter(torch.tensor(init_translation, dtype=torch.float32))
        else:
            self.translation = nn.Parameter(torch.zeros(3))
    
    def _matrix_to_6d(self, matrix):
        """Convert 3x3 rotation matrix to 6D representation"""
        # Take first two columns of rotation matrix
        return matrix[:, :2].flatten()
    
    def _6d_to_matrix(self, d6):
        """Convert 6D representation to 3x3 rotation matrix"""
        # Reshape to get first two columns
        a1, a2 = d6[:3], d6[3:]
        
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
        X = torch.eye(4)
        X[:3, :3] = rotation_matrix
        X[:3, 3] = self.translation
        
        return X

def solve_axb_pytorch(A_list, B_list, max_epochs=5000, learning_rate=0.01, verbose=True):
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
    
    # Convert to PyTorch tensors
    A_tensors = [torch.tensor(A, dtype=torch.float32) for A in A_list]
    B_tensors = [torch.tensor(B, dtype=torch.float32) for B in B_list]
    
    # Initialize network with Tsai-Lenz solution as starting point
    try:
        init_X = solve_tsai_lenz_init(A_list, B_list)
        init_R = init_X[:3, :3]
        init_t = init_X[:3, 3]
        model = HandEyeCalibrationNet(init_rotation=torch.tensor(init_R, dtype=torch.float32),
                                    init_translation=torch.tensor(init_t, dtype=torch.float32))
    except:
        # Random initialization if Tsai-Lenz fails
        model = HandEyeCalibrationNet()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
    
    losses = []
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        
        # Get current transformation
        X = model()
        
        # Compute loss: ||AX - XB||_F^2 for all pose pairs
        total_loss = 0
        for A, B in zip(A_tensors, B_tensors):
            AX = torch.matmul(A, X)
            XB = torch.matmul(X, B)
            loss = torch.norm(AX - XB, 'fro') ** 2
            total_loss += loss
        
        # Add regularization terms
        rotation_matrix = X[:3, :3]
        
        # Orthogonality constraint: R^T R = I
        ortho_loss = torch.norm(torch.matmul(rotation_matrix.T, rotation_matrix) - torch.eye(3)) ** 2
        
        # Determinant constraint: det(R) = 1
        det_loss = (torch.det(rotation_matrix) - 1) ** 2
        
        # Combined loss
        total_loss = total_loss + 0.01 * ortho_loss + 0.01 * det_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(total_loss.item())
        
        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.8f}, "
                  f"Ortho: {ortho_loss.item():.8f}, Det: {det_loss.item():.8f}")
    
    # Return final transformation as numpy array
    with torch.no_grad():
        X_final = model().numpy()
    
    return X_final, losses

class HandEyeCalibrationNetDualQuat(nn.Module):
    """
    Alternative implementation using dual quaternions
    """
    
    def __init__(self):
        super(HandEyeCalibrationNetDualQuat, self).__init__()
        
        # Real quaternion (rotation) - normalized
        self.q_real = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        
        # Dual quaternion (translation)
        self.q_dual = nn.Parameter(torch.zeros(4))
    
    def forward(self):
        # Normalize real quaternion
        q_r = self.q_real / torch.norm(self.q_real)
        
        # Convert to transformation matrix
        w, x, y, z = q_r
        
        # Rotation matrix from quaternion
        R = torch.stack([
            torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)]),
            torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)]),
            torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
        ])
        
        # Translation from dual quaternion
        q_d = self.q_dual
        t_quat = 2 * self.quaternion_multiply(q_d, self.quaternion_conjugate(q_r))
        t = t_quat[1:4]
        
        # Build transformation matrix
        X = torch.eye(4)
        X[:3, :3] = R
        X[:3, 3] = t
        
        return X
    
    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quaternion_conjugate(self, q):
        return torch.stack([q[0], -q[1], -q[2], -q[3]])

def solve_axb_dual_quaternion(A_list, B_list, max_epochs=3000, learning_rate=0.005):
    """
    Solve AX = XB using dual quaternion representation
    """
    A_tensors = [torch.tensor(A, dtype=torch.float32) for A in A_list]
    B_tensors = [torch.tensor(B, dtype=torch.float32) for B in B_list]
    
    model = HandEyeCalibrationNetDualQuat()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        
        X = model()
        
        total_loss = 0
        for A, B in zip(A_tensors, B_tensors):
            AX = torch.matmul(A, X)
            XB = torch.matmul(X, B)
            loss = torch.norm(AX - XB, 'fro') ** 2
            total_loss += loss
        
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.8f}")
    
    with torch.no_grad():
        X_final = model().numpy()
    
    return X_final, losses

def solve_tsai_lenz_init(A_list, B_list):
    """
    Quick Tsai-Lenz solution for initialization
    """
    from scipy.linalg import svd, lstsq
    
    n = len(A_list)
    
    # Extract rotations and solve
    alpha_list = []
    beta_list = []
    
    for i in range(n-1):
        A_rel = A_list[i+1] @ np.linalg.inv(A_list[i])
        B_rel = B_list[i+1] @ np.linalg.inv(B_list[i])
        
        # Use simple axis-angle conversion
        def mat_to_axis_angle(R):
            theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
            if theta < 1e-6:
                return np.zeros(3)
            axis = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / (2 * np.sin(theta))
            return theta * axis
        
        alpha = mat_to_axis_angle(A_rel[:3, :3])
        beta = mat_to_axis_angle(B_rel[:3, :3])
        
        if np.linalg.norm(alpha) > 1e-3 and np.linalg.norm(beta) > 1e-3:
            alpha_list.append(alpha)
            beta_list.append(beta)
    
    # Solve rotation
    M = np.zeros((3, 3))
    for alpha, beta in zip(alpha_list, beta_list):
        M += np.outer(beta, alpha)
    
    U, S, Vt = svd(M)
    R_X = U @ Vt
    if np.linalg.det(R_X) < 0:
        U[:, -1] *= -1
        R_X = U @ Vt
    
    # Solve translation
    C = []
    d = []
    for A, B in zip(A_list, B_list):
        C.append(np.eye(3) - A[:3, :3])
        d.append(A[:3, 3] - R_X @ B[:3, 3])
    
    t_X = lstsq(np.vstack(C), np.concatenate(d))[0]
    
    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X
    
    return X

def test_pytorch_solver():
    """
    Test the PyTorch solver with synthetic data
    """
    print("=== Testing PyTorch AX=XB Solver ===")
    
    # Generate ground truth
    np.random.seed(42)
    R_true = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
    t_true = np.array([0.1, 0.2, 0.3])
    X_true = np.eye(4)
    X_true[:3, :3] = R_true
    X_true[:3, 3] = t_true
    
    print("Ground truth X:")
    print(X_true)
    
    # Generate test data
    A_list, B_list = [], []
    for i in range(10):
        # Random A
        R_A = R.from_euler('xyz', np.random.uniform(-90, 90, 3), degrees=True).as_matrix()
        t_A = np.random.uniform(-0.5, 0.5, 3)
        A = np.eye(4)
        A[:3, :3] = R_A
        A[:3, 3] = t_A
        
        # Corresponding B
        B = np.linalg.inv(X_true) @ A @ X_true
        
        # Add noise
        A[:3, 3] += np.random.normal(0, 0.01, 3)
        B[:3, 3] += np.random.normal(0, 0.01, 3)
        
        A_list.append(A)
        B_list.append(B)
    
    # Test PyTorch solver
    print("\n--- PyTorch 6D Rotation Method ---")
    X_est, losses = solve_axb_pytorch(A_list, B_list, max_epochs=2000, learning_rate=0.01)
    
    # Compute errors
    rot_error = np.degrees(np.arccos(np.clip((np.trace(R_true.T @ X_est[:3, :3]) - 1) / 2, -1, 1)))
    trans_error = np.linalg.norm(t_true - X_est[:3, 3])
    
    print(f"Rotation error: {rot_error:.6f} degrees")
    print(f"Translation error: {trans_error:.6f}")
    print("Final loss:", losses[-1])
    
    # Verify solution
    total_error = 0
    real_error = 0
    for A, B in zip(A_list, B_list):
        error = np.linalg.norm(A @ X_est - X_est @ B, 'fro')
        re = np.linalg.norm(A @ X_true - X_true @ B, 'fro')
        total_error += error
        real_error += re
    
    print(f"Mean verification error: {total_error / len(A_list):.8f}")
    print(f"Real verification error: {real_error / len(A_list):.8f}")
    
    print("\nEstimated X:")
    print(X_est)

if __name__ == "__main__":
    test_pytorch_solver()