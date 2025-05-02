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

def logR(T):
    R = T[0:3, 0:3]
    theta = np.arccos((np.trace(R) - 1)/2)
    logr = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))
    return logr

def solve(A, B):
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
        M = M1+M2+M3
    theta = np.dot(sqrtm(np.linalg.inv(np.dot(M.T, M))), M.T)
    for i in range(n_data):
        rot_a = A[i][0:3, 0:3]
        rot_b = B[i][0:3, 0:3]
        trans_a = A[i][0:3, 3]
        trans_b = B[i][0:3, 3]
        C[3*i:3*i+3, :] = np.eye(3) - rot_a
        d[3*i:3*i+3, 0] = trans_a - np.dot(theta, trans_b)
    b_x  = np.dot(np.linalg.inv(np.dot(C.T, C)), np.dot(C.T, d))
    return theta, b_x