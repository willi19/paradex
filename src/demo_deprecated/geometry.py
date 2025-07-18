# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np 
import math

import torch
from torch.nn import functional as F


def rot6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural
    Networks", CVPR 2019

    Args:
        rot_6d (B x 6): Batch of 6D Rotation representation.

    Returns:
        Rotation matrices (B x 3 x 3).
    """
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(rotmat):
    """
    Convert rotation matrix to 6D rotation representation.

    Args:
        rotmat (B x 3 x 3): Batch of rotation matrices.

    Returns:
        6D Rotations (B x 3 x 2).
    """
    return rotmat.view(-1, 3, 3)[:, :, :2]


def combine_verts(verts_list):
    batch_size = verts_list[0].shape[0]
    all_verts_list = [v.reshape(batch_size, -1, 3) for v in verts_list]
    verts_combined = torch.cat(all_verts_list, 1)
    return verts_combined


def center_vertices(vertices, faces, flip_y=True):
    """
    Centroid-align vertices.

    Args:
        vertices (V x 3): Vertices.
        faces (F x 3): Faces.
        flip_y (bool): If True, flips y verts to keep with image coordinates convention.

    Returns:
        vertices, faces
    """
    vertices = vertices - vertices.mean(dim=0, keepdim=True)
    if flip_y:
        vertices[:, 1] *= -1
        faces = faces[:, [2, 1, 0]]
    return vertices, faces


def compute_dist_z(verts1, verts2):
    """
    Computes distance between sets of vertices only in Z-direction.

    Args:
        verts1 (V x 3).
        verts2 (V x 3).

    Returns:
        tensor
    """
    a = verts1[:, 2].min()
    b = verts1[:, 2].max()
    c = verts2[:, 2].min()
    d = verts2[:, 2].max()
    if d >= a and b >= c:
        return 0.0
    return torch.min(torch.abs(c - b), torch.abs(a - d))


def compute_random_rotations(B=10, upright=False):
    import pytorch3d.transforms as pytransform
    """
    Randomly samples rotation matrices.

    Args:
        B (int): Batch size.
        upright (bool): If True, samples rotations that are mostly upright. Otherwise,
            samples uniformly from rotation space.

    Returns:
        rotation_matrices (B x 3 x 3).
    """
    if upright:
        a1 = torch.FloatTensor(B, 1).uniform_(0, 2 * math.pi)
        a2 = torch.FloatTensor(B, 1).uniform_(-math.pi / 6, math.pi / 6)
        a3 = torch.FloatTensor(B, 1).uniform_(-math.pi / 12, math.pi / 12)

        angles = torch.cat((a1, a2, a3), 1).cuda()
        rotation_matrices = pytransform.euler_angles_to_matrix(angles, "YXZ")
    else:
        # Reference: J Avro. "Fast Random Rotation Matrices." (1992)
        x1, x2, x3 = torch.split(torch.rand(3 * B).cuda(), B)
        tau = 2 * math.pi
        R = torch.stack(
            (  # B x 3 x 3
                torch.stack((torch.cos(tau * x1), torch.sin(
                    tau * x1), torch.zeros_like(x1)), 1),
                torch.stack((-torch.sin(tau * x1), torch.cos(
                    tau * x1), torch.zeros_like(x1)), 1),
                torch.stack((torch.zeros_like(x1), torch.zeros_like(x1),
                             torch.ones_like(x1)), 1),
            ),
            1,
        )
        v = torch.stack(
            (  # B x 3
                torch.cos(tau * x2) * torch.sqrt(x3),
                torch.sin(tau * x2) * torch.sqrt(x3),
                torch.sqrt(1 - x3),
            ),
            1,
        )
        identity = torch.eye(3).repeat(B, 1, 1).cuda()
        H = identity - 2 * v.unsqueeze(2) * v.unsqueeze(1)
        rotation_matrices = -torch.matmul(H, R)
    return rotation_matrices


def rigid_transform_3D(pt1, pt2, with_scale=False): 
    assert pt1.shape == pt2.shape
    num_rows, num_cols = pt1.shape
    if num_cols != 3:
        raise Exception(f"matrix A is not Nx3, it is {num_rows}x{num_cols}")
    num_rows, num_cols = pt2.shape
    if num_cols != 3:
        raise Exception(f"matrix B is not Nx3, it is {num_rows}x{num_cols}")

    pt1, pt2 = pt1.T, pt2.T

    centroid_A = np.mean(pt1, axis=1)
    centroid_B = np.mean(pt2, axis=1)

    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = pt1 - centroid_A
    Bm = pt2 - centroid_B

    H = Am @ np.transpose(Bm)
    
    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if with_scale:
        # Using norm of Am and Bm to compute scale
        scale = np.average(np.linalg.norm(pt2.T[1:]-pt2.T[:-1], axis=0)/np.linalg.norm(pt1.T[1:]-pt1.T[:-1], axis=1))
    else:
        scale = 1

    # special reflection case
    if np.linalg.det(R) < 0:
        
        Vt[2,:] *= -1
        R = Vt.T @ U.T
        
    t = -(scale * R) @ centroid_A + centroid_B

    T = np.eye(4)
    T[:3, :3] = scale*R
    T[:3, 3] = t[:, 0]

    return R, t, T, scale


def rigid_transform_3D_ransac(pt1, pt2, sample_ratio=0.2, with_scale=False, num_iter=500, threshold=0.02):
    best_inlier_count = 0
    best_R = None
    best_t = None
    best_T = None
    best_scale = None
    best_inliers = None
    N = pt1.shape[0]
    sample_size = max(3, int(sample_ratio * N))  # 최소 3개 이상 필요

    for _ in range(num_iter):
        idx = np.random.choice(N, sample_size, replace=False)
        R, t, _, scale = rigid_transform_3D(pt1[idx], pt2[idx], with_scale)

        # Apply transformation
        pt1_transformed = (scale * (R @ pt1.T) + t).T
        distances = np.linalg.norm(pt1_transformed - pt2, axis=1)

        inliers = distances < threshold
        inlier_count = np.sum(inliers)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_R, best_t, best_scale = R, t, scale
            best_inliers = inliers

    # Refit using best inliers
    if best_inliers is not None:
        final_R, final_t, final_T, final_scale = rigid_transform_3D(pt1[best_inliers], pt2[best_inliers], with_scale)
    else:
        return None,None,None,None,None

    return final_R, final_t, final_T, final_scale, best_inliers



def triangulate_ransac_batched(corners:dict, projections:list, sample_ratio:float, num_iter = 2000, threshold=3):
    """
    N : number of images with same marker
    corners : {1: (N,2) array, 2:(N,2) array, 3:(N,2) array, 4: (N,2) array}
    projections : list of 3x4 matrices of length N
    sample_ratio : ratio to sample, e.g. 0.8
    num_iter : number of iterations (batched)
    """
    # numImg = len(projections)
    numSamples = int(sample_ratio*len(projections))
    kp3d_out = dict()
    valid = dict()
    #print(corners)
    projections_arr = np.array(projections) # (N,3,4)
    for corner_id, kps in corners.items():
        sample_indices_batched = np.random.randint(0, len(projections), size=(num_iter, numSamples)) 
        kps_use = kps[sample_indices_batched,:] # (B, N, 2)
        proj_use = projections_arr[sample_indices_batched,:,:]
        useX, useY = kps_use[:,:,0], kps_use[:,:,1]
        first = np.multiply(useY[:,:,None], proj_use[:,:,2]) - proj_use[:,:,1]  # y*P3 - P2, (N',4)
        second = np.multiply(useX[:,:,None], proj_use[:,:,2]) - proj_use[:,:,0] # x*P3 - P1, (N',4)
        A = np.concatenate([first, second], axis=2)
        A = A.reshape(num_iter, numSamples*2,4)
        U, S, V = np.linalg.svd(A)
        kp3d = V[:,3,:][:, :3]/V[:,3,:][:, 3][:,None] # size : (B, 3)
        
        # Use all for projections
        proj_repeat = np.repeat(projections_arr[None,:,:,:], num_iter, axis=0)
        kp3d_projective = np.concatenate([kp3d, np.ones((num_iter,1))], axis=1)[:,None,:]
        repr_kp = np.einsum('BNij, Bkj->BNi',proj_repeat, kp3d_projective) # (B,N,3)
        repr_kp = repr_kp[:,:,:2]/ repr_kp[:,:,2][:,:,None]
        #repr_kp = repr_kp.reshape(num_iter, len(projections), 2) # (B, N, 2)
        pix_dist = np.linalg.norm(repr_kp - kps, axis=2) # (B,N)
        inlier_ind = pix_dist <= threshold # (B,N)
        num_inliers = np.sum(inlier_ind, axis=1) # (B)
        avg_repr_error = np.mean(pix_dist, axis=1) # (B,)

        maxInliers = np.max(num_inliers)
        maxIndices = np.argwhere(num_inliers == maxInliers).reshape(-1)
        if maxIndices.shape[0] == 1:
            whichBatch = maxIndices[0]
            using_inliers = inlier_ind[whichBatch,:]
        else:
            avg_repr_filtered = avg_repr_error[maxIndices]
            minDistind = np.argmin(avg_repr_filtered)
            whichBatch = maxIndices[minDistind]
            using_inliers = inlier_ind[whichBatch,:]
        # need 2 views at least
        if np.sum(using_inliers) >= 2:
            kps_use = kps[using_inliers]
            proj_use = projections_arr[using_inliers]
            useX, useY = kps_use[:,0], kps_use[:,1]
            first = np.multiply(useY[:,None], proj_use[:,2]) - proj_use[:,1]  # y*P3 - P2, (N',4)
            second = np.multiply(useX[:,None], proj_use[:,2]) - proj_use[:,0] # x*P3 - P1, (N',4)
            A = np.concatenate([first, second], axis=1)
            A = A.reshape(-1,4)
            U, S, V = np.linalg.svd(A)
            kp3d = V[3][0:3]/V[3][3] # (3,)        
            kp3d_out[corner_id] = kp3d
            valid[corner_id] = True
        else:
            kp3d_out[corner_id] = kp3d[whichBatch]
            valid[corner_id] = False

    return kp3d_out, valid


def transform_by_extrinsic(point, extrinsic):
    if point.ndim==1:
        return extrinsic[:3,:3]@point+extrinsic[:3,3]
    elif point.dim==2:
        return (extrinsic[:3,:3]@point.T).T+extrinsic[:3,3]
    else:
        AssertionError("Not Implemented")



def project_3d_to_2d(points3d, projections):
    # points3d : NX3
    # projections: MX3X4

    points_3d_homo = np.concatenate((points3d, np.ones((points3d.shape[0],1))), axis=-1)
    points_2d_homo = np.einsum('nd,mjd->nmj',points_3d_homo, projections)

    points_2d_homo[:,:,0]/=points_2d_homo[:,:,2]
    points_2d_homo[:,:,1]/=points_2d_homo[:,:,2]

    return points_2d_homo[:,:,:2]


def project_3d_to_2d_tensor(points3d_t, projections_t):
    # points3d : NX3
    # projections: MX3X4

    points_3d_homo = torch.concat((points3d_t, torch.ones((points3d_t.shape[0],1), device=points3d_t.device)), axis=-1)
    points_2d_homo = torch.einsum('nd,mjd->nmj',points_3d_homo, projections_t)
    
    projected_joint = points_2d_homo[...,:2].clone()
    projected_joint[...,0]/=points_2d_homo[...,2]
    projected_joint[...,1]/=points_2d_homo[...,2]
    
    return projected_joint



def optimize_3d_to_target_multiview(point_3d, projections, target_2d, device):
    from torch import nn
    '''
        point_3d : NX3
        projections: MX3X4
        target_2d : NX2    
    '''

    reprojected_2d = project_3d_to_2d(point_3d, projections)
    norm_dist = np.linalg.norm(reprojected_2d-target_2d, axis=-1)
    initial_loss = np.mean(norm_dist)

    min_loss = initial_loss

    point_3d_t = nn.Parameter(torch.tensor(point_3d, device=device).float(), requires_grad=True)
    projections_t = torch.tensor(projections, requires_grad=False).float().to(device)
    target_2d_t = torch.tensor(target_2d, requires_grad=False).float().to(device)
    
    min_3d_t = None

    learning_rate = 0.01
    epochs = 100
    optimizer = torch.optim.AdamW([point_3d_t], lr=learning_rate)

    loss_list = []
    for epoch in range(epochs):
        
        optimizer.zero_grad()

        reprojected_2d_t = project_3d_to_2d_tensor(point_3d_t, projections_t)
        loss = torch.mean(torch.linalg.norm(reprojected_2d_t-target_2d_t, axis=-1))
            
        loss_list.append(loss.item())
        loss.backward()

        optimizer.step()

        if loss.item() < min_loss:
            min_3d_t = point_3d_t.detach()
            min_loss = loss.item()

    print(f'After optimization:: original_loss: ', initial_loss,  '--> min_loss: ',min_loss)

    if min_3d_t is not None:
        return min_3d_t.detach().cpu().numpy()
    else:
        return None


def rotation_matrix_from_vectors(v1, v2):
    """
    Returns the rotation matrix that aligns v1 to v2.
    v1 and v2 are 3-element vectors.
    """
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    # Cross product and dot product
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    # Check for special cases
    if np.allclose(dot, 1.0):
        return np.eye(3)  # No rotation needed
    if np.allclose(dot, -1.0):
        # 180 degree rotation around any axis perpendicular to v1
        orthogonal = np.array([1, 0, 0]) if not np.allclose(v1, [1, 0, 0]) else np.array([0, 1, 0])
        axis = np.cross(v1, orthogonal)
        axis /= np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + 2 * (K @ K)  # since sin(π)=0, (1 - cos(π)) = 2

    # Rodrigues' formula
    K = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])

    R = np.eye(3) + K + K @ K * ((1 - dot) / (np.linalg.norm(cross) ** 2))
    return R


def generate_voxel_grid(voxel_range, resolution=0.01):
    x = np.arange(voxel_range[0,0], voxel_range[1,0], resolution)
    y = np.arange(voxel_range[0,1], voxel_range[1,1], resolution)
    z = np.arange(voxel_range[0,2], voxel_range[1,2], resolution)
    grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape((-1,3))  # Shape: (res, res, res, 3)
    return grid


def check_points_in_mask(points, mask):
    H, W = mask.shape

    xs = points[:, 0].astype(int)
    ys = points[:, 1].astype(int)

    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)

    inside_mask = np.zeros(points.shape[0], dtype=bool)
    inside_mask[valid] = mask[ys[valid], xs[valid]] > 0

    return inside_mask

import os
def get_visualhull_ctr(scene, resolution=0.01, mask_dict = None):
    camera_center_arr = np.stack([scene.camera_centers[cam_id] for cam_id in scene.camera_centers])
    voxel_range = np.stack([np.min(camera_center_arr, axis=0), np.max(camera_center_arr, axis=0)]) * 1.5
    grid = generate_voxel_grid(voxel_range, resolution=resolution*3)

    in_mask_number = np.zeros(grid.shape[0])

    # ttl_trial = 0
    for cam_id in scene.cam_ids:
        # = str(self.image_root_dir/cam_id/('%05d.jpeg'%(fidx)))
        if os.path.exists(scene.video_root_dir/f'{cam_id}.avi'):
            if mask_dict is None:
                mask = scene.get_mask(cam_id)[:,:,0]
            else:
                if cam_id in mask_dict:
                    mask = mask_dict[cam_id]
                else:
                    continue
            projected_2d = project_3d_to_2d(grid, scene.proj_matrix[cam_id][np.newaxis,...])[:,0,:].astype(np.int64) # x,y 

            in_mask = check_points_in_mask(projected_2d, mask[:,:])
            in_mask_number[in_mask]+=1
            # ttl_trial+=1

    in_mask_points = grid[in_mask_number>in_mask_number.max()*0.75]
    mean_points = np.mean(in_mask_points, axis=0)

    return in_mask_points, mean_points


import numpy as np

def pixel_to_ray(K, R, t, pixel):
    """
    Convert 2D pixel to 3D ray in world coordinates.
    """
    K_inv = np.linalg.inv(K)
    pixel_h = np.array([pixel[0], pixel[1], 1.0])  # homogeneous
    direction_camera = K_inv @ pixel_h  # in camera coordinates

    # Normalize direction
    direction_camera /= np.linalg.norm(direction_camera)

    # Transform direction to world coordinates
    direction_world = R.T @ direction_camera  # inverse rotation
    camera_center_world = -R.T @ t  # camera origin in world coordinates

    return camera_center_world, direction_world


def ray_triangle_intersect(orig, dir, v0, v1, v2):
    """
    Möller–Trumbore ray-triangle intersection algorithm.
    Returns intersection point if hit, otherwise None.
    """
    EPSILON = 1e-8
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(dir, edge2)
    a = np.dot(edge1, h)

    if -EPSILON < a < EPSILON:
        return None  # Ray is parallel to triangle

    f = 1.0 / a
    s = orig - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = f * np.dot(dir, q)

    if v < 0.0 or u + v > 1.0:
        return None

    t = f * np.dot(edge2, q)
    if t > EPSILON:
        intersection_point = orig + dir * t
        return intersection_point
    else:
        return None  # Line intersects but not ray


def raycast_mesh(K, R, t, pixel, vertices, faces):
    origin, direction = pixel_to_ray(K, R, t, pixel)

    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        hit = ray_triangle_intersect(origin, direction, v0, v1, v2)
        if hit is not None:
            return hit  # Return first hit for simplicity

    return None



def deterministic_so3_fibonacci(n):
    """Fibonacci 기반으로 SO(3) uniformly 샘플링"""
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # Golden ratio
    gr = (1 + 5 ** 0.5) / 2

    quats = []
    for i in range(n):
        theta = 2 * np.pi * i / gr
        z = 1 - 2 * (i + 0.5) / n
        r = (1 - z ** 2) ** 0.5

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        angle = 2 * np.pi * i / n
        q = np.array([np.cos(angle/2), *(np.array([x, y, z]) * np.sin(angle/2))])
        quats.append(q)

    return R.from_quat(quats)


import numpy as np
from scipy.spatial.transform import Rotation as R

def uniform_so3_fibonacci(n):
    """
    SO(3) 균일 샘플링 (deterministic, Hopf fibration 기반)
    논문 기반: Yershova et al., 2010 (Fibonacci grids on SO(3))
    """
    quats = []
    for k in range(n):
        h = -1 + 2 * (k + 0.5) / n  # height
        theta = np.arccos(h)  # inclination
        phi = 2 * np.pi * ((k + 0.5) * (1 + 5 ** 0.5) / 2 % 1)  # golden angle

        # Hopf fibration coordinates
        psi = 2 * np.pi * k / n

        qw = np.cos(theta / 2)
        qx = np.sin(theta / 2) * np.cos(phi)
        qy = np.sin(theta / 2) * np.sin(phi)
        qz = np.sin(psi / 2)

        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        quat = [qw / norm, qx / norm, qy / norm, qz / norm]
        quats.append(quat)

    return R.from_quat(quats)



def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))



def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
