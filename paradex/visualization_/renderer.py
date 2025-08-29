# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
torch.multiprocessing.set_start_method('forkserver',force=True)

#----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
#----------------------------------------------------------------------------
def transform_pos(mtx, pos):
    '''
        pos: batch_num X J X 3
    '''
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def intr_opencv_to_opengl_proj(K, width, height, near=0.2, far=2.0):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 2 * fx / width
    proj[1, 1] = 2 * fy / height
    proj[0, 2] = 1 - 2 * cx / width
    proj[1, 2] = 2 * cy / height - 1
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1

    return proj

# def convect_opencv2_opengl_proj(cam_w2c, cam_intr, width, height, n=0.2, f=3.0):
#     flip_z = np.diag([1, -1, -1]).astype(np.float32)

#     # Convert to camera-to-world
#     R_c2w = cam_w2c[:3,:3].T
#     T_c2w = -cam_w2c[:3,:3].T @ cam_w2c[:3,3]

#     # Apply coordinate system flip
#     R_c2w = flip_z @ R_c2w
#     T_c2w = flip_z @ T_c2w

#     cam2world_opengl = np.eye(4, dtype=np.float32)
#     cam2world_opengl[:3, :3] = R_c2w
#     cam2world_opengl[:3, 3] = T_c2w

#     intr_opengl = intr_opencv_to_opengl_proj(cam_intr, width, height, n, f)

#     return intr_opengl, cam2world_opengl


def projection(x=0.1, n=1.0, f=50.0):
    '''
        x = tan(fov / 2) i
    '''
    return np.array([[n/x,    0,            0,              0],
                     [  0,  n/x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)


def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]).astype(np.float32)


def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0],
                     [0,  c, s, 0],
                     [0, -s, c, 0],
                     [0,  0, 0, 1]]).astype(np.float32)


def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)


def random_rotation_translation(t):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return m

#----------------------------------------------------------------------------
# Bilinear downsample by 2x.
#----------------------------------------------------------------------------

def bilinear_downsample(x):
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    w = w.expand(x.shape[-1], 1, 4, 4) 
    x = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1])
    return x.permute(0, 2, 3, 1)

#----------------------------------------------------------------------------
# Image display function using OpenGL.
#----------------------------------------------------------------------------

_glfw_window = None
def display_image(image, zoom=None, size=None, title=None): # HWC
    # Import OpenGL and glfw.
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image)
    if size is not None:
        assert zoom is None
        zoom = max(1, size // image.shape[0])
    if zoom is not None:
        image = image.repeat(zoom, axis=0).repeat(zoom, axis=1)
    height, width, channels = image.shape

    # Initialize window.
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.init()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

#----------------------------------------------------------------------------
# Image save helper.
#----------------------------------------------------------------------------

def save_image(fn, x):
    import imageio
    x = np.rint(x * 255.0)
    x = np.clip(x, 0, 255).astype(np.uint8)
    imageio.imsave(fn, x)

#----------------------------------------------------------------------------


# Unit quaternion.
def q_unit():
    return np.asarray([1, 0, 0, 0], np.float32)

# Get a random normalized quaternion.
def q_rnd():
    u, v, w = np.random.uniform(0.0, 1.0, size=[3])
    v *= 2.0 * np.pi
    w *= 2.0 * np.pi
    return np.asarray([(1.0-u)**0.5 * np.sin(v), (1.0-u)**0.5 * np.cos(v), u**0.5 * np.sin(w), u**0.5 * np.cos(w)], np.float32)

# Get a random quaternion from the octahedral symmetric group S_4.
_r2 = 0.5**0.5
_q_S4 = [[ 1.0, 0.0, 0.0, 0.0], [ 0.0, 1.0, 0.0, 0.0], [ 0.0, 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0, 1.0],
         [-0.5, 0.5, 0.5, 0.5], [-0.5,-0.5,-0.5, 0.5], [ 0.5,-0.5, 0.5, 0.5], [ 0.5, 0.5,-0.5, 0.5],
         [ 0.5, 0.5, 0.5, 0.5], [-0.5, 0.5,-0.5, 0.5], [ 0.5,-0.5,-0.5, 0.5], [-0.5,-0.5, 0.5, 0.5],
         [ _r2,-_r2, 0.0, 0.0], [ _r2, _r2, 0.0, 0.0], [ 0.0, 0.0, _r2, _r2], [ 0.0, 0.0,-_r2, _r2],
         [ 0.0, _r2, _r2, 0.0], [ _r2, 0.0, 0.0,-_r2], [ _r2, 0.0, 0.0, _r2], [ 0.0,-_r2, _r2, 0.0],
         [ _r2, 0.0, _r2, 0.0], [ 0.0, _r2, 0.0, _r2], [ _r2, 0.0,-_r2, 0.0], [ 0.0,-_r2, 0.0, _r2]]
def q_rnd_S4():
    return np.asarray(_q_S4[np.random.randint(24)], np.float32)

# Quaternion slerp.
def q_slerp(p, q, t):
    d = np.dot(p, q)
    if d < 0.0:
        q = -q
        d = -d
    if d > 0.999:
        a = p + t * (q-p)
        return a / np.linalg.norm(a)
    t0 = np.arccos(d)
    tt = t0 * t
    st = np.sin(tt)
    st0 = np.sin(t0)
    s1 = st / st0
    s0 = np.cos(tt) - d*s1
    return s0*p + s1*q

# Quaterion scale (slerp vs. identity quaternion).
def q_scale(q, scl):
    return q_slerp(q_unit(), q, scl)

# Quaternion product.
def q_mul(p, q):
    s1, V1 = p[0], p[1:]
    s2, V2 = q[0], q[1:]
    s = s1*s2 - np.dot(V1, V2)
    V = s1*V2 + s2*V1 + np.cross(V1, V2)
    return np.asarray([s, V[0], V[1], V[2]], np.float32)

# Angular difference between two quaternions in degrees.
def q_angle_deg(p, q):
    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()
    d = np.abs(np.dot(p, q))
    d = min(d, 1.0)
    return np.degrees(2.0 * np.arccos(d))

# Quaternion product
def q_mul_torch(p, q):
    a = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    b = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    c = p[0]*q[2] + p[2]*q[0] + p[3]*q[1] - p[1]*q[3]
    d = p[0]*q[3] + p[3]*q[0] + p[1]*q[2] - p[2]*q[1]
    return torch.stack([a, b, c, d])

# Convert quaternion to 4x4 rotation matrix.
def q_to_mtx(q):
    r0 = torch.stack([1.0-2.0*q[1]**2 - 2.0*q[2]**2, 2.0*q[0]*q[1] - 2.0*q[2]*q[3], 2.0*q[0]*q[2] + 2.0*q[1]*q[3]])
    r1 = torch.stack([2.0*q[0]*q[1] + 2.0*q[2]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[2]**2, 2.0*q[1]*q[2] - 2.0*q[0]*q[3]])
    r2 = torch.stack([2.0*q[0]*q[2] - 2.0*q[1]*q[3], 2.0*q[1]*q[2] + 2.0*q[0]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[1]**2])
    rr = torch.transpose(torch.stack([r0, r1, r2]), 1, 0)
    rr = torch.cat([rr, torch.tensor([[0], [0], [0]], dtype=torch.float32).cuda()], dim=1) # Pad right column.
    rr = torch.cat([rr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()], dim=0)  # Pad bottom row.
    return rr

# Transform vertex positions to clip space
def transform_pos(mtx, pos):
    # mtx: matrix for tranformation
    # pos: posisions 
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    if t_mtx.dim() == 2:
        t_mtx = t_mtx[None, ...]
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.einsum('nj,bij->bni',posw,t_mtx).contiguous()



from pathlib import Path
import copy
import sys
import torch
import numpy as np 

import open3d as o3d
import trimesh
import nvdiffrast.torch as dr

def mesh_to_obj_dict(mesh, device='cuda', texture_type='vertex_color'):
    """
    Converts a Trimesh or Open3D mesh into a dict compatible with BatchRenderer.

    Args:
        mesh: trimesh.Trimesh or open3d.geometry.TriangleMesh
        device: 'cuda' or 'cpu'
        texture_type: 'vertex_color' or 'triangle_uvs'

    Returns:
        obj_dict: dictionary with keys compatible with BatchRenderer
    """
    import torch
    import numpy as np

    if isinstance(mesh, o3d.geometry.TriangleMesh):
        mesh.compute_vertex_normals()
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        vtx_col = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.ones_like(verts)

    elif isinstance(mesh, trimesh.Trimesh):
        verts = mesh.vertices
        faces = mesh.faces
        if texture_type == 'vertex_color':
            if mesh.visual.kind == 'vertex' and hasattr(mesh.visual, 'vertex_colors'):
                vtx_col = mesh.visual.vertex_colors[:, :3] / 255.0  # RGBA → RGB
            else:
                vtx_col = np.zeros_like(verts)
        elif texture_type == 'triangle_uvs':
            assert mesh.visual.kind == 'texture', "Mesh does not have texture UVs"
            uvs = mesh.visual.uv
            uv_idx = faces.copy()
            texture_image = mesh.visual.material.image
            texture_tensor = torch.tensor(np.asarray(texture_image)).float() / 255.0
            texture_tensor = texture_tensor.to(device).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW

    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")

    verts = torch.tensor(verts, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N, 3]
    faces = torch.tensor(faces, dtype=torch.int32, device=device)                 # [F, 3]

    obj_dict = {
        'verts': verts,
        'faces': faces
    }

    if texture_type == 'vertex_color':
        vtx_col = torch.tensor(vtx_col, dtype=torch.float32, device=device)
        if vtx_col.max() > 1.0:
            vtx_col = vtx_col / 255.0
        obj_dict.update({
            'type': 'vertex_color',
            'vtx_col': vtx_col,
            'col_idx': faces.clone()
        })

    elif texture_type == 'triangle_uvs':
        obj_dict.update({
            'type': 'triangle_uvs',
            'uvs': torch.tensor(uvs, dtype=torch.float32, device=device).unsqueeze(0),  # [1, N, 2]
            'uv_idx': torch.tensor(uv_idx, dtype=torch.int32, device=device),
            'texture_tensor': texture_tensor.permute(0, 2, 3, 1),  # BxHxWx3
        })

    return obj_dict


class BatchRenderer:
    def __init__(self, cam_intrinsics, cam_extrinsics,  width, height, near=0.01, far=2, device=None):
        '''
            cam_intrinsic = scene.cam2intr[test_cam_id]
            cam_extrinsics = list of cam_extrinsic (scene.cam2extr[cam_id])

        '''
        # self.glctx = dr.RasterizeGLContext() if opengl else dr.RasterizeCudaContext()
        self.glctx = dr.RasterizeCudaContext()

        self.width, self.height = width, height
        self.device = device

        cam_extrs = []
        for cam_extrinsic in cam_extrinsics:
            org_extr = np.eye(4)
            org_extr[:3,:] = cam_extrinsic
            org_extr_t = torch.tensor(org_extr).to(device).float()
            cam_extrs.append(org_extr_t)
        self.cam_extrs_t = torch.stack(cam_extrs)
        self.batch_size = self.cam_extrs_t.shape[0]

        self.intr_opengl = torch.stack( [ torch.tensor(intr_opencv_to_opengl_proj(cam_intrinsic, width, height, \
                                                near=near, far=far).astype(np.float32)).to(device) for cam_intrinsic in cam_intrinsics
                                        ] 
                                        ) 
        self.flip_z = torch.tensor(np.diag([1, -1, -1, 1]).astype(np.float32)).to(device)


    def render_wvertexcolor(self, mtx, pos, pos_idx, vtx_col, col_idx):
        # Setup TF graph for reference.
        pos_clip    = transform_pos(mtx, pos)
        rast_out, _ = dr.rasterize(self.glctx, pos_clip, pos_idx, resolution=(self.height, self.width))
        color   , _ = dr.interpolate(vtx_col, rast_out, col_idx)
        color       = dr.antialias(color, rast_out, pos_clip, pos_idx)

        ones = torch.ones_like(pos[:, :1], device=pos.device)[None]  # [1, N, 1]
        mask_soft, _ = dr.interpolate(ones, rast_out, pos_idx)
        mask_soft = dr.antialias(mask_soft, rast_out, pos_clip, pos_idx)  # [1, H, W]

        background_color = torch.tensor([1.0, 1.0, 1.0], device=color.device)[None, None, :]  # [1, 1, 3]
        color = color * mask_soft + background_color * (1 - mask_soft)

        return color, mask_soft


    def render_wtexture(self, mtx, pos, pos_idx, uv, uv_idx, tex, enable_mip=False, max_mip_level=9):
        '''
            with enable_mip
            mtx : BX4X3
            pos : NX3
            pos_idx : MX3
            uv: BXUX2
            uv_idx: VX3
            tex: BXHXWX3

        '''
        
        pos_clip = transform_pos(mtx, pos) # position in clip space 
        #

        rast_out, rast_out_db = dr.rasterize(self.glctx, pos_clip, pos_idx, resolution=[self.height, self.width])

        if enable_mip:
            texc, texd = dr.interpolate(uv, rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
            color = dr.texture(tex, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
        else:
            texc, _ = dr.interpolate(uv, rast_out, uv_idx)
            if tex.ndim==3:
                tex = tex.unsqueeze(0) 
            color = dr.texture(tex, texc, filter_mode='linear')

        # Soft mask: differentiable visibility mask
        ones = torch.ones_like(pos[:, :1], device=pos.device)[None]  # [1, N, 1]
        mask_soft, _ = dr.interpolate(ones, rast_out, pos_idx)
        mask_soft = dr.antialias(mask_soft, rast_out, pos_clip, pos_idx)  # [1, H, W]

        background_color = torch.tensor([1.0, 1.0, 1.0], device=color.device)[None, None, :]  # [1, 1, 3]
        color = color * mask_soft + background_color * (1.0 - mask_soft)

        return color, mask_soft


    def render(self, mesh, render_rgb=True):
        '''
            texture_type == 'vectex_color'
                obj_dict = {'type':'vertex_color' ,'vtx_pos':, 'pos_idx':, 'vtx_col':, 'col_idx':}
            texture_type == 'triangle_uvs'
                obj_dict = {'type': triangle_uvs, 'vtx_pos':, 'pos_idx':, 'uvs':, 'uv_idx', 'texture' }

            The parameter render_rgb is not used 
        '''
        obj_dict = mesh_to_obj_dict(mesh)
        if obj_dict['type'] == 'vertex_color':
            # render_wvertexcolor(self, mtx, pos, pos_idx, col, col_idx):
            color, mask_soft = self.render_wvertexcolor(mtx=self.intr_opengl@self.flip_z@self.cam_extrs_t,\
                        pos=obj_dict['verts'][0], pos_idx=obj_dict['faces'], \
                        vtx_col=obj_dict['vtx_col'], col_idx=obj_dict['col_idx'])
        elif obj_dict['type'] == 'triangle_uvs':
            color, mask_soft = self.render_wtexture(self.intr_opengl@self.flip_z@self.cam_extrs_t, \
                        pos=obj_dict['verts'][0], pos_idx=obj_dict['faces'], \
                        uv=obj_dict['uvs'], uv_idx=obj_dict['uv_idx'], tex=obj_dict['texture_tensor'])

        '''
            color: B X H X W X 3
            depth: B X H X W X 1
        '''



        return torch.flip(color,dims=[1]), torch.flip(mask_soft, dims=[1])
    

    def render_id(self, obj_dict_list): # return mask with id in order 
        # return rendered_ids on image (acted like silhouette)
        
        # pos=obj_dict['verts'][0]
        # pos_idx=obj_dict['faces']
        # col_idx = obj_dict['col_idx']
        # pos_clip    = transform_pos(mtx, pos)
        # vtx_col=torch.ones_like(obj_dict['vtx_col'])
        # rast_out, _ = dr.rasterize(self.glctx, pos_clip, pos_idx, resolution=(self.height, self.width))
        # color   , _ = dr.interpolate(vtx_col, rast_out, col_idx)
        # color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
        mtx = self.intr_opengl@self.flip_z@self.cam_extrs_t

        stacked_pos = []
        stacked_pos_idx = []
        stacked_attr = []
        accumulated_vertex_numb = 0
        for mesh_id, obj_dict in enumerate(obj_dict_list):
            # pos
            pos = obj_dict['verts'][0] # VX3 float
            stacked_pos.append(pos)
            # pos.idx
            pos_idx = (obj_dict['faces']).clone().detach() # FX3 torch int32
            pos_idx += accumulated_vertex_numb
            stacked_pos_idx.append(pos_idx)

            # col_idx = FX3 torch in32
            # vtx_color VX3 float
            vtx_col = (torch.ones_like(obj_dict['vtx_col'])*(mesh_id+1.0)).float()
            stacked_attr.append(vtx_col)

            accumulated_vertex_numb+=pos.shape[0]

        stacked_pos = torch.vstack(stacked_pos)
        stacked_pos_idx = torch.vstack(stacked_pos_idx)
        stacked_attr = torch.vstack(stacked_attr)

        pos_clip    = transform_pos(mtx, stacked_pos)
        rast_out, _ = dr.rasterize(self.glctx, pos_clip, stacked_pos_idx, resolution=(self.height, self.width))
        stacked_attr_interpolated   , _ = dr.interpolate(stacked_attr, rast_out, stacked_pos_idx)
        # silhouette = dr.antialias(stacked_attr_interpolated, rast_out, pos_clip, stacked_pos_idx)
        silhouette = torch.flip(stacked_attr_interpolated, dims=[1])
        
        return silhouette # offset scale
    

    def get_rendered_faces(self, obj_dict):
        mtx=self.intr_opengl@self.flip_z@self.cam_extrs_t
        pos=obj_dict['verts'][0]
        pos_idx=obj_dict['faces']

        pos_clip = transform_pos(mtx, pos) # position in clip space 
        #

        rast_out, rast_out_db = dr.rasterize(self.glctx, pos_clip, pos_idx, resolution=[self.height, self.width])
        rast_out = torch.flip(rast_out, dims=[1])

        return rast_out[...,3]-1 # Triangle Index






