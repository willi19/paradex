import numpy as np
import cv2

import trimesh
import open3d as o3d

import torch

from paradex.transforms.conversion import project

try:
    import nvdiffrast.torch as dr
except:
    pass

def project_point(verts, cammtx, image, color=(255, 0, 0)):
    vert_2d = project(cammtx, verts)
    h, w, _ = image.shape
    for v in vert_2d:
        if v[0] > w or v[0] < 0 or v[1] > h or v[1] < 0:
            continue
        image = cv2.circle(image, (int(v[0]), int(v[1])), 5, color, -1)
    return image

#----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
#----------------------------------------------------------------------------

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
    def __init__(self, intrinsics, extrinsics, near=0.01, far=2):

        # self.glctx = dr.RasterizeGLContext() if opengl else dr.RasterizeCudaContext()
        self.glctx = dr.RasterizeCudaContext()

        serial_list = list(intrinsics.keys())
        serial_list.sort()
        self.serial_list = serial_list

        cam_intrinsics = [intrinsics[serial]['intrinsics_undistort'] for serial in serial_list]
        cam_extrinsics = [extrinsics[serial] for serial in serial_list]
        width = intrinsics[serial_list[0]]['width']
        height = intrinsics[serial_list[0]]['height']

        self.width, self.height = width, height
        self.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        """
        Render with color and depth in single pass.
        """
        # 1. Rasterize
        pos_clip = transform_pos(mtx, pos)
        rast_out, _ = dr.rasterize(self.glctx, pos_clip, pos_idx, 
                                resolution=(self.height, self.width))
        
        # 2. Color interpolation
        color, _ = dr.interpolate(vtx_col, rast_out, col_idx)
        color = dr.antialias(color, rast_out, pos_clip, pos_idx)
        
        # 3. Soft mask
        ones = torch.ones_like(pos[:, :1], device=pos.device)[None]
        mask_soft, _ = dr.interpolate(ones, rast_out, pos_idx)
        mask_soft = dr.antialias(mask_soft, rast_out, pos_clip, pos_idx)
        
        # 4. Extract depth
        depth_normalized = rast_out[..., 2:3]  # (B, H, W, 1)
        depth = self._denormalize_depth(depth_normalized)
        
        # 5. Background compositing
        background_color = torch.tensor([1.0, 1.0, 1.0], device=color.device)[None, None, :]
        color = color * mask_soft + background_color * (1 - mask_soft)
        
        return color, mask_soft, depth

    def _denormalize_depth(self, depth_normalized):
        """
        Convert normalized depth [0,1] to actual distance.
        
        Normalized depth from rasterizer:
            d_norm = (1/z - 1/far) / (1/near - 1/far)
        
        Inverse:
            z = 1 / (d_norm * (1/near - 1/far) + 1/far)
        """
        near = self.near
        far = self.far
        
        # Background pixels (triangle_id = 0) have depth = 0
        # Set them to far plane
        mask = (depth_normalized > 0).float()
        
        inv_z = depth_normalized * (1/near - 1/far) + 1/far
        depth = torch.where(mask > 0, 1.0 / (inv_z + 1e-8), far)
        
        return depth
    
    def render_wtexture(self, mtx, pos, pos_idx, uv, uv_idx, tex, 
                   enable_mip=False, max_mip_level=9):
        """
        Render with texture mapping.
        
        Args:
            mtx: (B, 4, 4) projection matrices
            pos: (N, 3) vertex positions
            pos_idx: (M, 3) face indices
            uv: (B, U, 2) or (U, 2) texture coordinates
            uv_idx: (V, 3) UV face indices
            tex: (B, H, W, 3) or (H, W, 3) texture images
            enable_mip: Use mipmapping for anti-aliasing
            max_mip_level: Maximum mipmap level
            
        Returns:
            color: (B, H, W, 3)
            mask_soft: (B, H, W, 1)
            depth: (B, H, W, 1)
        """
        # Transform to clip space
        pos_clip = transform_pos(mtx, pos)
        
        # Rasterize with derivatives for mipmapping
        rast_out, rast_out_db = dr.rasterize(self.glctx, pos_clip, pos_idx, 
                                            resolution=[self.height, self.width])
        
        # Texture sampling
        if enable_mip:
            texc, texd = dr.interpolate(uv, rast_out, uv_idx, 
                                        rast_db=rast_out_db, diff_attrs='all')
            color = dr.texture(tex, texc, texd, 
                            filter_mode='linear-mipmap-linear', 
                            max_mip_level=max_mip_level)
        else:
            texc, _ = dr.interpolate(uv, rast_out, uv_idx)
            if tex.ndim == 3:
                tex = tex.unsqueeze(0)
            color = dr.texture(tex, texc, filter_mode='linear')
        
        # Soft mask
        ones = torch.ones_like(pos[:, :1], device=pos.device)[None]  # [1, N, 1]
        mask_soft, _ = dr.interpolate(ones, rast_out, pos_idx)
        mask_soft = dr.antialias(mask_soft, rast_out, pos_clip, pos_idx)
        
        # Extract and denormalize depth
        depth_normalized = rast_out[..., 2:3]  # (B, H, W, 1)
        depth = self._denormalize_depth(depth_normalized)
        
        # Background compositing
        background_color = torch.tensor([1.0, 1.0, 1.0], device=color.device)[None, None, :]
        color = color * mask_soft + background_color * (1.0 - mask_soft)
        
        return color, mask_soft, depth

    def render(self, mesh):
        '''
            texture_type == 'vectex_color'
                obj_dict = {'type':'vertex_color' ,'vtx_pos':, 'pos_idx':, 'vtx_col':, 'col_idx':}
            texture_type == 'triangle_uvs'
                obj_dict = {'type': triangle_uvs, 'vtx_pos':, 'pos_idx':, 'uvs':, 'uv_idx', 'texture' }
        '''
        obj_dict = mesh_to_obj_dict(mesh)
        if obj_dict['type'] == 'vertex_color':
            # render_wvertexcolor(self, mtx, pos, pos_idx, col, col_idx):
            color, mask_soft, depth = self.render_wvertexcolor(mtx=self.intr_opengl@self.flip_z@self.cam_extrs_t,\
                        pos=obj_dict['verts'][0], pos_idx=obj_dict['faces'], \
                        vtx_col=obj_dict['vtx_col'], col_idx=obj_dict['col_idx'])
        elif obj_dict['type'] == 'triangle_uvs':
            color, mask_soft, depth = self.render_wtexture(self.intr_opengl@self.flip_z@self.cam_extrs_t, \
                        pos=obj_dict['verts'][0], pos_idx=obj_dict['faces'], \
                        uv=obj_dict['uvs'], uv_idx=obj_dict['uv_idx'], tex=obj_dict['texture_tensor'])

        '''
            color: B X H X W X 3
            depth: B X H X W X 1
        '''
        # torch to numpy
        
        color = torch.flip(color,dims=[1]).cpu().numpy()
        mask_soft = torch.flip(mask_soft, dims=[1]).cpu().numpy()
        depth = torch.flip(depth, dims=[1]).cpu().numpy()

        color_dict = {serial: color[i] for i, serial in enumerate(self.serial_list)}
        mask_dict = {serial: mask_soft[i] for i, serial in enumerate(self.serial_list)}
        depth_dict = {serial: depth[i] for i, serial in enumerate(self.serial_list)}

        return color_dict, mask_dict, depth_dict

    def render_multi(self, mesh_list):
        """
        Render multiple meshes in a single pass.
        
        Args:
            mesh_list: List of Trimesh or Open3D meshes
            return_id: If True, return instance ID map
            return_torch: Return torch tensors instead of numpy
            
        Returns:
            color_dict: {serial: (H, W, 3)} RGB images
            mask_dict: {serial: (H, W, 1)} soft masks
            depth_dict: {serial: (H, W, 1)} depth maps
            id_dict: {serial: (H, W, 3)} optional ID maps (1-indexed, 0=background)
            
        Example:
            >>> meshes = [apple_mesh, orange_mesh, banana_mesh]
            >>> colors, masks, depths, ids = renderer.render_multi(
            ...     meshes, return_id=True
            ... )
            >>> # ids['cam_01'][y, x] = object ID at pixel (x, y)
        """
        # Convert meshes to obj_dicts
        obj_dicts = [mesh_to_obj_dict(m, device=self.device) for m in mesh_list]
        
        mtx = self.intr_opengl @ self.flip_z @ self.cam_extrs_t
        
        # Concatenate all meshes
        stacked_pos = []
        stacked_pos_idx = []
        stacked_col = []
        stacked_id = []
        accumulated_vertex_numb = 0
        
        for mesh_id, obj_dict in enumerate(obj_dicts):
            # Vertices
            pos = obj_dict['verts'][0]  # (V, 3)
            stacked_pos.append(pos)
            
            # Face indices with offset
            pos_idx = obj_dict['faces'].clone()
            pos_idx += accumulated_vertex_numb
            stacked_pos_idx.append(pos_idx)
            
            # Colors (actual vertex colors from mesh)
            if 'vtx_col' in obj_dict:
                vtx_col = obj_dict['vtx_col']
            else:
                # Default gray if no color
                vtx_col = torch.ones_like(pos) * 0.5
            stacked_col.append(vtx_col)
            
            # IDs for segmentation
            vtx_id = torch.ones_like(pos) * (mesh_id + 1.0)
            stacked_id.append(vtx_id)
            
            accumulated_vertex_numb += pos.shape[0]
        
        # Stack
        stacked_pos = torch.vstack(stacked_pos)
        stacked_pos_idx = torch.vstack(stacked_pos_idx)
        stacked_col = torch.vstack(stacked_col)
        stacked_id = torch.vstack(stacked_id)
        
        # Transform and rasterize
        pos_clip = transform_pos(mtx, stacked_pos)
        rast_out, _ = dr.rasterize(self.glctx, pos_clip, stacked_pos_idx,
                                   resolution=(self.height, self.width))
        
        # Interpolate colors
        color, _ = dr.interpolate(stacked_col, rast_out, stacked_pos_idx)
        color = dr.antialias(color, rast_out, pos_clip, stacked_pos_idx)
        
        # Soft mask
        ones = torch.ones_like(stacked_pos[:, :1], device=stacked_pos.device)[None]
        mask_soft, _ = dr.interpolate(ones, rast_out, stacked_pos_idx)
        mask_soft = dr.antialias(mask_soft, rast_out, pos_clip, stacked_pos_idx)
        
        # Depth
        depth_normalized = rast_out[..., 2:3]
        depth = self._denormalize_depth(depth_normalized)
        
        # Background compositing
        background_color = torch.tensor([1.0, 1.0, 1.0], device=color.device)[None, None, :]
        color = color * mask_soft + background_color * (1 - mask_soft)
        
        # Flip
        color = torch.flip(color, dims=[1])
        mask_soft = torch.flip(mask_soft, dims=[1])
        depth = torch.flip(depth, dims=[1])
        
        
        color_np = color.cpu().numpy()
        mask_np = mask_soft.cpu().numpy()
        depth_np = depth.cpu().numpy()
        color_dict = {s: color_np[i] for i, s in enumerate(self.serial_list)}
        mask_dict = {s: mask_np[i] for i, s in enumerate(self.serial_list)}
        depth_dict = {s: depth_np[i] for i, s in enumerate(self.serial_list)}
        
        
        id_map, _ = dr.interpolate(stacked_id, rast_out, stacked_pos_idx)
        id_map = torch.flip(id_map, dims=[1])
        
        id_map_np = id_map.cpu().numpy()
        id_dict = {s: id_map_np[i] for i, s in enumerate(self.serial_list)}
        
        return color_dict, mask_dict, depth_dict, id_dict
