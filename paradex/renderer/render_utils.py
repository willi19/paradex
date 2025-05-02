import json
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import join_meshes_as_scene, Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import Transform3d, Translate, Rotate

from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    SoftSilhouetteShader, 
    SoftPhongShader,
    HardPhongShader,
    TexturesVertex,
    blending,
    BlendParams
)

device = torch.device("cuda")



def convert_mesho3d2py3d(o3d_mesh, device):
    # Extract vertices and faces
    vertices = torch.tensor(np.asarray(o3d_mesh.vertices), dtype=torch.float32)  # (V, 3)
    faces = torch.tensor(np.asarray(o3d_mesh.triangles), dtype=torch.int64)  # (F, 3)
    vertex_colors = torch.tensor(np.asarray(o3d_mesh.vertex_colors), dtype=torch.float32)  # (V, 3)
    vertex_colors = vertex_colors.clamp(0, 1)

    textures = TexturesVertex(verts_features=[vertex_colors])

    vertices, faces, textures = vertices.to(device), faces.to(device), textures.to(device)
    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
    
    return mesh, vertices, faces, textures


def convert_meshtrimesh2py3d(trimesh_mesh, device):
    # Extract vertices and faces

    vertices = torch.tensor(trimesh_mesh.vertices, dtype=torch.float, device=device)
    faces = torch.tensor(trimesh_mesh.faces, dtype=torch.int, device=device)
    vertex_colors = torch.tensor(trimesh_mesh.visual.vertex_colors/255, dtype=torch.float)

    textures = TexturesVertex(verts_features=[vertex_colors]).to(device)
    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
    
    return mesh, vertices, faces, textures


def combine_pytorch3d_meshes(link_list, mesh_info):
    combined_verts = []
    combined_faces = []
    combined_textures = []
    offset = 0

    for link_name in link_list:
        vertices, faces, textures = mesh_info[link_name]
        combined_verts.append(vertices)
        combined_faces.append(faces+offset)
        offset+=vertices.shape[0]
        combined_textures.append(textures.verts_features_packed()[:,:3])

    combined_textures = TexturesVertex(verts_features=[torch.vstack(combined_textures)]).to(device)
    return torch.vstack(combined_verts), torch.vstack(combined_faces), combined_textures


# 시작할때 선언
class Renderer:

    def __init__(self, extrinsic, intrinsic, Him, Wim, light_positions, device, num_face, views = [16,  8,  29,  18, 38]):
        self.views = views

        #self.intrinsic = np.concatenate([np.array(intrinsic[str(v)]["Intrinsics"]).reshape(1,3,3) for v in self.views], axis=0)
        #self.extrinsic = np.concatenate([extrinsic[str(v)][None,:] for v in self.views], axis=0)
        #self.imsize = torch.tensor([[intrinsic[str(v)]["height"],intrinsic[str(v)]["width"]] for v in self.views], device=device) 

        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.imsize = torch.tensor([[Him, Wim]])

        #R_w2c, T_w2c = torch.tensor((self.extrinsic[:, :3, :3]),  device=device, dtype=torch.float32), torch.tensor(self.extrinsic[:, :3, 3], device=device, dtype=torch.float32)
        R_w2c, T_w2c = torch.tensor(self.extrinsic[:3, :3][None,:],  device=device, dtype=torch.float32), torch.tensor(self.extrinsic[:3, 3][None,:], device=device, dtype=torch.float32)
        
        sigma = 1e-4
        cammtx = torch.tensor(self.intrinsic[None, :, :],  device=device, dtype=torch.float32)
        self.raster_settings = RasterizationSettings(
            image_size=(Him, Wim), # comes in tuple
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=10,
        )
        
        self.camera = cameras_from_opencv_projection(R=R_w2c, tvec=T_w2c, camera_matrix=cammtx, image_size=self.imsize)
        self.blend_params = blending.BlendParams(sigma=1e-8, gamma=1e-8, background_color=torch.ones((Him, Wim, 3)).to(device)*255)
        self.lights = PointLights(device=device, location=light_positions) # list of lights
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera, 
                raster_settings=self.raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=self.blend_params) #  HardPhongShader(device=device, cameras=self.camera)#  
        )    

        self.proj_matrix = torch.tensor(intrinsic @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]) @ extrinsic).to(device).float() # 3X4


    def render(self, target_lst):
        # DEBUGGING
        # plot = plot_scene({"k": {1 : self.camera, 2:target_lst}}, camera_scale=0.25)
        # plot.layout.scene.aspectmode = "data"
        # plot.show()
        return self.renderer(target_lst, cameras=self.camera, lights=self.lights)[...,3]


    def save_image(self, rendered_tensor, outpartial):
        simg = rendered_tensor.detach().cpu().numpy()[0]#[0, ...,3] # mask area
        # for i in range(simg.shape[0]):
        #     curview = self.views[i]
        #cv2.imwrite(outpartial + "_curview", simg*255)
        cv2.imwrite(outpartial, simg*255)
        return
    

    def project_joint(self, joints): # projection matrix 4X4
        coord = torch.cat((joints, torch.ones_like(joints[:,:,0:1])), axis=2)
        pix_point = torch.einsum("ij,bkj->bki", self.proj_matrix, coord)
        pix_point = pix_point[:,:,:2]/pix_point[:,:,[2]]
        return pix_point

    
    def save_process(self, mask, rendered_tensor, out_path, rgb_path, smpl_proj_joints=None, tg_proj_joints=None, tg_proj_mask=None, tips=None):
        rgbimg = cv2.resize(cv2.imread(str(rgb_path)), dsize=(mask.shape[2],mask.shape[1]))
        maskarr = mask[0].detach().cpu().numpy().astype(np.uint8)
        rgbimg[maskarr>0] = np.stack([np.clip(rgbimg[maskarr>0][:,0] + 80, 0, 255),rgbimg[maskarr>0][:,1],rgbimg[maskarr>0][:,2]], axis=1)
        renderedarr = rendered_tensor[0].detach().cpu().numpy().astype(np.uint8)
        rgbimg[renderedarr>0] = np.stack([np.clip(rgbimg[renderedarr>0][:,0] + 80, 0, 255),rgbimg[renderedarr>0][:,1],rgbimg[renderedarr>0][:,2]], axis=1)

        if tg_proj_joints is None:
            if smpl_proj_joints is not None:
                for joint in smpl_proj_joints[0]:
                    rgbimg = cv2.circle(rgbimg, (int(joint[0]),int(joint[1])), 3, (255, 0, 0), -1) # blue

        else:
            if smpl_proj_joints is not None:
                for idx in range(tg_proj_joints.shape[1]): 
                    if tg_proj_mask[0,idx,0] > 0:
                        rgbimg = cv2.circle(rgbimg, (int(smpl_proj_joints[0,idx,0]),int(smpl_proj_joints[0,idx,1])), 3, (255, 0, 0), -1) # blue

            for idx in range(tg_proj_joints.shape[1]): 
                if tg_proj_mask[0,idx,0] > 0:
                    rgbimg = cv2.circle(rgbimg, (int(tg_proj_joints[0,idx,0]),int(tg_proj_joints[0,idx,1])), 3, (0, 0, 255), -1) # Red


        if tips is not None:
            for joint_nm in tips:
                rgbimg = cv2.circle(rgbimg, (int(tips[joint_nm][0]),int(tips[joint_nm][1])), 3, (0, 255, 0), -1) # green


        cv2.imwrite(out_path, rgbimg)

            
    def save_process_befproj(self, mask, rendered_tensor, out_path, rgb_path, smpl_joints=None, rgb_joints=None, tips=None):
        rgbimg = cv2.resize(cv2.imread(str(rgb_path)), dsize=(mask.shape[2],mask.shape[1]))
        rgbimg[:,:,0]+=mask[0].detach().cpu().numpy().astype(np.uint8)*70
        rgbimg[:,:,1]+=rendered_tensor[0].detach().cpu().numpy().astype(np.uint8)*70

        if smpl_joints is not None:
            smpl_projected_joints = self.project_joint(smpl_joints)
            for joint in smpl_projected_joints[0]:
                rgbimg = cv2.circle(rgbimg, (int(joint[0]),int(joint[1])), 3, (255, 0, 0), -1) # blue

        if rgb_joints is not None:
            rgb_projected_joints = self.project_joint(rgb_joints)
            for joint in rgb_projected_joints[0]:
                rgbimg = cv2.circle(rgbimg, (int(joint[0]),int(joint[1])), 3, (0, 0, 255), -1) # Red

        if tips is not None:
            for joint_nm in tips:
                rgbimg = cv2.circle(rgbimg, (int(tips[joint_nm][0]),int(tips[joint_nm][1])), 3, (0, 255, 0), -1) # green


        cv2.imwrite(out_path, rgbimg)


    
# to global
def transform_mesh(mesh_path, transf, device):
    src_mesh = load_objs_as_meshes([mesh_path], device=device)
    transf_t = torch.tensor(transf, device=device, dtype=torch.float32)
    p3d_verts = src_mesh.verts_padded()
    p3d_faces = src_mesh.faces_padded()
    num_face = src_mesh.faces_packed().shape[0]    
    cur_color = torch.tensor([0.5,0.5,0.5]).to(device)*255
    verts_rgb = cur_color.repeat(p3d_verts.shape[1], 1)[None]
    textures = TexturesVertex(verts_features=verts_rgb.to(device))    
    v = torch.cat([p3d_verts, torch.ones_like(p3d_verts[...,0:1])], axis=-1)
    v = torch.einsum('ij, nbj -> nbi', transf_t, v)
    #v = torch.einsum('ij, nbj -> nbi', w2c_t, v)
    v = v[...,:3] / (v[..., 3:] + 1e-9)

    target_mesh = Meshes(verts=v, faces=p3d_faces, textures=textures)
    #target_mesh = join_meshes_as_scene(target_mesh, True)
    return target_mesh, src_mesh, num_face



# 시작할때 선언
class Batched_RGB_Silhouette_Renderer:
    def __init__(self, extrinsics, intrinsics, img_sizes, device):
        self.batch_size = extrinsics.shape[0]

        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.imsize = img_sizes
        
        self.imsize_tuple = (int(self.imsize[0,0].item()), int(self.imsize[0,1].item())) # height, width

        #R_w2c, T_w2c = torch.tensor((self.extrinsic[:, :3, :3]),  device=device, dtype=torch.float32), \
        # torch.tensor(self.extrinsic[:, :3, 3], device=device, dtype=torch.float32)
        R_w2c, T_w2c = self.extrinsics[:,:3, :3].clone().detach().float().to(device).requires_grad_(False), \
                    self.extrinsics[:,:3, 3].clone().detach().float().to(device).requires_grad_(False)
        
        sigma = 1e-8
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma # or 0
        cammtx = self.intrinsics.clone().detach().float().to(device).requires_grad_(False)
        # torch.tensor(self.intrinsics,  device=device, dtype=torch.float32)
        
        self.cameras = cameras_from_opencv_projection(R=R_w2c, tvec=T_w2c, \
                                                      camera_matrix=cammtx, image_size=self.imsize)

        self.lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
        # the difference between naive and coarse-to-fine rasterization. 

        # NOTE: in order to solve bin size error: look at https://github.com/facebookresearch/pytorch3d/issues/1064
        raster_settings = RasterizationSettings(
            image_size=self.imsize_tuple, 
            blur_radius=0, 
            faces_per_pixel=1, 
            bin_size=100,
            max_faces_per_bin=1000000,
        )

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model
        self.rgb_renderers = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=device, 
                cameras=self.cameras,
                lights=self.lights
            )
        )

        raster_settings_silhouette = RasterizationSettings(
            image_size=self.imsize_tuple, 
            blur_radius=blur_radius, 
            faces_per_pixel=10, 
            bin_size=100,
            max_faces_per_bin=1000000,
        )
        # Silhouette renderer 
        self.silhouette_renderers = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-8, gamma=1e-2))
        )


    def render(self, mesh, render_rgb=True):

        meshes = mesh.extend(self.batch_size)
        if render_rgb:
            rgb_images = self.rgb_renderers(meshes, lights=self.lights, cameras =self.cameras)[...,:-1]
        else:
            rgb_images = None

        silhouette_images = self.silhouette_renderers(meshes, lights=self.lights)[...,-1]

        return rgb_images, silhouette_images
        # plt.figure(figsize=(10, 10))
        # plt.imshow(images[0, ..., :3].cpu().numpy())
        # plt.axis("off")


        # images = renderer_silhouette(mesh, lights=None)
        # silhouette = images[0, ...,3]
        # plt.figure(figsize=(10, 10))
        # plt.imshow(silhouette.cpu().numpy())
        # plt.axis("off")





# if __name__ == '__main__':
#     obj_filename = '/home/jisoo/data2/paradex/meshes/spray.ply'
#     img_dir = "/home/jisoo/data2/paradex/visualization/0221_video"