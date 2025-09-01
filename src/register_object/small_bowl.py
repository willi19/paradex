import pickle
import os
import trimesh
import numpy as np
import cv2
import torch

from paradex.utils.env import get_pcinfo
from paradex.utils.file_io import shared_dir, home_path, load_latest_C2R, load_current_camparam, rsc_path, find_latest_index
from paradex.io.capture_pc.connect import run_script
from paradex.image.projection import get_cammtx, project_mesh_nvdiff, project
from paradex.image.aruco import triangulate_marker, draw_aruco, detect_aruco, undistort_img
from paradex.visualization_.renderer import BatchRenderer
from paradex.image.overlay import overlay_mask
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.geometry.math import rigid_transform_3D
from paradex.inference.object_6d import get_current_object_6d
from paradex.pose_utils.optimize_initial_frame import object6d_silhouette
from paradex.pose_utils.optimize_module import optimize_with_initial_sRt, get_gt_data, \
                                        load_obj_masks, parse_and_crop, render_and_compute_loss_all_cams
from paradex.pose_utils.vis_utils import crop_and_resize_by_mask, make_grid_image_np, parse_objectmesh_objdict, putText
from paradex.pose_utils.pre_render_obj import get_rendered_obj

def object6d_silhouette(
    scene_path,
    obj_name, 
    renderer_type = 'nvdiffrast',
    wo_simplify = False,
    rescale_factor = 0.25,
    confidence = 0.002,
    learning_rate = 1e-2,
    iou_weight = 0.5,
    tg_weight = 10.0,
    use_rgb = False,
    image_dir = None
    
    
):
    print(f"Initial 6D for {scene_path}")

    # obj_name = scene_path.split("/")[-2] 
    device = torch.device("cuda:0")
    obj_dict = parse_objectmesh_objdict(obj_name, renderer_type=renderer_type, device=device, simplify=(not wo_simplify))
    scaled = obj_dict['scaled']
    default_scale = 1.0

    assert scaled, 'You should use scaled mesh'

    # Get Prerendered object mask.
    rendered_obj = get_rendered_obj(obj_nm=obj_name, renderer_type='nvdiff', device=device)
    rendered_mask = np.zeros_like(rendered_obj['rendered_sils'])
    rendered_mask[rendered_obj['rendered_sils']>0] = 1

    mask_sub_dir = 'mask_hq/%s'%(obj_name)
    mask_root = Path(scene_path)/mask_sub_dir
    rescale_factor = rescale_factor # image rescale

    org_scene = Scene(scene_path=Path(scene_path), rescale_factor=1.0, mask_dir_nm=mask_root, device=device, mask_module='yolo', use_sam=True, dino_obj_nm=obj_name)
    rescaled_scene = Scene(scene_path=Path(scene_path), rescale_factor=rescale_factor, mask_dir_nm=mask_root, device=device, mask_module=None, use_sam=False)
    rescaled_scene.get_renderer()
    
    mask_dict_org, top_n_cams, top_n_confidence = load_obj_masks(org_scene, reload_mask=False, debug=True, confidence=confidence, image_dir = image_dir)
    mask_dict_scaled = {cam_id: get_binary_mask(cv2.resize( mask_dict_org[cam_id].astype(np.uint8), dsize=(rescaled_scene.width, rescaled_scene.height), interpolation=cv2.INTER_LINEAR)) for cam_id in top_n_cams}

    # Set Renderer and Prepare GT data
    rescaled_scene.get_batched_renderer(top_n_cams, renderer_type=renderer_type)
    obj_mask_t_dict, obj_masked_rgb_t_dict, obj_mask_stackted_t, gt_combined_mask, obj_masked_rgb_stacked_t, \
                gt_combined_maksed_rgb = get_gt_data(rescaled_scene, mask_root, top_n_cams, device, computed_mask=mask_dict_scaled, fidx=0, image_dir=image_dir)
    
    # Parse GT mask
    tg_rgbs, tg_sils, tg_rgbs_full, tg_sils_full = parse_and_crop(obj_mask_t_dict, obj_masked_rgb_t_dict)

    # Get Initial Translation
    initial_translate = get_bbox_center(org_scene, mask_dict=mask_dict_org)

    object_output_dir = Path(scene_path)/(f'{obj_name}_optim')
    os.makedirs(object_output_dir, exist_ok=True)
    object_final_output_dir = object_output_dir/'final'
    os.makedirs(object_final_output_dir, exist_ok=True)

    R_list = []
    rgb_loss_list = []
    sil_loss_list = []

    # Compute Best Rotation Candidate for Each Camera.
    for cidx, cam_id in enumerate(obj_mask_t_dict):
        tg_rgb, tg_sil = tg_rgbs[cidx], tg_sils[cidx]
        masked_palette = (rendered_obj['rendered_rgbs']*rendered_mask)
        masked_rgb = np.copy(tg_rgb)
        masked_rgb[tg_sil<=0]=0

        diff = np.linalg.norm((masked_palette-masked_rgb).reshape(masked_palette.shape[0],-1), axis=1)
        cv2.imwrite('test.png', np.vstack([masked_palette[np.argmin(diff)],masked_rgb]))

        sil_diff = np.linalg.norm((rendered_obj['rendered_sils']-tg_sil).reshape(masked_palette.shape[0],-1), axis=1)
        sil_min_idxes = sorted(range(len(sil_diff)), key=lambda i: sil_diff[i])
        for sil_min_idx in sil_min_idxes[:100]:
            cv2.imwrite(f'test/test{sil_min_idx}_{sil_diff[sil_min_idx]}.png', np.vstack([rendered_obj['rendered_sils'][sil_min_idx],tg_sil]))

        min_rgb = np.argmin(diff)
        rotmat = org_scene.cam2extr_t[cam_id][:3,:3].T@rendered_obj['rotmats'][min_rgb]

        # TODO add filtering here
        # Render to all cams and configure best candidate.

        rendered_rgbs, rendered_rgbs_full, rendered_sils, rendered_sils_full, \
                    rgb_diff_full, sil_diff_full, rgb_diff, sil_diff, \
                    rgb_filter, rgb_loss, sil_filter, sil_loss = \
                                        render_and_compute_loss_all_cams(obj_dict, rescaled_scene, \
                                        matrix_to_rotation_6d(rotmat), initial_translate, default_scale, \
                                        tg_rgbs, tg_sils, tg_rgbs_full, tg_sils_full, turn_on_filter= True, device=device)
        
        R_list.append(rotmat)
        rgb_loss_list.append(rgb_loss)
        sil_loss_list.append(sil_loss)

    rgb_filter = rgb_loss_list<=np.percentile(rgb_loss_list, 80)
    sil_filter = sil_loss_list<=np.percentile(sil_loss_list, 80)
    tmp_filter = np.logical_or(rgb_filter, sil_filter)

    if len(R_list)>2:
        if len(tmp_filter[tmp_filter]) >= 4:
            rotmat_candidates = cluster_rotations(torch.stack(R_list).detach().cpu().numpy()[tmp_filter])
        else:
            rotmat_candidates = cluster_rotations(torch.stack(R_list).detach().cpu().numpy())
    else:
        rotmat_candidates = R_list
        tmp_filter[:] = True
    
    tg_cams = list(np.array(top_n_cams)[tmp_filter])
    rescaled_scene.get_batched_renderer(tg_cams, renderer_type=renderer_type)
    # Parsing Learning Arguments
    learning_rate = learning_rate
    # sil_weight = 10
    iter_numb = 200
    early_stopping_max = 100
    early_stopping_unit = 0.0001 # (using silhouette)
    # Weights
    iou_weight = iou_weight
    tg_weight = tg_weight
    # additional flag
    average_iou_loss_filter = 0.7
    initial_useiou_flag = True

    recombined_mask = obj_mask_stackted_t[tmp_filter].transpose(1,0).reshape(rescaled_scene.height, len(tmp_filter[tmp_filter])*rescaled_scene.width).detach().cpu().numpy()[...,None]
    
    # argument for function
    learning_args = (learning_rate, iter_numb, early_stopping_max, early_stopping_unit, \
                        iou_weight, tg_weight, average_iou_loss_filter, initial_useiou_flag)
    model_inputs = rescaled_scene, obj_dict, len(tg_cams), \
                    obj_mask_stackted_t[tmp_filter], obj_masked_rgb_stacked_t[tmp_filter], recombined_mask, object_final_output_dir

    # with mp.Pool(processes=num_processes) as pool:
    #     results = pool.map(optimize_with_initial_sRt, args_tuple)
    for ridx, rotmat in enumerate(rotmat_candidates):

        optimize_with_initial_sRt((model_inputs, f'fibonacci_{ridx}_', scaled, default_scale, torch.tensor(rotmat, device=device).float(), \
                                    initial_translate, \
                                    learning_args, use_rgb, object_output_dir))
        
object6d_silhouette(
    scene_path,
    obj_name, 
    renderer_type = 'nvdiffrast',
    wo_simplify = False,
    rescale_factor = 0.25,
    confidence = 0.002,
    learning_rate = 1e-2,
    iou_weight = 0.5,
    tg_weight = 10.0,
    use_rgb = False,
    image_dir = os.path.join(home_path, image_path)
)

obj_mesh, obj_T, _ = get_obj_info(scene_path, obj_name, obj_status_path=None)
    
pc_info = get_pcinfo()
pc_list = list(pc_info.keys())

file_path = os.path.join(shared_dir, "Icra/rsc/smallbowl2/aruco_objctr.pickle")

# pickle 파일 로드 (with문 사용 - 권장)
with open(file_path, 'rb') as f:
    marker_offset = pickle.load(f)


mesh = trimesh.load(os.path.join(shared_dir, "Icra/rsc/smallbowl2/smallbowl2.obj"))
capture_image = False
if capture_image:
    run_script(f"python src/capture/camera/image_client.py", pc_list)
    camera = RemoteCameraController("image", None)

    index = int(find_latest_index(os.path.join(shared_dir, "object/marker_offset/smallbowl2")))+1
    os.makedirs(os.path.join(shared_dir, f"object/marker_offset/smallbowl2/{index}/image"), exist_ok=True)
    camera.start(f"shared_data/object/marker_offset/smallbowl2/{index}/image")
    camera.end()
    camera.quit()

else:
    index = int(find_latest_index(os.path.join(shared_dir, "object/marker_offset/smallbowl2")))

img_dir = os.path.join(shared_dir, f"object/marker_offset/smallbowl2/{index}/image")
img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(img_dir, img_name)) for img_name in os.listdir(img_dir)}
serial_list = list(img_dict.keys())
serial_list.sort()

intrinsic, extrinsic = load_current_camparam()
cammtx = get_cammtx(intrinsic, extrinsic)

c2r = load_latest_C2R()
r2c = np.linalg.inv(c2r)

obj_cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)
marker_index = [608, 609, 606]

A = []
for marker_id in marker_index:
    marker_pos = np.zeros((4, 3))
    for i in range(4):
        marker_pos[i] = marker_offset[marker_id][i]
    A.append(marker_pos)
        
A = np.concatenate(A)
B = np.concatenate([obj_cor_3d[marker_id] for marker_id in marker_index])
##3333333333333333  obj_T = rigid_transform_3D(A, B)

obj_T = get_current_object_6d("smallbowl", False, img_dict)
mesh.apply_transform(obj_T)

intrinsic_list = []
extrinsic_list = []
for serial_name in serial_list:
    extmat = extrinsic[serial_name]
    extrinsic_list.append(extmat)    
    intrinsic_list.append(intrinsic[serial_name]['intrinsics_undistort'])
    
renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')
frame, mask = project_mesh_nvdiff(mesh, renderer)
mask = mask.detach().cpu().numpy()[:,:,:,0].astype(np.bool_)

os.makedirs(os.path.join(shared_dir, f"object/marker_offset/smallbowl2/{index}/debug"), exist_ok=True)

for i, serial_num in enumerate(serial_list):
    img = undistort_img(img_dict[serial_num], intrinsic[serial_num])
    
    robot_2d = []
    
    for id, cor in obj_cor_3d.items():
        if cor is not None:
            robot_2d.append(project(cammtx[serial_num], cor))
    draw_aruco(img, robot_2d, list(obj_cor_3d.keys()))
    overlay_mask(img, mask[i], np.array((255,0, 0)), 0.7)
    
    cv2.imwrite(os.path.join(shared_dir, f"object/marker_offset/smallbowl2/{index}/debug/{serial_num}.png"), img)
    