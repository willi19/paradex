import sys, os
from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_PATH))
import numpy as np
import pickle
import cv2
from copy import deepcopy

import torch
from paradex.object_detection.obj_utils.scene import Scene
from paradex.object_detection.obj_utils.io import get_optimal_T, get_tracking_result
from paradex.object_detection.obj_utils.vis_utils import (
    parse_objectmesh_objdict,
    make_grid_image_np,
    overlay_mask,
    make_square_img,
    putText
)
from paradex.object_detection.default_config import (
    default_template_combined,
    template2camids_combined
)


# from utils.geometry import project_3d_to_2d_tensor, project_3d_to_2d
# Initialize the matcher with default settings

"""
    TODO:
    1. select view > Done > Reduce number
    2. diversify (by rotate or flip) > Done
    3. pre-cropped 
"""

device = "cuda"


class Template_Varied_Combined:
    """
    Template is a set of images and matched 6D pose of object
    """

    def __init__(
        self,
        root_dir: str,
        obj_name=None,
        render_template_path="./check_template.jpeg",
        render_template=False,
    ):
        assert root_dir is not None and os.path.exists(
            root_dir
        ), f"Should check directory: {root_dir}"

        if obj_name is None:
            obj_name = root_dir.split("/")[-2]

        template_path = str(Path(root_dir) / "template_varied_rots_combined.pkl")

        # Load Object Mesh
        obj_dict = parse_objectmesh_objdict(
            obj_name,
            min_vertex_num=1000,
            remove_uv=True,
            renderer_type="nvdiffrast",
            device=device,
        )
        if not obj_dict["canonicalized"]:
            obj_scale = obj_optim_output["scale"].detach().cpu().numpy().item(0)
            obj_dict["verts"] *= obj_scale
        else:
            obj_scale = 1.0
        self.obj_dict = obj_dict

        if os.path.exists(template_path):
            tmp_dict = pickle.load(open(template_path, "rb"))
            self.img2face = tmp_dict["img2face"]
            self.img2point3d = tmp_dict["img2point3d"]
            obj_optim_output = tmp_dict["obj_optim_output"]
            self.img_template = tmp_dict["img_template"]
            self.mask_template = tmp_dict["mask_template"]

        else:

            self.img2face = {}
            self.img2point3d = {}
            self.mask_template = {}
            self.img_template = {}

            for sub_dir in sorted(os.listdir(root_dir)):
                if sub_dir=='0':
                    main_dir = True
                else:
                    main_dir = False

                sub_dir_path = Path(root_dir)/sub_dir

                self.scene = Scene(
                    scene_path=sub_dir_path,
                    rescale_factor=0.5,
                    mask_dir_nm=f"mask_hq/{obj_name}",
                )
                if main_dir:
                    obj_initial_status = get_optimal_T(
                        str(sub_dir_path)
                    )  # find optimal T in first frame.
                    self.obj_initial_T = None
                    if obj_initial_status is not None:
                        obj_optim_output = pickle.load(open(obj_initial_status, "rb"))
                        for key in obj_optim_output:
                            if torch.is_tensor(obj_optim_output[key]):
                                obj_optim_output[key] = obj_optim_output[key].to(device)

                    obj_T = np.eye(4)
                    obj_T[:3, :3] = obj_optim_output["R"].detach().cpu().numpy()
                    obj_T[:3, 3] = obj_optim_output["t"].detach().cpu().numpy()
                    self.obj_initial_T = obj_T
                    print(f"Object R:{obj_T[:3,:3]}, obj_t {obj_T[:3,3]}")
                else:
                    obj_tracking_output = get_tracking_result(sub_dir_path)
                    self.obj_initial_T = obj_tracking_output[obj_name][0]
                    obj_optim_output = {'R':torch.tensor(self.obj_initial_T[:3,:3], device=device).float(),
                                        't':torch.tensor(self.obj_initial_T[:3,3], device=device).float().unsqueeze(0),
                                        'scale':torch.tensor(1.0, device=device).float().unsqueeze(0)}
                    print(f"Object R:{self.obj_initial_T [:3,:3]}, obj_t {self.obj_initial_T [:3,3]}")

                if obj_name in template2camids_combined and sub_dir in template2camids_combined[obj_name]:
                    self.tg_cams = template2camids_combined[obj_name][sub_dir]
                else:
                    self.tg_cams = []
                render_template_path =  f"./template_rendered/{obj_name}_template_render_{sub_dir}.png"
                if render_template:
                    self.render_template(
                        self.obj_dict, obj_optim_output, render_template_path, render_all=True
                    )

                self.init_obj_template(sub_dir, deepcopy(obj_dict), obj_optim_output, template_path)
                debug = True
                if debug:
                    import matplotlib.cm as cm
                    from matplotlib.colors import Normalize

                    cmap = cm.get_cmap("viridis")
                    imgs = []
                    for cam_id in self.img2face:
                        canvas = np.zeros(
                            (
                                self.img2face[cam_id].shape[0],
                                self.img2face[cam_id].shape[1],
                                3,
                            )
                        )
                        for r in range(self.img2face[cam_id].shape[0]):
                            for c in range(self.img2face[cam_id].shape[1]):
                                if self.img2face[cam_id][r][c] >= 0:
                                    face_float = self.img2face[cam_id][r][c] / len(
                                        self.obj_dict["faces"]
                                    )
                                    canvas[r][c] = cmap(face_float)[:3]
                        imgs.append(make_square_img(canvas * 255))
                    cv2.imwrite(
                        self.scene.scene_path / "template_faces.png",
                        make_grid_image_np(np.stack(imgs), int(len(imgs) / 4 ), 4),
                    )
                
                    # obj_optim_output, img_template, mask_template
            pickle.dump(
                {
                    "img2face": self.img2face,
                    "img2point3d": self.img2point3d,
                    "obj_optim_output": obj_optim_output,
                    "img_template": self.img_template,
                    "mask_template": self.mask_template,
                },
                open(template_path, "wb"),
            )


        for serial_num in self.img_template:
            self.dsize = self.img_template[serial_num].shape[:2][::-1]
            break

    def render_template(self, obj_dict, obj_optim_output, render_template_path, render_all=False):
        if render_all:
            tg_cams = self.scene.cam_ids
        else:
            tg_cams = self.tg_cams
        self.scene.get_batched_renderer(tg_cams)
        # Transform object
        transformed_obj = deepcopy(obj_dict)
        org_scaled_verts = transformed_obj["verts"].detach()
        transformed_verts = (
            torch.einsum("mn, bjn -> bjm", obj_optim_output["R"], (org_scaled_verts))
            + obj_optim_output["t"]
        )
        # transformed_obj = deepcopy(obj_dict)
        transformed_obj["verts"] = transformed_verts

        batch_rendered = self.scene.batch_render(transformed_obj)

        rendered_rgb, rendered_sil = batch_rendered
        rendered_sil = rendered_sil.squeeze()

        imgs = []
        # visualize on image
        for cidx, cam_id in enumerate(tg_cams):
            bgr_img = self.scene.get_image(cam_id, fidx=0)
            mask = rendered_sil[cidx].detach().cpu().numpy()
            overlaid = overlay_mask(bgr_img, mask=(mask > 0))
            overlaid = putText(overlaid, cam_id)
            if render_all:
                if cam_id in self.tg_cams:
                    overlaid = cv2.rectangle(overlaid, (0, 0), (overlaid.shape[1]-1, overlaid.shape[0]-1), (0,255,0), 10)
            imgs.append(overlaid)
            

        cv2.imwrite(
            render_template_path,
            make_grid_image_np(np.stack(imgs), 4, 6),
        )

    def init_obj_template(self, sub_dir, obj_dict, obj_optim_output, template_path):
        self.scene.get_batched_renderer(self.tg_cams)
        # Transform object
        transformed_obj = deepcopy(obj_dict)
        org_scaled_verts = transformed_obj["verts"].detach()
        transformed_verts = (
            torch.einsum("mn, bjn -> bjm", obj_optim_output["R"], (org_scaled_verts))
            + obj_optim_output["t"]
        )
        # transformed_obj = deepcopy(obj_dict)
        transformed_obj["verts"] = transformed_verts

        pix2face = (
            self.scene.batch_rasterize(transformed_obj).cpu().numpy()
        )  # CAM_NUMB, HEIGHT, WIDTH, 1 (-1 if not rendered, number of face index)


        for cidx, cam_id in enumerate(self.tg_cams):
            # TODO in here, diversify
            org_img = self.scene.get_image(cam_id, 0)
            for ridx, tg_rotation in enumerate([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]):
                if tg_rotation is not None:
                    tg_pix2face = cv2.rotate(pix2face[cidx], tg_rotation)
                    tg_img = cv2.rotate(org_img, tg_rotation)
                else:
                    tg_pix2face = pix2face[cidx]
                    tg_img = org_img
                view_id = f'{sub_dir}_{cam_id}_{ridx}'
                self.img_template[view_id] = tg_img

                self.img2face[view_id] = tg_pix2face
                self.img2point3d[view_id] = np.zeros(
                    (self.img2face[view_id].shape[0], self.img2face[view_id].shape[1], 3)
                )
                # array uint8
                tmp_mask = (self.img2face[view_id] > 0).astype(np.uint8) * 255
                self.mask_template[view_id] = np.tile(tmp_mask[..., np.newaxis], (1, 1, 3))
                ys, xs = np.where(self.img2face[view_id] != -1)
                for x, y in zip(xs, ys):
                    vertice_numbs = list(
                        obj_dict["faces"][int(self.img2face[view_id][y, x])]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    pos3d = torch.mean(obj_dict["verts"][0, vertice_numbs], axis=0)
                    # for vidx in vertice_numbs:
                    self.img2point3d[view_id][y, x] = pos3d.detach().cpu().numpy()

        # cv2.imwrite('test.png', make_grid_image_np(np.array([self.mask_template[cam_id] for cam_id in self.mask_template]), 4,6))


if __name__ == "__main__":
    colors = ['yellow']# ,'red','yellow'
    for color in colors:
        obj_name = f"{color}_ramen_von"
        tmp_template = Template_Varied_Combined(
            str(default_template_combined[obj_name]),
            obj_name=obj_name,
            render_template=True,
            render_template_path=f"./check_template_{obj_name}.jpeg",
        )
