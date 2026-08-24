import numpy as np
import torch

from paradex.image import projection


def test_render_mask_avoids_rgb_and_depth_work(monkeypatch):
    renderer = projection.BatchRenderer.__new__(projection.BatchRenderer)
    renderer.device = "cpu"
    renderer.serial_list = ["cam-a", "cam-b"]
    renderer.height = 2
    renderer.width = 3
    renderer.glctx = object()
    renderer.intr_opengl = torch.eye(4).repeat(2, 1, 1)
    renderer.flip_z = torch.eye(4)
    renderer.cam_extrs_t = torch.eye(4).repeat(2, 1, 1)

    monkeypatch.setattr(
        projection,
        "mesh_to_obj_dict",
        lambda *_args, **_kwargs: {
            "verts": torch.zeros((1, 3, 3)),
            "faces": torch.tensor([[0, 1, 2]], dtype=torch.int32),
        },
    )
    monkeypatch.setattr(
        projection,
        "transform_pos",
        lambda _matrix, _positions: torch.zeros((2, 3, 4)),
    )
    raster = torch.zeros((2, 2, 3, 4))
    monkeypatch.setattr(projection.dr, "rasterize", lambda *_args, **_kwargs: (raster, None))
    monkeypatch.setattr(
        projection.dr,
        "interpolate",
        lambda *_args, **_kwargs: (torch.ones((2, 2, 3, 1)), None),
    )
    monkeypatch.setattr(
        projection.dr,
        "antialias",
        lambda mask, *_args, **_kwargs: mask,
    )
    monkeypatch.setattr(
        renderer,
        "render_wvertexcolor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("RGB rendering must not run")
        ),
    )

    masks = renderer.render_mask(object())

    assert set(masks) == {"cam-a", "cam-b"}
    assert all(mask.dtype == np.bool_ for mask in masks.values())
    assert all(mask.all() for mask in masks.values())
