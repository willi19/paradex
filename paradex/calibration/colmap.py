import pycolmap

def get_two_view_geometries(cam1, cam2, pix1, pix2, indices, pair): # get tuple of cameras
    pycam1 = pycolmap.Camera(model="OPENCV", width=cam1["width"], height=cam1["height"], params=list(cam1["params"].reshape(-1)))
    pycam2 = pycolmap.Camera(model="OPENCV", width=cam2["width"], height=cam2["height"], params=list(cam2["params"].reshape(-1)))
    E = pycolmap.estimate_essential_matrix(pix1, pix2, pycam1, pycam2)
    F = pycolmap.estimate_fundamental_matrix(pix1, pix2)
    F = pycolmap.estimate_fundamental_matrix(pix1, pix2)['F']
    H = pycolmap.estimate_homography_matrix(pix1, pix2)['H']
    # database is shared resource here
    return pair[0], pair[1], indices, F, E['E'], H, 3 # ways to getting two-view-geometries
