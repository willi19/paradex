def get_cammtx(intrinsic, extrinsic):
    cammat = {}
    for serial_num in list(intrinsic.keys()):
        int_mat = intrinsic[serial_num]["intrinsics_undistort"]
        ext_mat = extrinsic[serial_num]
        cammat[serial_num] = int_mat @ ext_mat
    return cammat