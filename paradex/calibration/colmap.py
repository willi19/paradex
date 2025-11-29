import pycolmap
import numpy as np
import os
import cv2
import sys
import sqlite3
import numpy as np

def get_two_view_geometries(cam1, cam2, pix1, pix2, indices, pair): # get tuple of cameras
    pycam1 = pycolmap.Camera(model="OPENCV", width=cam1["width"], height=cam1["height"], params=list(cam1["params"].reshape(-1)))
    pycam2 = pycolmap.Camera(model="OPENCV", width=cam2["width"], height=cam2["height"], params=list(cam2["params"].reshape(-1)))
    E = pycolmap.estimate_essential_matrix(pix1, pix2, pycam1, pycam2)
    
    F = pycolmap.estimate_fundamental_matrix(pix1, pix2)
    H = pycolmap.estimate_homography_matrix(pix1, pix2)
    if E is None or F is None or H is None:
        return None
    # database is shared resource here
    return pair[0], pair[1], indices, F['F'], E['E'], H['H'], 3 # ways to getting two-view-geometries


def load_colmap_camparam(path, orig_size=(2048, 1536)):
    reconstruction = pycolmap.Reconstruction(path)
    cameras, images = dict(), dict()
    for camera_id, camera in reconstruction.cameras.items():
        cameras[camera_id] = camera # distortion params
    for image_id, image in reconstruction.images.items():
        images[image_id] = image # distortion params    

    intrinsics = dict()
    extrinsics = dict()
    
    for imid in images:
        serialnum = images[imid].name.split("_")[0][:-4]
        camid  = images[imid].camera_id

        w, h = cameras[camid].width, cameras[camid].height
        fx, fy, cx, cy, k1, k2, p1, p2 = cameras[camid].params

        cammtx = np.array([[fx,0.,cx],[0.,fy, cy], [0.,0.,1.]])
        dist_params = np.array([k1,k2,p1,p2])
        new_cammtx, roi = cv2.getOptimalNewCameraMatrix(cammtx, dist_params, (w, h), 1, orig_size)
        
        intrinsics[serialnum] = dict()
        # Save into parameters
        intrinsics[serialnum]["original_intrinsics"] = cammtx # calibrated
        intrinsics[serialnum]["intrinsics_undistort"] = new_cammtx # adjusted as pinhole
        # intrinsics[serialnum]["Intrinsics_warped"] = list(new_cammtx.reshape(-1))
        # intrinsics[serialnum]["Intrinsics_warped"][2] -= x   # check this to get undistorted information
        # intrinsics[serialnum]["Intrinsics_warped"][5] -= y
        intrinsics[serialnum]["dist_height"] = h 
        intrinsics[serialnum]["dist_height"] = w
        intrinsics[serialnum]["height"] = orig_size[1]
        intrinsics[serialnum]["width"] = orig_size[0]
        
        intrinsics[serialnum]["dist_params"] = dist_params


        extrinsics[serialnum] = dict()
        
        cam_pose = images[imid].cam_from_world.matrix()
        
        extrinsics[serialnum] = cam_pose.tolist()
    
    return intrinsics, extrinsics


IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return int(image_id1), image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)
# 

def blob_to_array(blob, dtype, shape=(-1,)):
    if blob is not None:
        if IS_PYTHON3:
            return np.fromstring(blob, dtype=dtype).reshape(*shape)
        else:
            return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64) # list
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid 

    def add_image(self, name, camera_id,
                  prior_q=np.full(4, np.NaN), prior_t=np.full(3, np.NaN), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches, # list
                              F=np.eye(3), E=np.eye(3), H=np.eye(3),
                              config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H),
             ))

    def get_camera(self):
        cameras = dict(
            (camera_id, {'model':model, 'width':width, 'height':height, 
                            'params': blob_to_array(params, np.float64, (-1, 4))})
            for camera_id, model, width, height, params, prior_focal_length  in self.execute(
                "SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras"))
        return cameras
    
    def get_descriptors(self):
        descrpitors = dict(
            (image_id, blob_to_array(data, np.uint8, (-1, 128)))
            for image_id, data in self.execute(
                "SELECT image_id, data FROM descriptors"))
        return descrpitors
    
    def get_keypoints(self): # only for sift
        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, 6))[:, :2])
            for image_id, data in self.execute(
                "SELECT image_id, data FROM keypoints"))
        return keypoints

    def get_corners(self): # only for sift
        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, 2)))
            for image_id, data in self.execute(
                "SELECT image_id, data FROM keypoints"))
        return keypoints    

    def get_matches(self):
        matches = dict(
            (pair_id_to_image_ids(pair_id), # image_id1, image_id2
            blob_to_array(data, np.uint32, (-1, 2)))
            for pair_id, data in self.execute(
                "SELECT pair_id, data FROM matches WHERE data IS NOT NULL")
        )
        return matches

    def get_images(self):
        images = dict(
            (image_id, {"name":name, "camera_id":camera_id})
            for image_id, name, camera_id in self.execute(
                "SELECT image_id, name, camera_id FROM images WHERE image_id IS NOT NULL")
        )
        return images

    def get_two_view_geometries(self): 
        twoview = dict(
            (pair_id_to_image_ids(pair_id),
            {"pair_id":pair_id_to_image_ids(pair_id), "data":blob_to_array(data,np.uint32 ,(-1,2)), "config":config, "F":blob_to_array(F, np.float32,(-1,3,3)),
                "E":blob_to_array(E,np.float32, (-1, 3,3)),  "H":blob_to_array(F,np.float32, (-1, 3,3))} # image id
            )
            for pair_id, data, config, F,E, H, in self.execute(
                "SELECT pair_id, data, config, F, E, H FROM two_view_geometries WHERE data IS NOT NULL")
        )
        return twoview