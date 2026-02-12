import numpy as np 

contact_tg = ['thumb_tip', 'thumb_distal', 'thumb_medial', 'palm_link', \
                    'index_tip','index_medial','index_proximal',
                    'middle_tip','middle_medial','middle_proximal',
                    'ring_tip','ring_medial','ring_proximal']

sensororder = ["thumb_tip_0","thumb_distal_0","thumb_medial_0",\
            "index_tip_0","index_medial_0","index_proximal_0",\
            "middle_tip_0","middle_medial_0","middle_proximal_0",\
            "ring_tip_0","ring_medial_0","ring_proximal_0",\
            "palm_link_0","palm_link_1","palm_link_2"]

idx2sensor = {i:sensororder[i] for i in range(len(sensororder))}
sensor2idx = {sensororder[i]:i for i in range(len(sensororder))}

def get_contact_ctr(link_nm, vertices, faces, vertex_normals, triangle_normals):
    '''
        return contact center point and normal vector
    '''
    if link_nm == 'palm_link':
        center_point_list, mean_normal_list = [], []
        vertex_indices_list = [np.array([5035,5039]),np.array([5043,5044]),np.array([5049,5066])]
        faces_indices_list = [np.array([3047]),np.array([3051]),np.array([3075])]
        for vertex_indices, faces_indices in zip(vertex_indices_list, faces_indices_list):
            center_point = np.mean(vertices[vertex_indices], axis=0)
            mean_normal = np.mean(triangle_normals[faces_indices], axis=0)
            mean_normal/=np.linalg.norm(mean_normal)
            center_point_list.append(center_point) 
            mean_normal_list.append(mean_normal)
        return center_point_list, mean_normal_list
    elif link_nm in ['thumb_tip','index_tip','middle_tip','ring_tip']:
        tg_face_idx = 12320
        vertex_indices = faces[tg_face_idx]
        center_point = np.mean(vertices[vertex_indices], axis=0)
        mean_normal = np.mean(vertex_normals[vertex_indices], axis=0)
        mean_normal/=np.linalg.norm(mean_normal)
        return [center_point], [mean_normal]
    elif link_nm == 'thumb_distal':    
        vertex_indices = np.array([95,96,97,86])
        center_point = np.mean(vertices[vertex_indices], axis=0)
        mean_normal = np.mean(vertex_normals[vertex_indices], axis=0)
        mean_normal/=np.linalg.norm(mean_normal)
        return [center_point], [mean_normal]
    elif link_nm == 'thumb_medial':    
        vertex_indices = np.array([13,10])
        center_point = np.mean(vertices[vertex_indices], axis=0)
        mean_normal = np.mean(vertex_normals[vertex_indices], axis=0)
        mean_normal/=np.linalg.norm(mean_normal)
        return [center_point], [mean_normal]
    elif link_nm == 'index_medial':    
        vertex_indices = np.array([1954,1947])
        center_point = np.mean(vertices[vertex_indices], axis=0)
        mean_normal = np.mean(vertex_normals[vertex_indices], axis=0)
        mean_normal/=np.linalg.norm(mean_normal)
        return [center_point], [mean_normal]
    elif link_nm == 'index_proximal':    
        vertex_indices = np.array([3,5])
        center_point = np.mean(vertices[vertex_indices], axis=0)
        mean_normal = np.mean(vertex_normals[vertex_indices], axis=0)
        mean_normal/=np.linalg.norm(mean_normal)
        return [center_point], [mean_normal]
    elif link_nm == 'middle_medial':    
        vertex_indices = np.array([1954,1947])
        center_point = np.mean(vertices[vertex_indices], axis=0)
        mean_normal = np.mean(vertex_normals[vertex_indices], axis=0)
        mean_normal/=np.linalg.norm(mean_normal)
        return [center_point], [mean_normal]
    elif link_nm == 'middle_proximal':    
        vertex_indices = np.array([3,5])
        center_point = np.mean(vertices[vertex_indices], axis=0)
        mean_normal = np.mean(vertex_normals[vertex_indices], axis=0)
        mean_normal/=np.linalg.norm(mean_normal)
        return [center_point], [mean_normal]
    elif link_nm == 'ring_medial':    
        vertex_indices = np.array([1954,1947])
        center_point = np.mean(vertices[vertex_indices], axis=0)
        mean_normal = np.mean(vertex_normals[vertex_indices], axis=0)
        mean_normal/=np.linalg.norm(mean_normal)
        return [center_point], [mean_normal]
    elif link_nm == 'ring_proximal':    
        vertex_indices = np.array([3,5])
        center_point = np.mean(vertices[vertex_indices], axis=0)
        mean_normal = np.mean(vertex_normals[vertex_indices], axis=0)
        mean_normal/=np.linalg.norm(mean_normal)
        return [center_point], [mean_normal]
    else:
        print("Not Implemented Yet!")
        return [],[]


import copy

def value_to_color(value):
    '''
    value -1...12
    '''
    # 각 구간에 맞는 색상 계산
    color = np.zeros((value.shape[0], 3))
    
    # 회색 (-1)
    color[value == -1] = [0.5, 0.5, 0.5]
    
    # 붉은색 (0, 1, 2)
    color[(value >= 0) & (value <= 2)] = np.stack([np.ones(np.sum((value >= 0) & (value <= 2))), 
                                                   np.zeros(np.sum((value >= 0) & (value <= 2))),
                                                   (value[(value >= 0) & (value <= 2)] * 0.5)], axis=-1)
    
    # 노란색 (3)
    color[value == 3] = [1, 1, 0]
    
    # 초록색 (4-6)
    green_condition = (value >= 4) & (value <= 6)
    color[green_condition] = np.stack([(6 - value[green_condition]) * 0.167,
                                        (value[green_condition] - 3) * 0.167,
                                        np.zeros(np.sum(green_condition))], axis=-1)
    
    # 파랑색 (7-9)
    blue_condition = (value >= 7) & (value <= 9)
    color[blue_condition] = np.stack([np.zeros(np.sum(blue_condition)),
                                      (value[blue_condition] - 6) * 0.167,
                                      1 - (value[blue_condition] - 6) * 0.167], axis=-1)
    
    # 보라색 (10-12)
    purple_condition = (value >= 10) & (value <= 12)
    color[purple_condition] = np.stack([(value[purple_condition] - 9) * 0.167,
                                        np.zeros(np.sum(purple_condition)),
                                        1 - (value[purple_condition] - 9) * 0.167], axis=-1)
    
    return color


import open3d as o3d
def get_categorized_mesh(o3d_mesh, vertex_values, save_path=None):
    '''
        vertex_values -1...12
    '''
    colorized_tg_link = value_to_color(vertex_values)
    colorized_tg_link_uint8 = (colorized_tg_link * 255).astype(np.uint8)
    colored_mesh = copy.deepcopy(o3d_mesh)

    if isinstance(o3d_mesh, o3d.geometry.TriangleMesh):
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colorized_tg_link)
    else: # for trimesh
        colored_mesh.visual.vertex_colors = colorized_tg_link_uint8

    if save_path is not None:
        colored_mesh.export(save_path)
    return colored_mesh