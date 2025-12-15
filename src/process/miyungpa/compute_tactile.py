@compute_contact_button.on_click
def _(_) -> None:
    # Get NN at point
    current_timestep = gui_timestep.value
    link_T =  robot_module.get_T_dict(current_timestep)
    
    cur_frame_node_name = f"/frames/t{current_timestep}"
    obj_mesh_nm = cur_frame_node_name+'_obj' if cur_frame_node_name+'_obj' in mesh_dictionary else object_nm
    cur_object_mesh = mesh_dictionary[obj_mesh_nm] 
    obj_vertices = cur_object_mesh.vertices
    
    combined_robot_vertices = original_vertices[cur_frame_node_name]

    points = []
    colors = []

    object_min_dist = np.ones(obj_vertices.shape[0])
    object_min_dist_tg = np.zeros(obj_vertices.shape[0])
    object_min_dist_tg.fill(-1) # Set to -1 (not contacted)
    
    #  [mesh_dictionary[obj_frame.name]]+[mesh_dictionary[frame_node.name] for frame_node in frame_nodes if frame_node.visible]object_mesh_vertices = 
    obj_tree = cKDTree(obj_vertices)
    for tidx, hand_tg in enumerate(contact_tg):
        vertex_range = robot_link2vertex_mapping[hand_tg]
        robot_part_vertices = combined_robot_vertices[vertex_range[0]:vertex_range[1]]
        distances, indices = obj_tree.query(robot_part_vertices, k=1)
        robot_index = np.argmin(distances)
        obj_index = indices[robot_index]
        robot_point = robot_part_vertices[robot_index]
        obj_point = obj_vertices[obj_index]
        points.append([robot_point, obj_point])
        colors.append([[0,0,255],[0,0,255]])

        hand_tree = cKDTree(robot_part_vertices)
        distance_obj2hand = hand_tree.query(obj_vertices, k=1)[0]
        update_filter = distance_obj2hand<object_min_dist
        object_min_dist[update_filter] = distance_obj2hand[update_filter]
        object_min_dist_tg[update_filter] = tidx

    points = np.stack(points)
    colors = np.stack(colors)

    line = server.scene.add_line_segments(f"/line_segments/contact_line{len(line_segments)}", points=points, colors=colors, line_width=3.0)
    contact_line_segments.append(line)

    # add point
    for point in np.vstack(points):
        point = server.scene.add_point_cloud(
            f"/point_segments/point{len(contact_point_segments)}",
            points=point[np.newaxis,...],
            colors=(
                np.array([[255,0,0]])
            ).astype(np.uint8),
            point_size=0.001,
            point_shape='circle'
        )
        contact_point_segments.append(point)

    # Transfer Contact to Object Mesh
    threshold_filter = object_min_dist>contact_thres.value
    object_min_dist_tg[threshold_filter] = -1
    object_min_dist_sensor = np.zeros_like(object_min_dist)

    current_contact = robot_module.contact[current_timestep]

    for sidx, sensor_nm in enumerate(sensororder):
        link_nm = sensor_nm[:-2]
        contact_value = current_contact[sidx]
        tg_filter = np.logical_and(object_min_dist_tg==contact_tg.index(link_nm),object_min_dist_sensor<contact_value)
        object_min_dist_sensor[tg_filter] = contact_value

    save_directory = scene_path/'contact'
    os.makedirs(save_directory, exist_ok=True)

    _ = get_colored_mesh(mesh_dictionary[obj_mesh_nm], object_min_dist, save_path=str(save_directory/'debug_distance.obj'), cmap_nm='viridis_r')
    _ = get_categorized_mesh(mesh_dictionary[obj_mesh_nm], object_min_dist_tg, save_path=str(save_directory/'debug_part.obj'))
    _ = get_colored_mesh(mesh_dictionary[obj_mesh_nm], object_min_dist_sensor, save_path=str(save_directory/'debug_sensor.obj'), cmap_nm='viridis')

    pickle.dump(object_min_dist, open(save_directory/'object_min_dist.pickle','wb'))
    pickle.dump(object_min_dist_tg, open(save_directory/'object_min_dist_tg.pickle','wb'))
    pickle.dump(object_min_dist_sensor, open(save_directory/'object_min_dist_sensor.pickle','wb'))

    robot_state = capture_scene.robot_traj[current_timestep] # 6 for arm, and 16 for hand
    pickle.dump({'current_timestep':current_timestep,'robot_traj':robot_state, 'link_T':link_T}, \
                open(save_directory/'robot_pose_in_contact.pickle','wb'))
    print("Transfer ended")