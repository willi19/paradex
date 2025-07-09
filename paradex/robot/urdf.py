from urdfpy import URDF

def get_end_links(urdf_path):
    robot = URDF.load(urdf_path)

    # All link names
    link_names = {link.name for link in robot.links}
    child_links = {joint.child for joint in robot.joints}
    parent_links = {joint.parent for joint in robot.joints}

    # End link: link that is never a parent
    end_links = list(link_names - parent_links)
    return end_links

def get_root_links(urdf_path):
    robot = URDF.load(urdf_path)

    # All link names
    link_names = {link.name for link in robot.links}
    child_links = {joint.child for joint in robot.joints}
    parent_links = {joint.parent for joint in robot.joints}

    # Root: link that is never a child
    root_links = list(link_names - child_links)
    if len(root_links) != 1:
        raise RuntimeError(f"Ambiguous or missing root link: {root_links}")
    return root_links