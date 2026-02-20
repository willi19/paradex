import copy
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom


def _prettify_xml(elem):
    rough = ET.tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")


def _merge_urdf_without_xacro(output_path, args_dict):
    required = ["arm_file", "hand_file", "parent_link", "child_link", "xyz", "rpy"]
    missing = [k for k in required if k not in args_dict]
    if missing:
        raise ValueError(f"Missing required args for fallback merge: {missing}")

    arm_tree = ET.parse(args_dict["arm_file"])
    hand_tree = ET.parse(args_dict["hand_file"])
    arm_root = arm_tree.getroot()
    hand_root = hand_tree.getroot()

    merged_root = ET.Element("robot", {"name": "combined_robot"})

    for child in list(arm_root):
        merged_root.append(copy.deepcopy(child))
    for child in list(hand_root):
        merged_root.append(copy.deepcopy(child))

    joint = ET.SubElement(merged_root, "joint", {"name": "arm_to_hand", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": str(args_dict["parent_link"])})
    ET.SubElement(joint, "child", {"link": str(args_dict["child_link"])})
    ET.SubElement(
        joint,
        "origin",
        {"xyz": str(args_dict["xyz"]), "rpy": str(args_dict["rpy"])},
    )

    xml_bytes = _prettify_xml(merged_root)
    with open(output_path, "wb") as f:
        f.write(xml_bytes)

def generate_urdf(xacro_path, output_path, args_dict):
    # Prepare command
    cmd = ["xacro", str(xacro_path)]

    # Add arguments
    for key, value in args_dict.items():
        cmd.append(f"{key}:={value}")

    # Write output to file
    try:
        with open(output_path, "w") as f:
            subprocess.run(cmd, stdout=f, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        # Some environments have xacro without ROS substitution support.
        # Fall back to direct XML merge to keep merge_urdf.py usable.
        print(f"xacro failed, falling back to direct URDF merge: {exc}")
        _merge_urdf_without_xacro(output_path, args_dict)

    print(f"Generated URDF saved to: {output_path}")
