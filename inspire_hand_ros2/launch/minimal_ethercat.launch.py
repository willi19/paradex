from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="inspire_hand_ros2",
            executable="inspire_ethercat_node",
            name="inspire_hand",
            output="screen",
            parameters=[
                {"ifname": "enp7s0f3"},
                {"hand_type": "rh56f1"},
                {"enable_tactile": True},
            ],
        )
    ])
