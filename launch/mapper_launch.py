from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mapper_speedrun',
            executable='mapper',
            name='mapper'
        ),
        # Node(package="tf2_ros", executable="static_transform_publisher", arguments=["0", "0", "0", "0", "0", "0", "base_footprint", "zed_camera_center"])
    ])