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
        Node(package="tf2_ros", executable="static_transform_publisher", arguments=["0", "0", "0", "0", "0", "0", "base_link", "zed_camera"]),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '/bags/rosbag2_2024_06_12-13_18_05'],
            output='screen'
        )
    ])