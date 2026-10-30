"""Camera must publish RAW images; rectify exactly once here. No motor node."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    share = Path(get_package_share_directory('hw2_navigation'))
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=str(share/'config/experiment.yaml')),
        DeclareLaunchArgument('enable_motion', default_value='false'),
        DeclareLaunchArgument('csv_path', default_value='/tmp/hw2-live.csv'),
        Node(package='image_proc', executable='rectify_node', namespace='camera',
             remappings=[('image', 'image_raw'), ('image_rect', 'image_rect')]),
        Node(package='apriltag_ros', executable='apriltag_node', name='hw2_detector',
             parameters=[str(share/'config/detector.yaml')],
             remappings=[('image_rect', '/camera/image_rect'),
                         ('camera_info', '/camera/camera_info')]),
        Node(package='hw2_navigation', executable='navigator', parameters=[{
            'config': LaunchConfiguration('config'),
            'enable_motion': LaunchConfiguration('enable_motion'),
            'csv_path': LaunchConfiguration('csv_path')}], output='screen')])
