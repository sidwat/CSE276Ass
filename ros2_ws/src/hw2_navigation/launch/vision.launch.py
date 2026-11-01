"""Use the existing AprilTag config unchanged, then consume its TF for navigation."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from hw2_navigation.configuration import default_detector_config


def generate_launch_description():
    share = Path(get_package_share_directory('hw2_navigation'))
    config = LaunchConfiguration('config')
    detector_config = LaunchConfiguration('detector_config')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=str(share/'config/experiment.yaml')),
        DeclareLaunchArgument('detector_config', default_value=default_detector_config()),
        DeclareLaunchArgument('enable_motion', default_value='false'),
        DeclareLaunchArgument('show_overlay', default_value='false'),
        DeclareLaunchArgument('csv_path', default_value='/tmp/hw2-live.csv'),
        Node(package='image_proc', executable='rectify_node', namespace='camera',
             remappings=[('image', 'image_raw'), ('image_rect', 'image_rect')]),
        Node(package='apriltag_ros', executable='apriltag_node', name='hw2_detector',
             parameters=[detector_config],
             remappings=[('image_rect', '/camera/image_rect'),
                         ('camera_info', '/camera/camera_info')]),
        Node(package='hw2_navigation', executable='overlay',
             condition=IfCondition(LaunchConfiguration('show_overlay'))),
        Node(package='hw2_navigation', executable='navigator', parameters=[{
            'config': config, 'detector_config': detector_config,
            'enable_motion': LaunchConfiguration('enable_motion'),
            'csv_path': LaunchConfiguration('csv_path')}], output='screen')])
