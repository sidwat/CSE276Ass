from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_dir = PathJoinSubstitution([FindPackageShare('robot_control'), 'config'])

    waypoints_file = DeclareLaunchArgument(
        'waypoints_file',
        default_value=PathJoinSubstitution([config_dir, 'waypoints.txt']),
        description='File of "x,y,theta" waypoints (metres, radians), relative to the start pose',
    )
    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([config_dir, 'trajectory_control.yaml']),
        description='Calibration, motion-profile and constraint parameters',
    )

    motor_node = Node(
        package='robot_control',
        executable='motor_control',
        name='motor_controller_node',
        output='screen',
        emulate_tty=True,
    )

    trajectory_node = Node(
        package='robot_control',
        executable='trajectory_control',
        name='trajectory_control_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {'waypoints_file': LaunchConfiguration('waypoints_file')},
        ],
    )

    return LaunchDescription([
        waypoints_file,
        params_file,
        motor_node,
        trajectory_node,
        RegisterEventHandler(OnProcessExit(
            target_action=trajectory_node,
            on_exit=[Shutdown(reason='Trajectory follower exited')])),
        RegisterEventHandler(OnProcessExit(
            target_action=motor_node,
            on_exit=[Shutdown(reason='Motor controller exited')])),
    ])
