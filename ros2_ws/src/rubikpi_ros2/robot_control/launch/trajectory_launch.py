from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml

# Optional overrides of params_file values, e.g. wheel_base:=0.70; empty means "use the YAML"
OVERRIDES = [
    'speed_per_unit', 'wheel_trim', 'start_delay',
    # turn-drive-turn mode
    'drive_cmd', 'spin_cmd', 'reverse_speed_per_unit', 'reverse_trim', 'drive_offset',
    'ccw_deg_per_s', 'ccw_offset_deg', 'cw_deg_per_s', 'cw_offset_deg', 'pause',
    # quintic mode
    'wheel_base', 'cruise_speed', 'ramp_time', 'max_curvature', 'corridor_width',
]
BOOL_OVERRIDES = ['quintic', 'allow_reverse']
# What run_recorder needs to dead-reckon the commands it sees
RECORDER_PARAMS = ['speed_per_unit', 'wheel_base', 'wheel_trim', 'reverse_speed_per_unit',
                   'reverse_trim', 'ccw_deg_per_s', 'ccw_offset_deg', 'cw_deg_per_s',
                   'cw_offset_deg']


def launch_nodes(context):
    overrides = {}
    for name in OVERRIDES:
        value = LaunchConfiguration(name).perform(context)
        if value:
            overrides[name] = float(value)  # ROS rejects an int for a float parameter
    for name in BOOL_OVERRIDES:
        value = LaunchConfiguration(name).perform(context).lower()
        if value:
            overrides[name] = value in ('true', '1', 'yes')

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
            overrides,
        ],
    )

    nodes = []
    if LaunchConfiguration('record').perform(context).lower() in ('true', '1', 'yes'):
        with open(LaunchConfiguration('params_file').perform(context)) as f:
            yaml_params = yaml.safe_load(f)['trajectory_control_node']['ros__parameters']
        merged = {**yaml_params, **overrides}
        calib = {k: float(merged[k]) for k in RECORDER_PARAMS if k in merged}
        calib['quintic'] = bool(merged.get('quintic', False))
        nodes.append(Node(
            package='robot_control',
            executable='run_recorder',
            name='run_recorder',
            output='screen',
            emulate_tty=True,
            sigterm_timeout='20',
            parameters=[calib, {
                'waypoints_file': LaunchConfiguration('waypoints_file'),
                'output_dir': LaunchConfiguration('output_dir'),
                'tag': 'trajectory',
            }],
        ))

    return nodes + [
        motor_node,
        trajectory_node,
        RegisterEventHandler(OnProcessExit(
            target_action=trajectory_node,
            on_exit=[Shutdown(reason='Trajectory follower exited')])),
        RegisterEventHandler(OnProcessExit(
            target_action=motor_node,
            on_exit=[Shutdown(reason='Motor controller exited')])),
    ]


def generate_launch_description():
    config_dir = PathJoinSubstitution([FindPackageShare('robot_control'), 'config'])

    return LaunchDescription([
        DeclareLaunchArgument(
            'waypoints_file',
            default_value=PathJoinSubstitution([config_dir, 'waypoints.txt']),
            description='File of "x,y,theta" waypoints (metres, radians), relative to the start pose',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([config_dir, 'trajectory_control.yaml']),
            description='Calibration, motion-profile and constraint parameters',
        ),
        *[DeclareLaunchArgument(name, default_value='', description=f'override {name}')
          for name in OVERRIDES + BOOL_OVERRIDES],
        DeclareLaunchArgument('record', default_value='true',
                              description='record motor_commands/imu_data and plot on exit'),
        DeclareLaunchArgument('output_dir', default_value='~/robot_runs',
                              description='where run_recorder writes CSVs and plots'),
        OpaqueFunction(function=launch_nodes),
    ])
