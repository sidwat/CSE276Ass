"""
HW4 path-planning run: plan from the world file, then camera + apriltag + motors + follower.

  ros2 launch robot_control hw4_launch.py mode:=fast
  ros2 launch robot_control hw4_launch.py mode:=safe record_bag:=true run_tag:=try2
  ros2 launch robot_control hw4_launch.py mode:=fast world_file:=/path/to/my_world.yaml

At launch the planner (robot_control/hw4_tool.py) reads world_file (default:
config/hw4_world.yaml), plans the requested path and writes it into a new run folder
<workspace>/hw4_logs/<mode>_<date>_<time>[_tag]/ (waypoints, plan JSON, plan PNG). The follower
starts from the world's start pose, localises against the world's landmarks, logs into the same
folder and, when it exits, `hw4 eval` runs on the folder (auto_eval:=false to skip).
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction,
    RegisterEventHandler, Shutdown)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

BAG_TOPICS = ['/tf', '/detections', '/motor_commands', '/imu_data', '/camera/camera_info',
              '/rosout']


def launch_setup(context):
    def arg(name):
        return context.launch_configurations[name]

    from robot_control import hw4_tool
    from robot_control.world_model import load_world

    mode = arg('mode')
    world_file = os.path.realpath(os.path.expanduser(arg('world_file')
                                                     or hw4_tool.default_world_file()))
    params_file = arg('params_file') or hw4_tool.default_params_file()
    log_dir = os.path.expanduser(arg('log_dir')) if arg('log_dir') else hw4_tool.logs_dir()
    name = hw4_tool.run_name(mode, arg('run_tag'))
    run_dir = os.path.join(log_dir, name)

    # Plan now, so every run folder holds exactly the plan that was driven
    world = load_world(world_file)
    plan = hw4_tool.make_plan(world, mode)
    wp_file, m, png = hw4_tool.write_plan_files(world, plan, run_dir,
                                                plot=arg('plot').lower() == 'true')
    print(f'[hw4] {mode} plan: {len(plan.path) - 1} waypoints, {m["length"]:.2f} m, min body '
          f'gap {100 * m["min_body_gap"]:.1f} cm, estimated {m["time"]:.1f} s\n'
          f'[hw4] world: {world_file}\n[hw4] run folder: {run_dir}', flush=True)

    overrides = {
        'waypoints_file': wp_file,
        'world_file': world_file,
        'log_dir': log_dir,
        'run_name': name,
        'path_name': mode,
        'post_run_eval': arg('auto_eval').lower() == 'true',
    }
    if arg('use_imu'):
        overrides['use_imu'] = arg('use_imu').lower() == 'true'
    if arg('initial_pose'):
        overrides['initial_pose'] = arg('initial_pose')

    velocity_node = Node(
        package='robot_control',
        executable='velocity_control',
        name='velocity_control_node',
        output='screen',
        emulate_tty=True,
        parameters=[params_file, overrides],
    )
    actions = [
        velocity_node,
        RegisterEventHandler(OnProcessExit(
            target_action=velocity_node,
            on_exit=[Shutdown(reason='Waypoint follower exited')])),
    ]
    if arg('record_bag').lower() == 'true':
        topics = list(BAG_TOPICS)
        if arg('record_images').lower() == 'true':
            topics.append('/camera/image_raw/compressed')
        actions.append(ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', os.path.join(run_dir, 'bag')] + topics,
            output='screen'))
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument('mode', default_value='fast',
                              description='fast (visibility graph + A*), safe (Voronoi) or '
                                          'weighted (clearance-weighted grid search)'),
        DeclareLaunchArgument('world_file', default_value='',
                              description='world yaml (default: config/hw4_world.yaml)'),
        DeclareLaunchArgument('params_file', default_value='',
                              description='node parameters (default: config/hw4.yaml)'),
        DeclareLaunchArgument('initial_pose', default_value='',
                              description='"x,y,theta" overriding the world start pose'),
        DeclareLaunchArgument('use_imu', default_value='',
                              description='true/false (default: from the params file)'),
        DeclareLaunchArgument('record_bag', default_value='false'),
        DeclareLaunchArgument('record_images', default_value='false'),
        DeclareLaunchArgument('log_dir', default_value='',
                              description='runs folder (default: <workspace>/hw4_logs)'),
        DeclareLaunchArgument('run_tag', default_value='',
                              description='optional suffix for the run folder name'),
        DeclareLaunchArgument('plot', default_value='true',
                              description='write plan PNG at launch (needs matplotlib)'),
        DeclareLaunchArgument('auto_eval', default_value='true',
                              description='run hw4 eval on the run folder when it ends'),
    ]
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('robot_vision_camera'), 'launch', 'robot_vision_camera.launch.py'])),
        launch_arguments={'image_rectify': 'true'}.items(),
    )
    apriltag = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('apriltag_ros'), 'launch', 'apriltag_launch.py'])),
    )
    motor_node = Node(
        package='robot_control',
        executable='motor_control',
        name='motor_controller_node',
        output='screen',
        emulate_tty=True,
    )
    shutdown_when_motor_exits = RegisterEventHandler(OnProcessExit(
        target_action=motor_node, on_exit=[Shutdown(reason='Motor controller exited')]))
    return LaunchDescription(args + [
        camera,
        apriltag,
        motor_node,
        shutdown_when_motor_exits,
        OpaqueFunction(function=launch_setup),
    ])
