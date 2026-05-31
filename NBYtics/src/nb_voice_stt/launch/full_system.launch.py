from pathlib import Path
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    zeri_root = os.environ.get("ZERI_ROOT", str(Path.home() / "ze-ri"))
    nbytics_root = os.environ.get("NBYTICS_ROOT", os.path.join(zeri_root, "NBYtics"))

    scan_front_filter_script = os.path.join(nbytics_root, "tools", "scan_front_filter.py")
    cmd_odom_bridge_script = os.path.join(nbytics_root, "tools", "cmd_odom_serial_bridge.py")
    respeaker_node_script = os.path.join(zeri_root, "src", "lerobot", "vlm_agent", "respeaker_vad_doa_node.py")
    voice_doa_turn_script = os.path.join(zeri_root, "src", "lerobot", "vlm_agent", "voice_doa_turn_node.py")
    slam_params_file = os.path.join(nbytics_root, "slam_config", "mapper_params_online_async.yaml")

    bridge_port = LaunchConfiguration("bridge_port")
    bridge_baud = LaunchConfiguration("bridge_baud")
    raw_scan_topic = LaunchConfiguration("raw_scan_topic")
    scan_front_topic = LaunchConfiguration("scan_front_topic")
    mode_cmd_topic = LaunchConfiguration("mode_cmd_topic")
    safe_cmd_topic = LaunchConfiguration("safe_cmd_topic")
    lidar_port = LaunchConfiguration("lidar_port")
    lidar_yaw_deg = LaunchConfiguration("lidar_yaw_deg")
    lidar_yaw_rad = LaunchConfiguration("lidar_yaw_rad")
    voice_front_deg = LaunchConfiguration("voice_front_deg")
    voice_invert_direction = LaunchConfiguration("voice_invert_direction")
    enable_slam = LaunchConfiguration("enable_slam")
    enable_stt = LaunchConfiguration("enable_stt")
    enable_respeaker = LaunchConfiguration("enable_respeaker")
    enable_voice_doa = LaunchConfiguration("enable_voice_doa")

    nb_voice_stt_share = get_package_share_directory("nb_voice_stt")
    slam_toolbox_share = get_package_share_directory("slam_toolbox")

    stt_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nb_voice_stt_share, "launch", "stt.launch.py")
        ),
        condition=IfCondition(enable_stt),
    )

    lidar_node = Node(
        package="ydlidar_ros2_driver",
        executable="ydlidar_ros2_driver_node",
        name="ydlidar_ros2_driver_node",
        output="screen",
        parameters=[{
            "port": lidar_port,
            "frame_id": "laser_frame_raw",
            "baudrate": 115200,
            "lidar_type": 1,
            "device_type": 0,
            "isSingleChannel": True,
            "intensity": False,
        }],
    )

    lidar_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_pub_laser_raw",
        output="screen",
        arguments=[
            "--x", "0.20",
            "--y", "0.00",
            "--z", "0.10",
            "--roll", "0",
            "--pitch", "0",
            "--yaw", lidar_yaw_rad,
            "--frame-id", "base_link",
            "--child-frame-id", "laser_frame_raw",
        ],
    )

    scan_front_filter = ExecuteProcess(
        cmd=[
            "python3", scan_front_filter_script,
            "--ros-args",
            "-p", ["input_topic:=", raw_scan_topic],
            "-p", ["output_topic:=", scan_front_topic],
            "-p", "min_angle_deg:=-90.0",
            "-p", "max_angle_deg:=90.0",
            "-p", ["lidar_yaw_deg:=", lidar_yaw_deg],
            "-p", "min_keep_range:=0.45",
            "-p", "max_keep_range:=6.0",
            "-p", "fixed_bins:=720",
        ],
        name="scan_front_filter",
        output="screen",
    )

    cmd_odom_bridge = ExecuteProcess(
        cmd=[
            "python3", cmd_odom_bridge_script,
            "--ros-args",
            "-p", ["port:=", bridge_port],
            "-p", ["baudrate:=", bridge_baud],
            "-p", ["input_topic:=", safe_cmd_topic],
            "-p", "cmd_period_sec:=0.08",
            "-p", "command_timeout_sec:=0.8",
            "-p", "ticks_per_rev:=3464.0",
            "-p", "wheel_radius:=0.075",
            "-p", "lx:=0.1575",
            "-p", "ly:=0.2125",
            "-p", "invert_vy:=true",
            "-p", "invert_wz:=true",
        ],
        name="cmd_odom_serial_bridge",
        output="screen",
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_share, "launch", "online_async_launch.py")
        ),
        launch_arguments={
            "use_sim_time": "false",
            "slam_params_file": slam_params_file,
        }.items(),
        condition=IfCondition(enable_slam),
    )

    respeaker_vad_doa = ExecuteProcess(
        cmd=["python3", respeaker_node_script],
        name="zeri_respeaker_vad_doa_node",
        output="screen",
        condition=IfCondition(enable_respeaker),
    )

    voice_mode_manager_node = Node(
        package="nb_voice_stt",
        executable="voice_mode_manager",
        name="voice_mode_manager",
        output="screen",
        parameters=[{
            "output_cmd_topic": mode_cmd_topic,
            "allow_auto_mode": True,
        }],
    )

    lidar_guard_node = Node(
        package="nb_voice_stt",
        executable="lidar_guard_node",
        name="lidar_guard_node",
        output="screen",
        parameters=[{
            "scan_topic": scan_front_topic,
            "input_cmd_topic": mode_cmd_topic,
            "output_cmd_topic": safe_cmd_topic,
            "lidar_yaw_deg": ParameterValue(lidar_yaw_deg, value_type=float),
            "stop_distance": 0.60,
            "front_half_angle_deg": 18.0,
        }],
    )

    voice_doa_turn = ExecuteProcess(
        cmd=[
            "python3", voice_doa_turn_script,
            "--ros-args",
            "-p", "speech_vad_topic:=/zeri/audio/vad",
            "-p", "doa_topic:=/zeri/audio/doa_deg",
            "-p", ["front_deg:=", voice_front_deg],
            "-p", ["invert_direction:=", voice_invert_direction],
            "-p", "turn_start_deadband_deg:=18.0",
            "-p", "turn_stop_deadband_deg:=8.0",
            "-p", "forward_hold_sec:=6.0",
            "-p", "smoothing_alpha:=0.35",
        ],
        name="voice_doa_turn_node",
        output="screen",
        condition=IfCondition(enable_voice_doa),
    )

    return LaunchDescription([
        DeclareLaunchArgument("bridge_port", default_value="/dev/arduino", description="Arduino wheel controller serial port"),
        DeclareLaunchArgument("bridge_baud", default_value="115200", description="Arduino serial baudrate"),
        DeclareLaunchArgument("lidar_port", default_value="/dev/lidar", description="YDLiDAR serial port"),
        DeclareLaunchArgument("raw_scan_topic", default_value="/scan", description="Raw LaserScan topic"),
        DeclareLaunchArgument("scan_front_topic", default_value="/scan_front", description="Robot-front filtered LaserScan topic"),
        DeclareLaunchArgument("mode_cmd_topic", default_value="/mode_cmd", description="Mode manager output topic"),
        DeclareLaunchArgument("safe_cmd_topic", default_value="/safe_cmd", description="Lidar guard output topic"),
        DeclareLaunchArgument("lidar_yaw_deg", default_value="180.0", description="LiDAR yaw offset in robot frame"),
        DeclareLaunchArgument("lidar_yaw_rad", default_value="3.14159", description="LiDAR yaw offset for static TF"),
        DeclareLaunchArgument("voice_front_deg", default_value="0.0", description="DOA angle treated as robot front"),
        DeclareLaunchArgument("voice_invert_direction", default_value="false", description="Invert DOA turn direction"),
        DeclareLaunchArgument("enable_slam", default_value="true", description="Start slam_toolbox mapping"),
        DeclareLaunchArgument("enable_stt", default_value="true", description="Start SenseVoice STT"),
        DeclareLaunchArgument("enable_respeaker", default_value="true", description="Start ReSpeaker VAD/DOA"),
        DeclareLaunchArgument("enable_voice_doa", default_value="true", description="Turn and drive toward voice DOA"),

        lidar_node,
        lidar_tf_node,
        stt_launch,
        respeaker_vad_doa,

        TimerAction(period=1.0, actions=[scan_front_filter]),
        TimerAction(period=2.0, actions=[cmd_odom_bridge]),
        TimerAction(period=2.0, actions=[voice_mode_manager_node]),
        TimerAction(period=3.0, actions=[lidar_guard_node]),
        TimerAction(period=4.0, actions=[slam_launch]),
        TimerAction(period=4.0, actions=[voice_doa_turn]),
    ])
