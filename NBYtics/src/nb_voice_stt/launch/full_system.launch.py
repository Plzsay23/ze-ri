from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    bridge_port = LaunchConfiguration("bridge_port")
    bridge_baud = LaunchConfiguration("bridge_baud")
    scan_topic = LaunchConfiguration("scan_topic")
    mode_cmd_topic = LaunchConfiguration("mode_cmd_topic")
    safe_cmd_topic = LaunchConfiguration("safe_cmd_topic")

    nb_voice_stt_share = get_package_share_directory("nb_voice_stt")
    ydlidar_share = get_package_share_directory("ydlidar_ros2_driver")

    stt_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nb_voice_stt_share, "launch", "stt.launch.py")
        )
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ydlidar_share, "launch", "ydlidar_launch.py")
        )
    )

    voice_mode_manager_node = Node(
        package="nb_voice_stt",
        executable="voice_mode_manager",
        name="voice_mode_manager",
        output="screen",
        parameters=[{
            "output_cmd_topic": mode_cmd_topic,
        }]
    )

    lidar_guard_node = Node(
        package="nb_voice_stt",
        executable="lidar_guard_node",
        name="lidar_guard_node",
        output="screen",
        parameters=[{
            "scan_topic": scan_topic,
            "input_cmd_topic": mode_cmd_topic,
            "output_cmd_topic": safe_cmd_topic,
        }]
    )

    cmd_serial_bridge_node = Node(
        package="nb_voice_stt",
        executable="cmd_serial_bridge",
        name="cmd_serial_bridge",
        output="screen",
        parameters=[{
            "port": bridge_port,
            "baudrate": bridge_baud,
            "input_topic": safe_cmd_topic,
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "bridge_port",
            default_value="/dev/arduino_nb",
            description="Arduino serial port"
        ),
        DeclareLaunchArgument(
            "bridge_baud",
            default_value="9600",
            description="Arduino serial baudrate"
        ),
        DeclareLaunchArgument(
            "scan_topic",
            default_value="/scan",
            description="LaserScan topic"
        ),
        DeclareLaunchArgument(
            "mode_cmd_topic",
            default_value="/mode_cmd",
            description="Mode manager output topic"
        ),
        DeclareLaunchArgument(
            "safe_cmd_topic",
            default_value="/safe_cmd",
            description="Lidar guard output topic"
        ),

        lidar_launch,
        stt_launch,

        TimerAction(
            period=2.0,
            actions=[voice_mode_manager_node]
        ),

        TimerAction(
            period=3.0,
            actions=[lidar_guard_node]
        ),

        TimerAction(
            period=4.0,
            actions=[cmd_serial_bridge_node]
        ),
    ])