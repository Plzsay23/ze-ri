from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "sensevoice_ko"
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_dir = LaunchConfiguration("model_dir")
    audio_device = LaunchConfiguration("audio_device")
    channels = LaunchConfiguration("channels")
    use_channel_index = LaunchConfiguration("use_channel_index")
    sample_rate = LaunchConfiguration("sample_rate")
    device = LaunchConfiguration("device")

    return LaunchDescription([
        DeclareLaunchArgument(
            "model_dir",
            default_value=str(DEFAULT_MODEL_DIR),
        ),
        DeclareLaunchArgument(
            "audio_device",
            default_value="-1",
        ),
        DeclareLaunchArgument(
            "channels",
            default_value="1",
        ),
        DeclareLaunchArgument(
            "use_channel_index",
            default_value="0",
        ),
        DeclareLaunchArgument(
            "sample_rate",
            default_value="16000",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cuda",
        ),

        Node(
            package="nb_voice_stt",
            executable="stt_node",
            name="sensevoice_stt_node",
            output="screen",
            parameters=[{
                "model_dir": model_dir,
                "audio_device": audio_device,
                "channels": channels,
                "use_channel_index": use_channel_index,
                "sample_rate": sample_rate,
                "device": device,
            }],
        )
    ])
