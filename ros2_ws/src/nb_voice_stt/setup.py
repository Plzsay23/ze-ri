from setuptools import setup, find_packages
import os
from glob import glob

package_name = "nb_voice_stt"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="NBYtics",
    maintainer_email="yjhanna3@hansung.ac.kr",
    description="Real-time SenseVoice STT node for ROS 2",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "stt_node = nb_voice_stt.stt_node:main",
            "cmd_serial_bridge = nb_voice_stt.cmd_serial_bridge:main",
            'lidar_guard_node = nb_voice_stt.lidar_guard_node:main',
            "voice_mode_manager = nb_voice_stt.voice_mode_manager:main",
        ],
    },
)
