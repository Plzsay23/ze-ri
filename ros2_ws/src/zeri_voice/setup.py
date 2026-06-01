from setuptools import find_packages, setup

package_name = 'zeri_voice'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zeri',
    maintainer_email='zeri@example.com',
    description='Ze-Ri voice VAD/DOA and voice-follow driving nodes',
    license='MIT',
    entry_points={
        'console_scripts': [
            'respeaker_vad_doa_node = zeri_voice.respeaker_vad_doa_node:main',
            'voice_follow_cmd_node = zeri_voice.voice_follow_cmd_node:main',
            'voice_stop_guard_node = zeri_voice.voice_stop_guard_node:main',
        ],
    },
)
