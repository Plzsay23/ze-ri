from setuptools import find_packages, setup

package_name = 'zeri_camera'

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
    description='Ze-Ri camera publisher and camera-follow nodes',
    license='MIT',
    entry_points={
        'console_scripts': [
            'realsense_rgbd_node = zeri_camera.realsense_rgbd_node:main',
            'camera_depth_follow_node = zeri_camera.camera_depth_follow_node:main',
            'camera_person_follow_node = zeri_camera.camera_person_follow_node:main',
        ],
    },
)
