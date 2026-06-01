from setuptools import find_packages, setup

package_name = 'zeri_lidar'

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
    description='Ze-Ri LiDAR ROS2 package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scan_front_filter_node = zeri_lidar.scan_front_filter_node:main',
            'lidar_guard_node = zeri_lidar.lidar_guard_node:main',
            'lidar_depth_guard_node = zeri_lidar.lidar_depth_guard_node:main',
        ],
    },
)