from setuptools import find_packages, setup

package_name = 'zeri_base'

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
    description='Ze-Ri serial mobile base bridge and odometry',
    license='MIT',
    entry_points={
        'console_scripts': [
            'base_key_odom_serial_node = zeri_base.base_key_odom_serial_node:main',
            'base_velocity_odom_serial_node = zeri_base.base_velocity_odom_serial_node:main',
        ],
    },
)
