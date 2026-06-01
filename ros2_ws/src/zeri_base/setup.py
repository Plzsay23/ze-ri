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
    maintainer='hansungai',
    maintainer_email='yjhanna3@hansung.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cmd_vel_serial_node = zeri_base.cmd_vel_serial_node:main',
            'cmd_vel_safety_node = zeri_base.cmd_vel_safety_node:main',
            'cmd_vel_key_serial_node = zeri_base.cmd_vel_key_serial_node:main',
            'base_key_odom_serial_node = zeri_base.base_key_odom_serial_node:main',
        ],
    },
)
