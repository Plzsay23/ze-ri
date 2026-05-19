from setuptools import setup

package_name = 'nb_odom'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='hansungai',
    maintainer_email='hansungai@example.com',
    description='Encoder odometry node for NBYtics mecanum robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'encoder_odom_node = nb_odom.encoder_odom_node:main',
            'keyboard_odom_node = nb_odom.keyboard_odom_node:main',
        ],
    },
)
