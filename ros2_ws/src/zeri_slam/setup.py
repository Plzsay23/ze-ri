from setuptools import find_packages, setup

package_name = 'zeri_slam'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='NBYtics',
    maintainer_email='yjhanna3@hansung.ac.kr',
    description='Zero-Risk SLAM integration package',
    license='MIT',
    entry_points={
        'console_scripts': [
        ],
    },
)
