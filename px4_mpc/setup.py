from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'px4_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob(os.path.join('px4_mpc/launch', '*launch.[pxy][yma]*'))),
        # (os.path.join('share', package_name), glob('launch/*.[pxy][yma]*')),
        (os.path.join('share', package_name), glob(os.path.join('px4_mpc/config', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jaeyoung',
    maintainer_email='jalim@ethz.ch',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'mpc_quadrotor = px4_mpc.mpc_quadrotor:main',
                'rviz_pos_marker = px4_mpc.rviz_pos_marker:main',
        ],
    },
)
