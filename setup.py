from setuptools import setup
from setuptools import find_packages
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
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
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
                'quadrotor_demo = px4_mpc.quadcopter_demo:main',
                'astrobee_demo = px4_mpc.astrobee_demo:main',
        ],
    },
)
