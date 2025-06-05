#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Pedro Roque, Jaeyoung Lim"
__contact__ = "padr@kth.se, jalim@ethz.ch"

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare launch arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='wrench',
        description='Mode of the controller (rate, wrench, direct_allocation)'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',  # Default namespace is empty
        description='Namespace for all nodes'
    )

    setpoint_from_rviz_arg = DeclareLaunchArgument(
        'setpoint_from_rviz',
        default_value='true',
        description='Publish setpoint pose via rviz'
    )

    mode = LaunchConfiguration('mode')
    namespace = LaunchConfiguration('namespace')
    setpoint_from_rviz = LaunchConfiguration('setpoint_from_rviz')
    sitl = LaunchConfiguration('sitl', default='true')

    return LaunchDescription([
        mode_arg,
        namespace_arg,
        setpoint_from_rviz_arg,
        Node(
            package='px4_mpc',
            namespace=namespace,
            executable='mpc_spacecraft',
            name='mpc_spacecraft',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'mode': mode},
                {'namespace': namespace},
                {'setpoint_from_rviz': setpoint_from_rviz},
                {'sitl': sitl}
            ]
        ),
        Node(
            package='px4_mpc',
            namespace=namespace,
            executable='rviz_pos_marker',
            name='rviz_pos_marker',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'namespace': namespace}
            ],
            condition=IfCondition(LaunchConfiguration('setpoint_from_rviz'))
        ),
        Node(
            package='px4_mpc',
            namespace=namespace,
            executable='test_setpoints',
            name='test_setpoints',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'namespace': namespace},
                {'sitl': sitl}
            ],
            condition=UnlessCondition(LaunchConfiguration('setpoint_from_rviz'))
        ),
        Node(
            package='px4_offboard',
            namespace=namespace,
            executable='visualizer',
            name='visualizer',
            parameters=[
                {'namespace': namespace}
            ],
            condition=IfCondition(LaunchConfiguration('setpoint_from_rviz'))
        ),
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(get_package_share_directory('px4_mpc'), 'config.rviz')],
            condition=IfCondition(LaunchConfiguration('setpoint_from_rviz'))
        ),
    ])
