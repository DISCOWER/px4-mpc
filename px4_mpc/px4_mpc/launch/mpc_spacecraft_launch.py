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
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import tempfile


def generate_launch_description():
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for all nodes'
    )
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='wrench',
        description='Mode of the controller (rate, wrench, direct_allocation, propeller)'
    )
    px4_uses_ned_arg = DeclareLaunchArgument(
        'px4_uses_ned',
        default_value='true',
        description='PX4 uses NED frame (default: true) or ENU frame (false)'
    )
    rviz_mode_arg = DeclareLaunchArgument(
        'rviz_mode',
        default_value='setpoint',
        choices=['off', 'viz', 'setpoint'],
        description='RViz mode: off (no RViz), viz (visualization only) or setpoint (visualization and setpoints from RViz)'
    )
    skip_build_arg = DeclareLaunchArgument(
        'skip_build',
        default_value='false',
        description='Skip acados build (default: false)'
    )
    kthspace_constraints_arg = DeclareLaunchArgument(
        'kthspace_constraints',
        default_value='false',
        description='Use KTH Space Lab constraints (default: false)'
    )
    sitl_arg = DeclareLaunchArgument(
        'sitl',
        default_value='false',
        description='Running in SITL (default: false)'
    )

    namespace = LaunchConfiguration('namespace')
    mode = LaunchConfiguration('mode')
    px4_uses_ned = LaunchConfiguration('px4_uses_ned')
    rviz_mode = LaunchConfiguration('rviz_mode')
    skip_build = LaunchConfiguration('skip_build')
    kthspace_constraints = LaunchConfiguration('kthspace_constraints')
    sitl = LaunchConfiguration('sitl')

    ld = LaunchDescription()

    ld.add_action(namespace_arg)
    ld.add_action(mode_arg)
    ld.add_action(px4_uses_ned_arg)
    ld.add_action(rviz_mode_arg)
    ld.add_action(skip_build_arg)
    ld.add_action(kthspace_constraints_arg)
    ld.add_action(sitl_arg)

    ld.add_action(Node(
        package='px4_mpc',
        namespace=namespace,
        executable='mpc_spacecraft',
        name='mpc_spacecraft',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'mode': mode},
            {'px4_uses_ned': px4_uses_ned},
            {'rviz_mode': rviz_mode},
            {'skip_build': skip_build},
            {'kthspace_constraints': kthspace_constraints},
            {'sitl': sitl}
        ]
    ))
    ld.add_action(OpaqueFunction(function=launch_setup))

    return ld

def patch_rviz_config(original_config_path, namespace):
    """
    Patch the RViz configuration file to replace the namespace placeholder with the actual namespace.
    """
    with open(original_config_path, 'r') as f:
        content = f.read()

    # Replace placeholder with actual namespace
    content = content.replace('__NS__', f'/{namespace}' if namespace else '')

    # Write to temporary file
    tmp_rviz_config = tempfile.NamedTemporaryFile(delete=False, suffix='.rviz')
    tmp_rviz_config.write(content.encode('utf-8'))
    tmp_rviz_config.close()

    return tmp_rviz_config.name


def launch_setup(context, *args, **kwargs):
    """
    Function to set up the launch context, the setpoint source and RViz.
    """
    namespace = LaunchConfiguration('namespace').perform(context)
    rviz_mode = LaunchConfiguration('rviz_mode').perform(context)

    actions = []

    # Setpoint source: interactive marker in RViz, or predefined setpoints
    if rviz_mode == 'setpoint':
        actions.append(Node(
            package='px4_mpc',
            namespace=namespace,
            executable='rviz_pos_marker',
            name='rviz_pos_marker',
            output='screen',
            emulate_tty=True
        ))
    else:
        actions.append(Node(
            package='px4_mpc',
            namespace=namespace,
            executable='test_setpoints',
            name='test_setpoints',
            output='screen',
            emulate_tty=True
        ))

    # RViz and vehicle visualization
    if rviz_mode in ('viz', 'setpoint'):
        rviz_config_path = os.path.join(get_package_share_directory('px4_mpc'), 'config.rviz')
        patched_config = patch_rviz_config(rviz_config_path, namespace)
        actions.append(Node(
            package='px4_offboard',
            namespace=namespace,
            executable='visualizer',
            name='visualizer',
            parameters=[
                {'px4_uses_ned': LaunchConfiguration('px4_uses_ned')},
            ]
        ))
        actions.append(Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', patched_config]
        ))

    # Package for propeller plate interface (only resolved in propeller mode)
    if LaunchConfiguration('mode').perform(context) == 'propeller':
        config = os.path.join(
            get_package_share_directory("propeller_plate_iface"),
            "config",
            "parameters.yaml",
        )
        actions.append(Node(
            package="propeller_plate_iface",
            executable="propeller_plate_iface_node",
            name="propeller_plate_iface_node",
            namespace=namespace,
            output="screen",
            parameters=[config],
        ))

    return actions