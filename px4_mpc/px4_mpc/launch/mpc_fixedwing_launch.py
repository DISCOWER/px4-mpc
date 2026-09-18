#!/usr/bin/env python3
__author__ = "Your Name"

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import tempfile

def patch_rviz_config(original_config_path, namespace):
    """
    Patch the RViz configuration file to replace the namespace placeholder with the actual namespace.
    """
    # Safe fallback just in case the .rviz file isn't in the install space yet
    if not os.path.exists(original_config_path):
        return ''
        
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
    Function to set up the launch context and patch the RViz configuration.
    """
    namespace = LaunchConfiguration('namespace').perform(context)
    
    rviz_config_path = os.path.join(get_package_share_directory('px4_mpc'), 'fw_config.rviz')
    patched_config = patch_rviz_config(rviz_config_path, namespace)

    rviz_args = ['-d', patched_config] if patched_config else []

    return [
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=rviz_args,
            condition=IfCondition(LaunchConfiguration('use_rviz'))
        )
    ]

def generate_launch_description():
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for all nodes'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2 alongside the MPC node'
    )

    namespace = LaunchConfiguration('namespace')

    return LaunchDescription([
        namespace_arg,
        use_rviz_arg,
        Node(
            package='px4_mpc',
            namespace=namespace,
            executable='mpc_fixedwing', # Triggers the entry_point in your setup.py
            name='mpc_fixedwing',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'namespace': namespace}
            ]
        ),
        Node(
            package='px4_mpc',
            namespace=namespace,
            executable='fw_viz', 
            name='fw_viz',
            output='screen',
            emulate_tty=True
        ),
        OpaqueFunction(function=launch_setup),
    ])