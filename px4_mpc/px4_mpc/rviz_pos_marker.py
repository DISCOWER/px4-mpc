#!/usr/bin/env python3

# Copyright (c) 2011, Willow Garage, Inc.
# All rights reserved.
#
# Software License Agreement (BSD License 2.0)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Willow Garage, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import copy
from math import sin
from random import random
import sys

from geometry_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from interactive_markers import InteractiveMarkerServer
from geometry_msgs.msg import Pose
from interactive_markers import MenuHandler
import rclpy
from rosidl_runtime_py import set_message_fields
from tf2_ros.transform_broadcaster import TransformBroadcaster
from visualization_msgs.msg import InteractiveMarker
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import InteractiveMarkerFeedback
from visualization_msgs.msg import Marker
from rclpy.node import Node

from mpc_msgs.srv import SetPose


def rand(min_, max_):
    return min_ + random() * (max_ - min_)

def makeBox(msg):
    marker = Marker()

    marker.type = Marker.SPHERE
    marker.scale.x = msg.scale * 0.45
    marker.scale.y = msg.scale * 0.45
    marker.scale.z = msg.scale * 0.45
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    return marker


def makeBoxControl(msg):
    control = InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append(makeBox(msg))
    msg.controls.append(control)
    return control


def normalizeQuaternion(quaternion_msg):
    norm = quaternion_msg.x**2 + quaternion_msg.y**2 + quaternion_msg.z**2 + quaternion_msg.w**2
    s = norm**(-0.5)
    quaternion_msg.x *= s
    quaternion_msg.y *= s
    quaternion_msg.z *= s
    quaternion_msg.w *= s


def make6DofMarker(server, menu_handler, process_feedback, fixed, interaction_mode, position, show_6dof=False):
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = 'map'
    int_marker.pose.position = position
    int_marker.scale = 1.0

    int_marker.name = 'simple_6dof'
    int_marker.description = 'Simple 6-DOF Control'

    # insert a box
    makeBoxControl(int_marker)
    int_marker.controls[0].interaction_mode = InteractiveMarkerControl.MENU

    if fixed:
        int_marker.name += '_fixed'
        int_marker.description += '\n(fixed orientation)'

    if interaction_mode != InteractiveMarkerControl.NONE:
        control_modes_dict = {
            InteractiveMarkerControl.MOVE_3D: 'MOVE_3D',
            InteractiveMarkerControl.ROTATE_3D: 'ROTATE_3D',
            InteractiveMarkerControl.MOVE_ROTATE_3D: 'MOVE_ROTATE_3D'
        }
        int_marker.name += '_' + control_modes_dict[interaction_mode]
        int_marker.description = '3D Control'
        if show_6dof:
            int_marker.description += ' + 6-DOF controls'
        int_marker.description += '\n' + control_modes_dict[interaction_mode]

    if show_6dof:
        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 1.0
        control.orientation.y = 0.0
        control.orientation.z = 0.0
        normalizeQuaternion(control.orientation)
        control.name = 'move_x'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 1.0
        control.orientation.z = 0.0
        normalizeQuaternion(control.orientation)
        control.name = 'move_z'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 0.0
        control.orientation.z = 1.0
        normalizeQuaternion(control.orientation)
        control.name = 'move_y'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)
    server.insert(int_marker, feedback_callback=process_feedback)
    menu_handler.apply(server, int_marker.name)

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(SetPose, '/set_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetPose.Request()

    def send_request(self, pose):
        self.req.pose.position.x = pose.position.x
        self.req.pose.position.y = pose.position.y
        self.req.pose.position.z = pose.position.z
        self.req.pose.orientation.w = pose.orientation.w
        self.req.pose.orientation.x = pose.orientation.x
        self.req.pose.orientation.y = pose.orientation.y
        self.req.pose.orientation.z = pose.orientation.z
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


class ProcessFeedback():
    def __init__(self, node):
        self.marker_pose = Pose()
        self.minimal_client = MinimalClientAsync()
        self.node = node
        self.br = TransformBroadcaster(self.node)
        self.counter = 0
        # create a timer to update the published transforms
        timer = self.node.create_timer(0.01, self.frameCallback)
        self.menu_handler = MenuHandler()

        self.server = InteractiveMarkerServer(node, 'rviz_target_pose_marker')

        self.menu_handler.insert('Command Pose', callback=self.processFeedback)
        sub_menu_handle = self.menu_handler.insert('Reset')
        # self.menu_handler.insert('Reset Position', parent=sub_menu_handle, callback=process_feedback.processFeedback)
        # self.menu_handler.insert('Reset Orientation', parent=sub_menu_handle, callback=process_feedback.processFeedback)

        position = Point(x=0.0, y=3.0, z=3.0)
        make6DofMarker(self.server, self.menu_handler, self.processFeedback, True, InteractiveMarkerControl.NONE, position, True)
        self.server.applyChanges()

    
    def processFeedback(self, feedback):
        log_prefix = (
            f"Feedback from marker '{feedback.marker_name}' / control '{feedback.control_name}'"
        )

        log_mouse = ''
        if feedback.mouse_point_valid:
            log_mouse = (
                f'{feedback.mouse_point.x}, {feedback.mouse_point.y}, '
                f'{feedback.mouse_point.z} in frame {feedback.header.frame_id}'
            )

        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            self.node.get_logger().info(f'{log_prefix}: button click at {log_mouse}')
        elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            if feedback.menu_entry_id == 1:
                response = self.minimal_client.send_request(self.marker_pose)

        elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            # node.get_logger().info(
            #     f'{log_prefix}: pose changed\n'
            #     f'position: '
            #     f'{feedback.pose.position.x}, {feedback.pose.position.y}, {feedback.pose.position.z}\n'
            #     f'orientation: '
            #     f'{feedback.pose.orientation.w}, {feedback.pose.orientation.x}, '
            #     f'{feedback.pose.orientation.y}, {feedback.pose.orientation.z}\n'
            #     f'frame: {feedback.header.frame_id} '
            #     f'time: {feedback.header.stamp.sec} sec, '
            #     f'{feedback.header.stamp.nanosec} nsec'
            # )
            self.marker_pose = feedback.pose
            # node.get_logger().info(
            #     f'{log_prefix}: Setpose: {self.marker_pose}'
            # )
            
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self.node.get_logger().info(f'{log_prefix}: mouse down at {log_mouse}')
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            self.node.get_logger().info(f'{log_prefix}: mouse up at {log_mouse}')

    def frameCallback(self):
        time = self.node.get_clock().now()
        transform = TransformStamped()
        set_message_fields(
            transform,
            {
                'header': {'frame_id': 'map', 'stamp': time.to_msg()},
                'transform': {
                    'translation': {
                        'x': 0.0,
                        'y': 0.0,
                        'z': sin(self.counter / 140.0) * 2.0,
                    },
                    'rotation': {
                        'x': 0.0,
                        'y': 0.0,
                        'z': 0.0,
                        'w': 1.0,
                    },
                },
                'child_frame_id': 'moving_frame',
            },
        )
        self.br.sendTransform(transform)
        self.counter += 1

    def alignMarker(self, feedback):
        pose = feedback.pose

        pose.position.x = round(pose.position.x - 0.5) + 0.5
        pose.position.y = round(pose.position.y - 0.5) + 0.5

        self.node.get_logger().info(
            f'{feedback.marker_name}: aligning position = {feedback.pose.position.x}, '
            f'{feedback.pose.position.y}, {feedback.pose.position.z} to '
            f'{pose.position.x}, {pose.position.y}, {pose.position.z}'
        )

        self.server.setPose(feedback.marker_name, pose)
        self.server.applyChanges()

def main(args=None):
    rclpy.init(args=sys.argv)
    node = rclpy.create_node('rviz_target_pos_marker')

    process_feedback = ProcessFeedback(node)

    rclpy.spin(node)
    # server.shutdown()

if __name__ == '__main__':
    main()
