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

import rclpy
from geometry_msgs.msg import Point
from interactive_markers import InteractiveMarkerServer, MenuHandler
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from mpc_msgs.srv import SetPose
from math import sqrt

def makeSphereControl(int_marker):
    marker = Marker()
    marker.type = Marker.SPHERE
    marker.scale.x = int_marker.scale * 0.4
    marker.scale.y = int_marker.scale * 0.4
    marker.scale.z = int_marker.scale * 0.4
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    control = InteractiveMarkerControl()
    control.always_visible = True
    control.interaction_mode = InteractiveMarkerControl.MENU
    control.markers.append(marker)
    return control

def makeAxisControls():
    # Move and rotate controls about each axis, fixed to the world frame
    controls = []
    for axis in 'xyz':
        for action, mode in [('move', InteractiveMarkerControl.MOVE_AXIS),
                             ('rotate', InteractiveMarkerControl.ROTATE_AXIS)]:
            control = InteractiveMarkerControl()
            control.orientation.w = 1.0 / sqrt(2.0)
            setattr(control.orientation, axis, 1.0 / sqrt(2.0))
            control.name = f'{action}_{axis}'
            control.interaction_mode = mode
            control.orientation_mode = InteractiveMarkerControl.FIXED
            controls.append(control)
    return controls

def makeTargetMarker(position):
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = 'map'
    int_marker.pose.position = position
    int_marker.scale = 0.3
    int_marker.name = 'target_pose'
    int_marker.controls.append(makeSphereControl(int_marker))
    int_marker.controls.extend(makeAxisControls())
    return int_marker

class RvizPosMarker(Node):
    def __init__(self):
        super().__init__('rviz_pos_marker')

        self.set_pose_client = self.create_client(SetPose, 'set_pose')

        self.server = InteractiveMarkerServer(self, 'rviz_target_pose_marker')
        self.menu_handler = MenuHandler()
        self.menu_handler.insert('Command Pose', callback=self.commandPoseCallback)

        int_marker = makeTargetMarker(Point(x=1.0, y=1.0, z=0.0))
        self.server.insert(int_marker)
        self.menu_handler.apply(self.server, int_marker.name)
        self.server.applyChanges()

    def commandPoseCallback(self, feedback):
        # feedback.pose is the marker's current pose, whether or not it has been moved
        if not self.set_pose_client.service_is_ready():
            self.get_logger().warn("Service 'set_pose' not available, is the MPC running?")
            return

        request = SetPose.Request()
        request.pose = feedback.pose
        self.set_pose_client.call_async(request)
        p = feedback.pose.position
        self.get_logger().info(f'Commanded pose: {p.x:.2f}, {p.y:.2f}, {p.z:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = RvizPosMarker()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.server.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()