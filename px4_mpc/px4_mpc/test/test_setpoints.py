#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseStamped
import numpy as np
import time

class SetpointPublisher(Node):
    def __init__(self):
        super().__init__('setpoint_publisher')

        self.namespace = self.declare_parameter('namespace', '').value
        self.namespace_prefix = f'/{self.namespace}' if self.namespace else ''

        #! FIX THIS
        self.publisher_ = self.create_publisher(PoseStamped, f'/itrl_rov_1/px4_mpc/setpoint_pose', 10)
        self.timer_period = 0.01  # seconds
        time.sleep(5) # Give time for all inits...
        self.counter = 0
        # self.setpoints = [
        #     (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        #     (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        #     (1.0, 0.0, 0.0, 0.0, 0.0, 0.383, 0.924),
        #     (1.0, 1.0, 0.0, 0.0, 0.0, 0.707, 0.707),
        #     (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        # ]
        self.setpoints = [
            (5.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0),
            (4.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0),
            (4.0, 0.0, 1.5, 0.0, 0.0, 0.383, 0.924),
            (4.0, 1.0, 1.5, 0.0, 0.0, 0.707, 0.707),
            (5.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0),
        ]
        self.index = -1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        x, y, z, qx, qy, qz, qw = self.setpoints[self.index]
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = Clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        q = np.array([qx, qy, qz, qw])
        q = q / np.linalg.norm(q)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        self.publisher_.publish(pose)

        if self.counter % 2000 == 0:
            self.index = (self.index + 1) % len(self.setpoints)
            print(f"Publishing setpoint {self.index}: {self.setpoints[self.index]}")
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = SetpointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
