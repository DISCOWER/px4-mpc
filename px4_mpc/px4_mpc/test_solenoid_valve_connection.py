import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import ActuatorMotors
from px4_msgs.msg import VehicleCommand


import time


class MinimalPublisherPX4(Node):

    def __init__(self):
        super().__init__("minimal_publisher_to_px4")
        self.namespace_prefix = ""

        # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0,
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0,
        )

        # Subscrivers
        self.status_sub = self.create_subscription(
            VehicleStatus,
            f"{self.namespace_prefix}/fmu/out/vehicle_status",
            self.vehicle_status_callback,
            qos_profile_sub,
        )

        # Publishers
        self.publisher_vehicle_command = self.create_publisher(
            VehicleCommand,
            f"{self.namespace_prefix}/fmu/in/vehicle_command",
            qos_profile_pub,
        )

        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode,
            f"{self.namespace_prefix}/fmu/in/offboard_control_mode",
            qos_profile_pub,
        )

        self.publisher_direct_actuator = self.create_publisher(
            ActuatorMotors,
            f"{self.namespace_prefix}/fmu/in/actuator_motors",
            qos_profile_pub,
        )

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        # Enable direct actuator mode
        self.enable_offboard_control()
        # self.enable_direct_actuator_mode()

        # Enable arm
        self.arm()

        self.curr_time = 0

    def arm(self):
        self.get_logger().info("Arming vehicle")
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0
        )

    def enable_offboard_control(self):
        self.get_logger().info("Enabling offboard control")
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            6.0,  # Offboard mode
        )

    def enable_direct_actuator_mode(self):
        self.get_logger().info("Enabling direct actuator mode")
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            7.0,  # Direct actuator mode
        )

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 2
        msg.target_component = 1
        msg.source_system = 1  # your node ID
        msg.source_component = 100  # your component ID
        msg.from_external = True
        self.publisher_vehicle_command.publish(msg)
        self.get_logger().info(
            f"Sent command: {command}, param1: {param1}, param2: {param2}"
        )

    def publish_direct_actuator_mode(self):
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.direct_actuator = True
        self.publisher_offboard_mode.publish(offboard_msg)
        self.get_logger().info("Sent direct actuator mode")

    def publish_direct_actuator_setpoint(self, u_command):
        actuator_outputs_msg = ActuatorMotors()
        actuator_outputs_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        # NOTE:
        # Output is float[16]
        # u1 needs to be divided between 1 and 2
        # u2 needs to be divided between 3 and 4
        # u3 needs to be divided between 5 and 6
        # u4 needs to be divided between 7 and 8
        # positve component goes for the first, the negative for the second
        thrust = u_command[0, :] / 1.5  # normalizes w.r.t. max thrust
        # print("Thrust rates: ", thrust[0:4])

        thrust_command = np.zeros(12, dtype=np.float32)
        thrust_command[0] = 0.0 if thrust[0] <= 0.0 else thrust[0]
        thrust_command[1] = 0.0 if thrust[0] >= 0.0 else -thrust[0]

        thrust_command[2] = 0.0 if thrust[1] <= 0.0 else thrust[1]
        thrust_command[3] = 0.0 if thrust[1] >= 0.0 else -thrust[1]

        thrust_command[4] = 0.0 if thrust[2] <= 0.0 else thrust[2]
        thrust_command[5] = 0.0 if thrust[2] >= 0.0 else -thrust[2]

        thrust_command[6] = 0.0 if thrust[3] <= 0.0 else thrust[3]
        thrust_command[7] = 0.0 if thrust[3] >= 0.0 else -thrust[3]

        actuator_outputs_msg.control = thrust_command.flatten()
        self.publisher_direct_actuator.publish(actuator_outputs_msg)

    def cmdloop_callback(self):
    
        self.publish_direct_actuator_mode()

        u_command = np.zeros((1, 8))
        u_command[0, 3] = 1.0
        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_direct_actuator_setpoint(u_command)


def main(args=None):
    rclpy.init(args=args)

    spacecraft_mpc = MinimalPublisherPX4()
    rclpy.spin(spacecraft_mpc)
    spacecraft_mpc.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
