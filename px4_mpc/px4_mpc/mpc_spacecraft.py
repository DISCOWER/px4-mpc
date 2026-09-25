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

__author__ = "Pedro Roque, Jaeyoung Lim, Elias Krantz"
__contact__ = "padr@kth.se, jalim@ethz.ch, eliaskra@kth.se"

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from visualization_msgs.msg import Marker
from px4_msgs.msg import (OffboardControlMode, VehicleStatus, VehicleTorqueSetpoint, VehicleThrustSetpoint,
                        VehicleOdometry, VehicleRatesSetpoint, ActuatorMotors)
from mpc_msgs.srv import SetPose

DATA_VALIDITY_STREAM = 0.5 # seconds, threshold for (pos,att,vel) messages
DATA_VALIDITY_STATUS = 2.0 # seconds, threshold for status message
INPUT_DEADBAND = 0.005

class SpacecraftMPC(Node):
    def __init__(self):
        super().__init__('spacecraft_mpc')
        self._setup_parameters()

        # Create Spacecraft and controller objects
        if self.mode == 'rate':
            from px4_mpc.models.spacecraft_rate_model import SpacecraftRateModel
            from px4_mpc.controllers.spacecraft_rate_mpc import SpacecraftRateMPC
            self.model = SpacecraftRateModel()
            self.mpc = SpacecraftRateMPC(model=self.model,
                                         skip_build=self.skip_build,
                                         kthspace_constraints=self.kthspace_constraints)
        elif self.mode == 'wrench':
            from px4_mpc.models.spacecraft_wrench_model import SpacecraftWrenchModel
            from px4_mpc.controllers.spacecraft_wrench_mpc import SpacecraftWrenchMPC
            self.model = SpacecraftWrenchModel()
            self.mpc = SpacecraftWrenchMPC(model=self.model,
                                            skip_build=self.skip_build,
                                            kthspace_constraints=self.kthspace_constraints)
        elif self.mode == 'offset_free_wrench':
            from px4_mpc.controllers.spacecraft_offset_free_wrench_mpc import SpacecraftOffsetFreeWrenchMPC
            self.mpc = SpacecraftOffsetFreeWrenchMPC()
        elif self.mode == 'lqr_wrench':
            from px4_mpc.models.spacecraft_wrench_model import SpacecraftWrenchModel
            from px4_mpc.controllers.spacecraft_wrench_lqr import SpacecraftWrenchLQR
            self.mpc = SpacecraftWrenchLQR(model=SpacecraftWrenchModel())
        elif self.mode == 'direct_allocation':
            from px4_mpc.models.spacecraft_direct_allocation_model import SpacecraftDirectAllocationModel
            from px4_mpc.controllers.spacecraft_direct_allocation_mpc import SpacecraftDirectAllocationMPC
            self.model = SpacecraftDirectAllocationModel()
            self.mpc = SpacecraftDirectAllocationMPC(model=self.model,
                                                     skip_build=self.skip_build,
                                                     kthspace_constraints=self.kthspace_constraints)
        elif self.mode == 'propeller':
            from px4_mpc.models.spacecraft_propeller_model import SpacecraftPropellerModel
            from px4_mpc.controllers.spacecraft_propeller_mpc import SpacecraftPropellerMPC
            self.model = SpacecraftPropellerModel()
            self.mpc = SpacecraftPropellerMPC(model=self.model,
                                              skip_build=self.skip_build,
                                              kthspace_constraints=self.kthspace_constraints)
        self.get_logger().info('MPC ready')

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.vehicle_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_position = np.array([0.0, 0.0, 0.0])
        self.setpoint_attitude = np.array([1.0, 0.0, 0.0, 0.0])

        # Set initial timestamps
        self.vehicle_odometry_timestamp = -np.inf
        self.vehicle_status_timestamp = -np.inf

        # Setup publishers and subscribers
        self.set_publishers_subscribers()
        timer_period = 0.05 if self.mode == 'propeller' else 0.1
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

    def _setup_parameters(self):
        self.declare_parameter('mode', 'wrench')
        self.declare_parameter('px4_uses_ned', True)
        self.declare_parameter('rviz_mode', 'setpoint')
        self.declare_parameter('skip_build', False)
        self.declare_parameter('kthspace_constraints', False)
        self.declare_parameter('sitl', False)

        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        self.get_logger().info(f"Mode: {self.mode}")
        self.use_ned = self.get_parameter('px4_uses_ned').get_parameter_value().bool_value
        self.get_logger().info(f"PX4 uses NED frame: {self.use_ned}")
        self.rviz_mode = self.get_parameter('rviz_mode').get_parameter_value().string_value
        if self.rviz_mode not in ('off', 'viz', 'setpoint'):
            raise ValueError(f"Invalid rviz_mode: {self.rviz_mode} (expected off, viz or setpoint)")
        self.get_logger().info(f"RViz mode: {self.rviz_mode}")
        self.setpoint_from_rviz = self.rviz_mode == 'setpoint'
        self.skip_build = self.get_parameter('skip_build').get_parameter_value().bool_value
        self.get_logger().info(f"Skip acados build: {self.skip_build}")
        self.kthspace_constraints = self.get_parameter('kthspace_constraints').get_parameter_value().bool_value
        self.get_logger().info(f"KTH-Space constraints: {self.kthspace_constraints}")
        self.sitl = self.get_parameter('sitl').get_parameter_value().bool_value
        self.get_logger().info(f"SITL: {self.sitl}")

    def set_publishers_subscribers(self):
        qos_fmu_in = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_fmu_out = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.vehicle_odom_sub = self.create_subscription(
            VehicleOdometry,
            'fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback,
            qos_fmu_out)
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status_v4',
            self.vehicle_status_callback,
            qos_fmu_out
        )
        if self.setpoint_from_rviz:
            self.set_pose_srv = self.create_service(
                SetPose,
                'set_pose',
                self.add_set_pose_callback
            )
        else:
            self.setpoint_pose_sub = self.create_subscription(
                PoseStamped,
                'px4_mpc/setpoint_pose',
                self.get_setpoint_pose_callback,
                0
            )

        # Publishers
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode,
            'fmu/in/offboard_control_mode',
            qos_fmu_in)
        self.publisher_rates_setpoint = self.create_publisher(
            VehicleRatesSetpoint,
            'fmu/in/vehicle_rates_setpoint',
            qos_fmu_in)
        self.publisher_direct_actuator = self.create_publisher(
            ActuatorMotors,
            'fmu/in/actuator_motors',
            qos_fmu_in)
        self.publisher_thrust_setpoint = self.create_publisher(
            VehicleThrustSetpoint,
            'fmu/in/vehicle_thrust_setpoint',
            qos_fmu_in)
        self.publisher_torque_setpoint = self.create_publisher(
            VehicleTorqueSetpoint,
            'fmu/in/vehicle_torque_setpoint',
            qos_fmu_in)
        self.publisher_propeller_setpoint = self.create_publisher(
            Float32MultiArray,
            'prop_plate/external_motor_cmd',
            10)
        self.predicted_path_pub = self.create_publisher(
            Path,
            'px4_mpc/predicted_path',
            10)
        self.reference_pub = self.create_publisher(
            Marker,
            'px4_mpc/reference',
            10)
        if self.mode == 'offset_free_wrench':
            self.disturbance_rotation_pub = self.create_publisher(
                Vector3Stamped,
                'px4_mpc/translation_d_hat',
                10)

            self.disturbance_translation_pub = self.create_publisher(
                Vector3Stamped,
                'px4_mpc/attitude_d_hat',
                10)

        if self.sitl:
            self.odom_pub = self.create_publisher(
                Odometry,
                'odom',
                10)
        return

    def vehicle_odometry_callback(self, msg: VehicleOdometry):
        # Store message arrival time in ROS clock domain for validity checking
        self.vehicle_odometry_timestamp = self.get_clock().now().nanoseconds / 1e9

        if self.use_ned:
            # NED-> ENU transformation
            p = np.array([msg.position[1], msg.position[0], -msg.position[2]])
            v = np.array([msg.velocity[1], msg.velocity[0], -msg.velocity[2]])
            q_enu = 1/np.sqrt(2) * np.array([msg.q[0] + msg.q[3], msg.q[1] + msg.q[2], msg.q[1] - msg.q[2], msg.q[0] - msg.q[3]])
            q_enu /= np.linalg.norm(q_enu)
            # FRD -> FLU transformation
            w = np.array([msg.angular_velocity[0], -msg.angular_velocity[1], -msg.angular_velocity[2]])
        else:
            p = np.array([msg.position[0], msg.position[1], msg.position[2]])
            v = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])
            q_enu = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])
            q_enu /= np.linalg.norm(q_enu)
            w = np.array([msg.angular_velocity[0], msg.angular_velocity[1], msg.angular_velocity[2]])

        p[2] = 0.0
        v[2] = 0.0
        w[0:1] = 0.0
        self.vehicle_position = p.astype(float)
        self.vehicle_velocity = v.astype(float)
        self.vehicle_attitude = q_enu.astype(float)
        self.vehicle_angular_velocity = w.astype(float)

    def vehicle_status_callback(self, msg):
        # Store message arrival time in ROS clock domain for validity checking
        self.vehicle_status_timestamp = self.get_clock().now().nanoseconds / 1e9
        self.nav_state = msg.nav_state

    def publish_reference(self, pub, reference):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        # msg.header.stamp = Clock().now().nanoseconds / 1000
        msg.ns = "arrow"
        msg.id = 1
        msg.type = Marker.SPHERE
        msg.scale.x = 0.09
        msg.scale.y = 0.09
        msg.scale.z = 0.09
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = reference[0]
        msg.pose.position.y = reference[1]
        msg.pose.position.z = reference[2]
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        pub.publish(msg)

    def publish_rate_setpoint(self, u_pred):
        F_cmd = u_pred[0, 0:3]
        w_cmd = u_pred[0, 3:6]

        # The PX4 uses normalized force input. Scaling with respect to the maximum force.
        F_scaling = 1/(2 * 1.4)
        F_cmd *= F_scaling

        # Apply deadband to avoid sending very small commands to the solenoids
        F_cmd[np.abs(F_cmd) < INPUT_DEADBAND] = 0.0

        rates_setpoint_msg = VehicleRatesSetpoint()
        rates_setpoint_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        rates_setpoint_msg.roll  = float(w_cmd[0])
        rates_setpoint_msg.pitch = -float(w_cmd[1])
        rates_setpoint_msg.yaw   = -float(w_cmd[2])
        rates_setpoint_msg.thrust_body[0] = float(F_cmd[0])
        rates_setpoint_msg.thrust_body[1] = -float(F_cmd[1])
        rates_setpoint_msg.thrust_body[2] = -float(F_cmd[2])
        self.publisher_rates_setpoint.publish(rates_setpoint_msg)

    def publish_wrench_setpoint(self, u_pred):
        # u_pred is [Fx, Fy, Fz, Tx, Ty, Tz]] in FLU frame
        # The PX4 uses normalized wrench input. Scaling with respect to the maximum force and torque.
        F_scaling = 1/(2 * 1.4)
        T_scaling = 1/(4 * 0.12 * 1.4)
        F = F_scaling * u_pred[0, :3]
        T = T_scaling * u_pred[0, 3:6]
        # Apply deadband to avoid sending very small commands to the solenoids
        F[np.abs(F) < INPUT_DEADBAND] = 0.0
        T[np.abs(T) < INPUT_DEADBAND] = 0.0

        timestamp = int(self.get_clock().now().nanoseconds / 1000)

        thrust_outputs_msg = VehicleThrustSetpoint()
        thrust_outputs_msg.timestamp = timestamp

        torque_outputs_msg = VehicleTorqueSetpoint()
        torque_outputs_msg.timestamp = timestamp

        if self.use_ned:
            # FLU -> FRD transformation
            thrust_outputs_msg.xyz = [F[0], -F[1], -F[2]]
            torque_outputs_msg.xyz = [T[0], -T[1], -T[2]]
        else:
            thrust_outputs_msg.xyz = [F[0], F[1], F[2]]
            torque_outputs_msg.xyz = [T[0], T[1], T[2]]

        self.publisher_thrust_setpoint.publish(thrust_outputs_msg)
        self.publisher_torque_setpoint.publish(torque_outputs_msg)

    def publish_direct_actuator_setpoint(self, u_pred):
        actuator_outputs_msg = ActuatorMotors()
        actuator_outputs_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # Normalize thrust values w.r.t. max thrust
        thrust = u_pred[0, :] / self.model.max_thrust

        # Apply deadband
        thrust[np.abs(thrust) < INPUT_DEADBAND] = 0.0

        # Generate actuator outputs dynamically
        thrust_command = []
        for t in thrust:
            thrust_command.extend([max(t, 0.0), max(-t, 0.0)])
        thrust_command = np.clip(np.array(thrust_command, dtype=np.float32), 0.0, 1.0)

        actuator_outputs_msg.control[:len(thrust_command)] = thrust_command
        self.publisher_direct_actuator.publish(actuator_outputs_msg)

    def publish_propeller_setpoint(self, u_pred):
        propeller_outputs_msg = Float32MultiArray()
        thrust_command = u_pred[0, :]
        thrust_command = np.clip(np.array(thrust_command, dtype=np.float32), self.model.min_thrust, self.model.max_thrust)
        propeller_outputs_msg.data = thrust_command.tolist()
        self.publisher_propeller_setpoint.publish(propeller_outputs_msg)

    def publish_disturbance_estimate(self, d_hat):
        disturbance_msg = Vector3Stamped()
        disturbance_msg.header.stamp = self.get_clock().now().to_msg()
        disturbance_msg.vector.x = d_hat[0]
        disturbance_msg.vector.y = d_hat[1]
        disturbance_msg.vector.z = d_hat[2]
        self.disturbance_translation_pub.publish(disturbance_msg)

        disturbance_msg = Vector3Stamped()
        disturbance_msg.header.stamp = self.get_clock().now().to_msg()
        disturbance_msg.vector.x = d_hat[3]
        disturbance_msg.vector.y = d_hat[4]
        disturbance_msg.vector.z = d_hat[5]
        self.disturbance_rotation_pub.publish(disturbance_msg)

    def publish_sitl_odometry(self):
        msg = Odometry()
        msg.header.frame_id = "mocap"
        msg.child_frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = self.vehicle_position[0]
        msg.pose.pose.position.y = self.vehicle_position[1]
        msg.pose.pose.position.z = self.vehicle_position[2]
        msg.pose.pose.orientation.w = self.vehicle_attitude[0]
        msg.pose.pose.orientation.x = self.vehicle_attitude[1]
        msg.pose.pose.orientation.y = self.vehicle_attitude[2]
        msg.pose.pose.orientation.z = self.vehicle_attitude[3]
        msg.twist.twist.linear.x = self.vehicle_velocity[0]
        msg.twist.twist.linear.y = self.vehicle_velocity[1]
        msg.twist.twist.linear.z = self.vehicle_velocity[2]
        msg.twist.twist.angular.x = self.vehicle_angular_velocity[0]
        msg.twist.twist.angular.y = self.vehicle_angular_velocity[1]
        msg.twist.twist.angular.z = self.vehicle_angular_velocity[2]
        self.odom_pub.publish(msg)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "mocap"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = self.vehicle_position[0]
        pose_msg.pose.position.y = self.vehicle_position[1]
        pose_msg.pose.position.z = self.vehicle_position[2]
        pose_msg.pose.orientation.w = self.vehicle_attitude[0]
        pose_msg.pose.orientation.x = self.vehicle_attitude[1]
        pose_msg.pose.orientation.y = self.vehicle_attitude[2]
        pose_msg.pose.orientation.z = self.vehicle_attitude[3]
        self.sitl_pose_pub.publish(pose_msg)
        return

    def check_data_validity(self):
        ret_val = True
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Check if the data is valid based on the timestamps
        if (current_time - self.vehicle_odometry_timestamp > DATA_VALIDITY_STREAM):
            self.get_logger().warn("Vehicle odometry data is too old. Skipping offboard control...", throttle_duration_sec=1.0)
            ret_val = False
        if (current_time - self.vehicle_status_timestamp > DATA_VALIDITY_STATUS):
            self.get_logger().warn("Vehicle status data is too old. Skipping offboard control...", throttle_duration_sec=1.0)
            ret_val = False
        return ret_val

    def cmdloop_callback(self):

        # Publish odometry for SITL
        if self.sitl:
            self.publish_sitl_odometry()

        # Check data validity
        if not self.check_data_validity():
            return

        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.direct_actuator = False
        if self.mode == 'rate':
            offboard_msg.body_rate = True
        elif self.mode == 'direct_allocation' or self.mode == 'propeller':
            offboard_msg.direct_actuator = True
        elif self.mode == 'wrench' or self.mode == 'offset_free_wrench' or self.mode == 'lqr_wrench':
            offboard_msg.thrust_and_torque = True
        self.publisher_offboard_mode.publish(offboard_msg)

        # Set state and references for each MPC
        if self.mode == 'rate':
            x0 = np.array([self.vehicle_position[0],
                           self.vehicle_position[1],
                           self.vehicle_position[2],
                           self.vehicle_velocity[0],
                           self.vehicle_velocity[1],
                           self.vehicle_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3]]).reshape(10, 1)
            ref = np.concatenate((self.setpoint_position,       # position
                                  np.zeros(3),                  # velocity
                                  self.setpoint_attitude,       # attitude
                                  np.zeros(6)), axis=0)         # inputs reference (F, w)
            ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)
        elif self.mode == 'wrench' or self.mode == 'offset_free_wrench':
            x0 = np.array([self.vehicle_position[0],
                           self.vehicle_position[1],
                           self.vehicle_position[2],
                           self.vehicle_velocity[0],
                           self.vehicle_velocity[1],
                           self.vehicle_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3],
                           self.vehicle_angular_velocity[0],
                           self.vehicle_angular_velocity[1],
                           self.vehicle_angular_velocity[2]]).reshape(13, 1)
            ref = np.concatenate((self.setpoint_position,       # position
                                  np.zeros(3),                  # velocity
                                  self.setpoint_attitude,       # attitude
                                  np.zeros(3),                  # angular velocity
                                  np.zeros(6)), axis=0)         # inputs reference (F, torque)
            ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)
        elif self.mode == 'lqr_wrench':
            x0 = np.array([self.vehicle_position[0],
                           self.vehicle_position[1],
                           self.vehicle_position[2],
                           self.vehicle_velocity[0],
                           self.vehicle_velocity[1],
                           self.vehicle_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3],
                           self.vehicle_angular_velocity[0],
                           self.vehicle_angular_velocity[1],
                           self.vehicle_angular_velocity[2]]).reshape(13, 1)
            ref = np.concatenate((self.setpoint_position,       # position
                                  np.zeros(3),                  # velocity
                                  self.setpoint_attitude[0:],       # attitude
                                  np.zeros(3)), axis=0)         # angular velocity
        elif self.mode == 'direct_allocation' or self.mode == 'propeller':
            x0 = np.array([self.vehicle_position[0],
                           self.vehicle_position[1],
                           self.vehicle_position[2],
                           self.vehicle_velocity[0],
                           self.vehicle_velocity[1],
                           self.vehicle_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3],
                           self.vehicle_angular_velocity[0],
                           self.vehicle_angular_velocity[1],
                           self.vehicle_angular_velocity[2]]).reshape(13, 1)
            ref = np.concatenate((self.setpoint_position,       # position
                                  np.zeros(3),                  # velocity
                                  self.setpoint_attitude,       # attitude
                                  np.zeros(3),                  # angular velocity
                                  np.zeros(4)), axis=0)         # inputs reference (u1, ..., u4) for 2D platform
            ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        # Solve MPC
        u_pred, x_pred = self.mpc.solve(x0, ref=ref)

        if self.mode == 'offset_free_wrench':
            # Publish disturbance
            self.publish_disturbance_estimate(self.mpc.get_disturbance_estimate())

        # Colect data
        idx = 0
        predicted_path_msg = Path()
        for predicted_state in x_pred:
            idx = idx + 1
            # Publish time history of the vehicle path
            predicted_pose_msg = self.vector2PoseMsg('map', predicted_state[0:3], self.setpoint_attitude)
            predicted_path_msg.header = predicted_pose_msg.header
            predicted_path_msg.poses.append(predicted_pose_msg)
        self.predicted_path_pub.publish(predicted_path_msg)
        self.publish_reference(self.reference_pub, self.setpoint_position)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if self.mode == 'rate':
                self.publish_rate_setpoint(u_pred)
            elif self.mode == 'direct_allocation' or self.mode == 'direct_allocation_trajectory':
                self.publish_direct_actuator_setpoint(u_pred)
            elif self.mode == 'wrench' or self.mode == 'offset_free_wrench' or self.mode == 'lqr_wrench':
                self.publish_wrench_setpoint(u_pred)
            elif self.mode == 'propeller':
                self.publish_propeller_setpoint(u_pred)

    def add_set_pose_callback(self, request, response):
        self.setpoint_position[0] = request.pose.position.x
        self.setpoint_position[1] = request.pose.position.y
        self.setpoint_position[2] = request.pose.position.z
        self.setpoint_attitude[0] = request.pose.orientation.w
        self.setpoint_attitude[1] = request.pose.orientation.x
        self.setpoint_attitude[2] = request.pose.orientation.y
        self.setpoint_attitude[3] = request.pose.orientation.z
        return response

    def get_setpoint_pose_callback(self, msg):
        self.setpoint_position[0] = msg.pose.position.x
        self.setpoint_position[1] = msg.pose.position.y
        self.setpoint_position[2] = msg.pose.position.z
        self.setpoint_attitude[0] = msg.pose.orientation.w
        self.setpoint_attitude[1] = msg.pose.orientation.x
        self.setpoint_attitude[2] = msg.pose.orientation.y
        self.setpoint_attitude[3] = msg.pose.orientation.z

    def vector2PoseMsg(self, frame_id, position, attitude):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.orientation.w = attitude[0]
        pose_msg.pose.orientation.x = attitude[1]
        pose_msg.pose.orientation.y = attitude[2]
        pose_msg.pose.orientation.z = attitude[3]
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        return pose_msg


def main(args=None):
    rclpy.init(args=args)
    spacecraft_mpc = SpacecraftMPC()
    rclpy.spin(spacecraft_mpc)
    spacecraft_mpc.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
