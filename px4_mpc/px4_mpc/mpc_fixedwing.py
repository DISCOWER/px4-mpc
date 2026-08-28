#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import PointStamped # We will use Point xyz fields to hold 3 variables
from std_msgs.msg import Float64MultiArray


from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleRatesSetpoint

from px4_mpc.models.fixedwing_model import FixedwingModel
from px4_mpc.controllers.fixedwing_mpc import FixedwingMPC
from px4_mpc.safety_filters import CBFSafetyFilter

class FixedwingMPCNode(Node):
    def __init__(self):
        super().__init__('fw_mpc_publisher')
        
        qos_profile_pub = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, 
                                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, 
                                history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        qos_profile_sub = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, 
                                durability=QoSDurabilityPolicy.VOLATILE, 
                                history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        # will have to check the topic versions
        self.status_sub = self.create_subscription(VehicleStatus, 
            '/fmu/out/vehicle_status_v4', self.vehicle_status_callback, qos_profile_sub)
        self.attitude_sub = self.create_subscription(VehicleAttitude, 
            '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile_sub)
        self.local_position_sub = self.create_subscription(VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1', self.vehicle_local_position_callback, qos_profile_sub)

        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, 
            '/fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_rates_setpoint = self.create_publisher(VehicleRatesSetpoint, 
            '/fmu/in/vehicle_rates_setpoint', qos_profile_pub)
        
        self.optimal_path_raw_pub = self.create_publisher(Float64MultiArray,"mpc/traj_optimal_raw",1)
        self.error_pub = self.create_publisher(PointStamped, "/mpc/debug/tracking_error_xyz", 1)
        self.effort_pub = self.create_publisher(PointStamped, "/mpc/debug/control_effort", 1)
        self.speed_pub = self.create_publisher(PointStamped, "/mpc/debug/speeds", 1)
        
        
        # Path following parameters from pytorch implementation
        self.path_center = np.array([30.0, 15.0, 30.0]) # ENU: Loitering at 20m altitude
        self.path_radius = 50.0
        self.target_speed = 10.0
        self.speed_min = 0.1

        # initialize objs
        self.model = FixedwingModel()
        self.mpc = FixedwingMPC(self.model, N=40, Tf=2.0,trackingAttitude=False)
        self.safetyfilter= CBFSafetyFilter(v_min = 12.0, lambda_cbf=2.0, t_max=self.model.max_thrust_acc,t_min=0.0)
        self.dt = self.mpc.Tf / self.mpc.N
        
        # preallocate buffers
        self.model = FixedwingModel()
        self.nx = self.model.get_acados_model().x.size()[0]
        
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        
        # system state
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.timer = self.create_timer(0.02, self.cmdloop_callback) # 50Hz Loop

        # check if the vehicle has received valid state messages
        self.received_pos = False
        self.received_attitude=False
        self.mpc_initialized = False

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
    
    def vehicle_attitude_callback(self, msg):
        """update local attitude from msg. NED (qw,qx,qy,qz) -> ENU"""
        q_enu = 1/np.sqrt(2) * np.array([msg.q[0] + msg.q[3], msg.q[1] + msg.q[2], msg.q[1] - msg.q[2], msg.q[0] - msg.q[3]])
        self.vehicle_attitude[:] = q_enu / np.linalg.norm(q_enu) # normalize and assign in-place as float
        self.received_attitude=True
        
    def vehicle_local_position_callback(self, msg):
        """update local state from msg. NED → ENU (In-place assignment)"""
        self.vehicle_local_position[:] = [msg.y, msg.x, -msg.z]
        self.vehicle_local_velocity[:] = [msg.vy, msg.vx, -msg.vz]
        self.received_pos=True

    def compute_vehicle_cmd(self,u_pred,speed):
        """compute and return control outputs to PX4"""
        thrust_cmd_raw, lift_cmd, roll_rate_cmd = u_pred[0]
        # extract pitch & yaw rates from gravity vector in body frame
        # quat: (w x y z), ENU
        R_mat = self.quat_to_rot_matrix(self.vehicle_attitude)
        g_enu = np.array([0.0, 0.0, -9.81])
        g_body = R_mat.T @ g_enu
        
        speed_safe = max(speed, self.speed_min) # div0
        pitch_rate_cmd = -(lift_cmd + g_body[2]) / speed_safe 
        yaw_rate_cmd   = (g_body[1]) / speed_safe

        setpoint_msg = VehicleRatesSetpoint()
        setpoint_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        setpoint_msg.roll  = float(roll_rate_cmd)
        setpoint_msg.pitch = float(-pitch_rate_cmd) 
        setpoint_msg.yaw   = float(-yaw_rate_cmd)
        
        # just find what PX4 expects for thrust and scale it down to [0,1]
        normalized_throttle = max(0.0, min(thrust_cmd_raw/1.079,1.0))
        print(f"cmd rates: thrust: {normalized_throttle:.3f} | speed: {speed:.2f}\n| roll rate: {roll_rate_cmd:.2f}| pitch rate:{pitch_rate_cmd:.2f} | yaw rate: {yaw_rate_cmd:.2f}")
        setpoint_msg.thrust_body[:] = [float(normalized_throttle), 0.0, 0.0]
        
        return setpoint_msg

    # ---------- main loop ----------- 
    def cmdloop_callback(self):
            # Maintain Offboard Heartbeat
            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
            offboard_msg.position = False
            offboard_msg.velocity = False
            offboard_msg.acceleration = False
            offboard_msg.attitude = False
            offboard_msg.body_rate = True
            self.publisher_offboard_mode.publish(offboard_msg)
            
            # skip the rest if we cant get the state initialized
            if not (self.received_pos and self.received_attitude):
                return
            # update solver to initialize at current state
            speed = np.linalg.norm(self.vehicle_local_velocity)
            self.speed_min = 0.10 if speed > 0.5 else 0.0
            x0= np.concatenate((self.vehicle_local_position, [max(speed, self.speed_min)], self.vehicle_attitude))
            
            # runs in first iteration of cmd loop OR restarting mpc
            # TODO? set mpc_initialized = false when we stop the node (and plan to allow restart?)
            if not self.mpc_initialized:
                self.get_logger().info(f"Initializing Acados Solver at Speed: {x0[3]:.1f} m/s")
                self.mpc = FixedwingMPC(self.model, x0, N=40, Tf=2.0)
                self.dt = self.mpc.Tf / self.mpc.N
                self.mpc_initialized = True
    
            yref_traj = self.generate_reference_trajectory()
            x_pred=None
            u_pred=None
            # extract controls and publish if offboard mode on
            if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:   
                # generate and solve over horizon
                u_pred, x_pred = self.mpc.solve(x0, yref_traj)
                # u_safe = self.safetyfilter.filter_airspeed(u_pred,x0[3],x0[4:8])
                # calculate and publish control outputs
                setpoint_msg = self.compute_vehicle_cmd(u_pred,speed)
                self.publisher_rates_setpoint.publish(setpoint_msg)
            
            # publish visualizations
            # self.publish_visuals(x_pred)
            self.publish_diagnostics(x_pred,u_pred, yref_traj,speed)

    def publish_diagnostics(self, x_pred, u_pred, yref_traj, speed):
        stamp = self.get_clock().now().to_msg()
        
        # 1. Publish Optimal Trajectory Array (For fw_viz.py)
        if x_pred is not None:
            array_msg = Float64MultiArray()
            array_msg.data = x_pred.flatten().tolist()
            self.optimal_path_raw_pub.publish(array_msg)
            
        # 2. Publish Tracking Error (Actual Position - Target Reference Point)
        # Using PointStamped simply as a container for 3 variables (X, Y, Z error)
        err_msg = PointStamped()
        err_msg.header.stamp = stamp
        err_msg.point.x = float(self.vehicle_local_position[0] - yref_traj[0, 0])
        err_msg.point.y = float(self.vehicle_local_position[1] - yref_traj[0, 1])
        err_msg.point.z = float(self.vehicle_local_position[2] - yref_traj[0, 2])
        self.error_pub.publish(err_msg)

        # 3. Publish Control Effort (Thrust, Lift, Roll Rate)
        if u_pred is not None:
            effort_msg = PointStamped()
            effort_msg.header.stamp = stamp
            effort_msg.point.x = float(u_pred[0][0]) # Thrust
            effort_msg.point.y = float(u_pred[0][1]) # Lift
            effort_msg.point.z = float(u_pred[0][2]) # Roll rate
            self.effort_pub.publish(effort_msg)

        # 4. Publish Speeds (Actual vs Target)
        speed_msg = PointStamped()
        speed_msg.header.stamp = stamp
        speed_msg.point.x = float(speed)
        speed_msg.point.y = float(self.target_speed)
        speed_msg.point.z = 0.0 # Unused
        self.speed_pub.publish(speed_msg)

    def generate_reference_trajectory(self):
        """Generates a sliding reference for CIRCULAR path"""
        yref_traj = np.zeros((self.mpc.N + 1, self.nx))
        
        # closest angle on circle rn
        dx = self.vehicle_local_position[0] - self.path_center[0]
        dy = self.vehicle_local_position[1] - self.path_center[1]
        closest_angle = np.arctan2(dy, dx)
        
        # projection based on actual speed
        current_speed = np.linalg.norm(self.vehicle_local_velocity)
        safe_speed = max(current_speed, 1.0) # Prevent division by zero if stationary
        actual_omega = safe_speed / self.path_radius 
        
        g = 9.81
        # ENU; (+) omega = left turn, requires negative roll
        bank_angle = np.arctan2(-self.target_speed*actual_omega, g)
        
        for i in range(self.mpc.N + 1):
            # The reference points now march forward at the exact speed the drone is flying
            theta = closest_angle + actual_omega * (i * self.dt)
            
            yref_traj[i, 0] = self.path_center[0] + self.path_radius * np.cos(theta)
            yref_traj[i, 1] = self.path_center[1] + self.path_radius * np.sin(theta)
            yref_traj[i, 2] = self.path_center[2]
            yref_traj[i, 3] = self.target_speed # Still tell it we WANT to fly at 15 m/s
            
            # tangent to heading
            psi = theta + (np.pi / 2.0)
            
            # quaternions for if i am weighting attitude too
            cy = np.cos(psi * 0.5)
            sy = np.sin(psi * 0.5)
            cp = np.cos(0.0) 
            sp = np.sin(0.0)
            cr = np.cos(bank_angle * 0.5)
            sr = np.sin(bank_angle * 0.5)
            
            yref_traj[i, 4] = cr * cp * cy + sr * sp * sy
            yref_traj[i, 5] = sr * cp * cy - cr * sp * sy
            yref_traj[i, 6] = cr * sp * cy + sr * cp * sy
            yref_traj[i, 7] = cr * cp * sy - sr * sp * cy
        
        return yref_traj
    # Helper functions for quaternion operations
    def quat_multiply(self, q1, q2):
        """multiply two wxyz quaternions"""
        w1, x1, y1, z1 = q1 # unpack sequence
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.array([w, x, y, z])

    def quat_to_rot_matrix(self, q):
        """return SO(3) rot matrix from wxyz quaternion """
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

def main(args=None):
    rclpy.init(args=args)
    fixedwing_mpc_node = FixedwingMPCNode()
    rclpy.spin(fixedwing_mpc_node)
    fixedwing_mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
