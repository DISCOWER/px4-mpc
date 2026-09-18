#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float64MultiArray, Int32, Float32

from px4_msgs.msg import VehicleStatus, VehicleAttitude, VehicleLocalPosition
from px4_msgs.msg import VehicleRatesSetpoint, TrajectorySetpoint
from px4_msgs.msg import SensorCombined

from px4_mpc.models.fixedwing_model import FixedWingModel
from px4_mpc.controllers.fixedwing_mpc import FixedWingMPC
from px4_mpc.safety_filters import CBFSafetyFilter, CompositeCBFSafetyFilter

class FixedWingMPCNode(Node):
    def __init__(self):
        super().__init__('fw_mpc_publisher')

        # qos profiles for px4 dds bridge
        qos_pub = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        qos_sub = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        qos_latched = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        # px4 telemetry subscribers
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_sub)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_sub)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_sub)
        self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self.imu_callback, qos_sub)
        # control publishers
        self.pub_traj_setpoint = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_pub)
        self.pub_rates_setpoint = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_pub)

        # path viz
        self.global_ref_path_pub = self.create_publisher(Path, '/px4_mpc/global_reference_path', qos_latched)
        
        # diagnostics, QoS is best effort as these are only for visualization and debugging
        self.optimal_path_raw_pub = self.create_publisher(Float64MultiArray, '/mpc/traj_optimal_raw', 1)
        self.plane_actual_pos_pub = self.create_publisher(PointStamped, '/mpc/state/plane_actual_pos', 1)
        self.solver_state_pos_pub = self.create_publisher(PointStamped, '/mpc/state/solver_x0_pos', 1)
        self.error_pub = self.create_publisher(PointStamped, '/mpc/debug/tracking_error_xyz', 1)
        self.effort_pub = self.create_publisher(PointStamped, '/mpc/debug/control_effort', 1)
        
        # solver statuses
        self.solver_status_pub = self.create_publisher(Int32, '/mpc/debug/solver_diagnostics/solver_status', 1)
        self.solver_fail_pub = self.create_publisher(Int32, '/mpc/debug/solver_diagnostics/solver_fail', 1)
        self.detect_offboard_pub = self.create_publisher(Int32, '/mpc/debug/solver_diagnostics/detect_offboard', 1)
        
        # cbf statuses
        self.cbf_tripped_pub = self.create_publisher(Int32, '/mpc/debug/cbf/tripped', 1)
        self.cbf_penalty_pub = self.create_publisher(Float32, '/mpc/debug/cbf/penalty', 1)
        self.cbf_status_pub = self.create_publisher(Int32, '/mpc/debug/cbf/solver_status', 1)
        
        self.f_w_flu_desired_pub = self.create_publisher(PointStamped, '/mpc/debug/f_w_flu_desired', 1)
        self.f_w_flu_achieved_pub = self.create_publisher(PointStamped, '/mpc/debug/f_w_flu_achieved', 1)

        # mpc constraints and targets
        self.Ts = 0.05
        self.N_horizon = 40
        self.v_min = 10.0
        self.z_min = 20.0
        includeCBF = True
        self.warmstart = False # technically the acados solver also has a warm start inside
        # safety parameters
        self.N_repair = 30
        self.filter_mode ="HOCBF" # "HOCBF" or "composite" or None
        self.filter_params_hocbf = {
            "filter_mode": "HOCBF", # "FIRST_ORDER", "HOCBF", "THIRD_ORDER"
            # THIRD_ORDER is experimental, not fully fleshed out
            "solver_mode": "custom", # acados or (hocbf only) "custom"
            "repair_horizon": self.N_repair,
            "vmin": self.v_min,
            "gamma2": 1.0,
            "gamma1": 1.0,
            "beta": 8.5, # only used for custom
            "max_iters": 8, # only used for custom
            "CT_or_DT": "DT",
            "gamma3": 1.0, # only used for 3rd order
        }
        self.filter_params_composite = {
            "solver_mode": "acados",
            "vmin": self.v_min,
            "zmin": self.z_min,
            "repair_horizon": self.N_repair,
            "gamma": 90.0,
            "p0": -6.0,
            "kappa": 5.0,
            "alpha": 0.4,
            "beta": 5.0,
            "max_iters": 10
        }
        
        self.target_speed = 14.0
        # PATH ARRAY
        self.path_idx = 0
        self.path_center = np.array([0.0, 0.0, 35.0])
        self.path_radius = 40.0
        # precompute a dense array of the circle for the orthogonal projection to snap to
        theta_array = np.linspace(0.0, 2.0 * np.pi, 2000)
        self.global_path_array = np.zeros((2000, 3))
        angle_incline = np.radians(50.0) #np.pi/5 # or do np.radians(incline deg)
        
        R_incline = np.array([
            [np.cos(angle_incline),  0.0, np.sin(angle_incline)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle_incline), 0.0, np.cos(angle_incline)]
        ])
        # trajectory shapes
        shape_circ = [self.path_radius * np.cos(theta_array), self.path_radius * np.sin(theta_array),np.zeros(2000)]
        
        rect_L = 100.0; rect_W = 40.0
        p = 4.0  # the "roundness" knob. 2.0 = exact ellipse, 4.0 = rounded rectangle, 15.0+ = razor sharp
        rect_r = 1.0 / ((np.abs(np.cos(theta_array)) / rect_L)**p + (np.abs(np.sin(theta_array)) / rect_W)**p)**(1.0 / p)
        shape_rect = [rect_r * np.cos(theta_array), rect_r * np.sin(theta_array), np.zeros(2000)]
        
        # apply rotation for incline then move its center to desired location
        local_shape = R_incline @ shape_circ
        
        self.global_path_array[:, 0] = self.path_center[0] + local_shape[0,:]
        self.global_path_array[:, 1] = self.path_center[1] + local_shape[1,:] 
        self.global_path_array[:, 2] = self.path_center[2] + local_shape[2,:]
        
        self.model = FixedWingModel()
        self.nx = 8
        self.nu = 3
        
        self.cbf_filter = CBFSafetyFilter(self.model, N=self.N_horizon, Ts=self.Ts, 
                **self.filter_params_hocbf) if self.filter_mode == "HOCBF" else \
            CompositeCBFSafetyFilter(self.model, N=self.N_horizon, Ts=self.Ts,
                **self.filter_params_composite) if self.filter_mode == "composite" else None
        x0_init=np.zeros(self.nx)
        x0_init[4]=1.0
        self.get_logger().info("initializing mpc solver for indi position tracking")
        self.mpc = FixedWingMPC( self.model, N=self.N_horizon, Ts=self.Ts, x0_init=x0_init,
            cbf_filter=self.cbf_filter if includeCBF else None,
            trackingAttitude=False)
        
        # vehicle state in enu
        self.pos_enu = np.zeros(3)
        self.vel_enu = np.zeros(3)
        self.acc_enu = np.zeros(3)
        self.q_enu = np.array([1.0, 0.0, 0.0, 0.0])
        
        # # precomputed matrices to save loop time
        # self.R_flu_to_enu = np.eye(3) # only used in velocity anyway
        # self.R_enu_to_flu = np.eye(3)

        self.x_safe_guess = None #np.zeros((self.N_horizon + 1, self.nx))
        # self.x_safe_guess[:,4]=1.0
        self.u_safe_guess = np.zeros((self.N_horizon, self.nu))
        
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.received_pos = False
        self.received_att = False
        self.sim_step = 0 # just for the visualization updates
        self.last_valid_x_pred = None
        self.last_px4_timestamp = 0
        
        self._publish_global_orbit_path()
        self.timer = self.create_timer(self.Ts, self.control_loop)

    # px4 to mpc and indi frame
    def _ned_to_enu(self, v):
        return np.array([v[1], v[0], -v[2]])

    # from indi frame back to px4 frame
    def _enu_to_ned(self, v):
        return np.array([v[1], v[0], -v[2]])

    # return normalized quaternion product of two quaternions
    def _quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        q_prod = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        n = np.linalg.norm(q_prod)
        return q_prod / (n+1e-6)

    def _quat_to_rotmat(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
            [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
            [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)]
        ])

    def _correct_ekf_yaw_drift(self, q_enu, vel_enu, speed):
        """
        PX4 simulation's attitude/orientation sometimes accumulates a severe drift in yaw.
        correct by:
        1. extracting roll and pitch
        2. use XY velocity vector to get geometric heading to replace yaw
        """
        # replaces drifting ekf yaw with kinematic course over ground
        w, x, y, z = q_enu
        # extract roll and pitch from drifting quaternion
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        
        if speed > 2.0:
            yaw = np.arctan2(vel_enu[1], vel_enu[0])
        else:
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

        # individual rot quaternions for roll, pitch, yaw
        q_yaw   = np.array([np.cos(yaw   * 0.5), 0.0,                 0.0, np.sin(yaw * 0.5)])
        q_pitch = np.array([np.cos(pitch * 0.5), 0.0, np.sin(pitch * 0.5), 0.0])
        q_roll  = np.array([np.cos(roll  * 0.5), np.sin(roll * 0.5),  0.0, 0.0])

        # recombine rpy rotation quats (Standard ZYX sequence: q = q_yaw * q_pitch * q_roll)
        # _quat_mul automatically normalizes
        q_pitch_roll = self._quat_mul(q_pitch, q_roll)
        q_new = self._quat_mul(q_yaw, q_pitch_roll)
        # enforce continuity and ensure same hemisphere as previous quaternion
        return -q_new if np.dot(self.q_enu, q_new) < 0.0 else q_new
    
    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.last_px4_timestamp = msg.timestamp
    
    def vehicle_attitude_callback(self, msg):
        """
        process px4 attitude; convert from NED ⇒ ENU 
        """
        q_raw_ned = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]]) 
        # unnormalized quat for ned to enu for indi controller
        # x_enu = y_ned, y_enu = x_ned, z_enu = -z_ned
        self.q_enu = (1.0 / np.sqrt(2.0)) * np.array([
            q_raw_ned[0] + q_raw_ned[3], # w_enu
            q_raw_ned[1] + q_raw_ned[2], # x_enu
            q_raw_ned[1] - q_raw_ned[2], # y_enu
            q_raw_ned[0] - q_raw_ned[3]  # z_enu
        ])
        # # precompute matrices for use in control loop
        # self.R_flu_to_enu = self._quat_to_rotmat(self.q_enu)
        # self.R_enu_to_flu = self.R_flu_to_enu.T
        self.received_att = True
        self.last_px4_timestamp = msg.timestamp
    
    def vehicle_local_position_callback(self, msg):
        if not (np.isfinite(msg.x) and np.isfinite(msg.y) and np.isfinite(msg.z)):
            return
            
        self.pos_enu = self._ned_to_enu([msg.x, msg.y, msg.z])
        self.vel_enu = self._ned_to_enu([msg.vx, msg.vy, msg.vz])
        self.current_speed = np.linalg.norm(self.vel_enu)
        # Alternative: project velocity onto thrust axis 
        # self.current_speed = np.dot(self.vel_enu, self.R_flu_to_enu[:,0])
        
        if hasattr(msg, 'ax') and np.isfinite(msg.ax):
            self.acc_enu = self._ned_to_enu([msg.ax, msg.ay, msg.az])
            
        self.received_pos = True
        self.last_px4_timestamp = msg.timestamp
    
    def imu_callback(self,msg):
        """
        Just for comparison, not used in control:
        Subscribe to NED accelerometer and publish in ENU
        """
        achieved_msg = PointStamped()
        achieved_msg.header.stamp = self.get_clock().now().to_msg()
        achieved_msg.header.frame_id = 'base_link'
        
        # Map from PX4 NED to MPC FLU (Forward-Left-Up)
        achieved_msg.point.x = float(msg.accelerometer_m_s2[0])
        achieved_msg.point.y = float(-msg.accelerometer_m_s2[1])
        achieved_msg.point.z = float(-msg.accelerometer_m_s2[2])
        
        self.f_w_flu_achieved_pub.publish(achieved_msg)
    
    # for visualization
    def _publish_global_orbit_path(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        # slice the precomputed array, stepping by 10 so we only send 100 points to rviz
        for pt in self.global_path_array[::10]:
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x = pt[0]
            p.pose.position.y = pt[1]
            p.pose.position.z = pt[2]
            p.pose.orientation.w = 1.0
            path_msg.poses.append(p)
            
        self.global_ref_path_pub.publish(path_msg)

    def generate_reference_trajectory(self, global_path_array):
        """
        Generate reference path for MPC solver starting from nearest global waypoint.
        Points are paced based on airspeed
        """
        yref = np.zeros((self.N_horizon + 1, self.nx))
        # lookahead window
        search_window = 300
        n_points = len(global_path_array)
        # grab subset of indices ahead of last position
        idx_array = (self.path_idx + np.arange(search_window)) % n_points
        window_pts = global_path_array[idx_array]
        # find how far plane is from every point in local window
        dists = np.linalg.norm(window_pts - self.pos_enu, axis=1)
        # lock closest point and advance fwd
        self.path_idx = idx_array[np.argmin(dists)]
        curr_idx = self.path_idx
        # pacing btwn points for the reference
        track_pace = max(self.current_speed, self.v_min)
        step_dist = track_pace * self.Ts
        
        yref[0, 0:3] = global_path_array[curr_idx]
        yref[0, 4] = 1.0
        
        search_idx = curr_idx
        # iterate through path array
        for i in range(1, self.N_horizon + 1):
            accumulated = 0.0
            # sample points, spaced out by track pace
            while accumulated < step_dist:
                next_idx = (search_idx + 1) % n_points
                pt1 = global_path_array[search_idx]
                pt2 = global_path_array[next_idx]
                accumulated += np.linalg.norm(pt2 - pt1)
                search_idx = next_idx
            
            yref[i, 0:3] = global_path_array[search_idx]
            yref[i, 4] = 1.0 
        
        return yref

    def control_loop(self):
        if not (self.received_pos and self.received_att):
            return

        timestamp = int(self.last_px4_timestamp) if self.last_px4_timestamp > 0 else int(Clock().now().nanoseconds / 1000)
        
        # q_corrected = self._correct_ekf_yaw_drift(self.q_enu, self.vel_enu, self.current_speed)
        x0 = np.concatenate([self.pos_enu, [self.current_speed], self.q_enu])

        yref = self.generate_reference_trajectory(self.global_path_array)
        
        # dynamic cold start if the arrays are empty or need a hard reset
        
        if self.warmstart and self.x_safe_guess is not None:
            u_pred, x_pred, mpc_status = self.mpc.solve(x0, yref,
                                    x_warm_start=self.x_safe_guess, u_warm_start=self.u_safe_guess)
        else:
            u_pred, x_pred, mpc_status = self.mpc.solve(x0, yref)

        is_x_pred_valid = (x_pred is not None) and np.all(np.isfinite(x_pred))
        is_u_pred_valid = (u_pred is not None) and np.all(np.isfinite(u_pred))
        solver_failed = (mpc_status != 0) or (not is_x_pred_valid) or (not is_u_pred_valid)

        # grab the prediction for visualization even if status is bad as long as it isnt nans
        if is_x_pred_valid:
            viz_x_pred = x_pred.copy()
            ctrl_x_pred = x_pred.copy()
        elif self.last_valid_x_pred is not None:
            viz_x_pred = self.last_valid_x_pred.copy()
            ctrl_x_pred = self.last_valid_x_pred.copy()
        else: # fallback that only runs when we have no previous valid MPC sol + x_pred isn't valid
            # this yref won't get any further than visuals, as conds also hit a return before ctrl pubs
            viz_x_pred = yref[:, :self.nx].copy()
            ctrl_x_pred = yref[:, :self.nx].copy()
        # default diagnostic variables for the cbf
        shield_tripped = False
        penalty = 0.0
        cbf_status = 0

        if not solver_failed:
            if self.cbf_filter is not None:
                u_filtered, shield_tripped, penalty, diag = self.cbf_filter.filter(x0, u_pred, self.sim_step)
                u_act = u_filtered[0, :]
                
                # unpack diagnostic dict if the cbf solver threw an error
                if diag is not None:
                    cbf_status = diag.get('status_code', -1)
                
                if shield_tripped: 
                    # update the predicted states for repaired horizon (k steps)
                    u_for_rollout = u_filtered[:self.cbf_filter.K,:].T
                    x_repaired_dm = np.array(self.cbf_filter._rollout_func(x0, u_for_rollout)).T
                    
                    # overwrite the mpc prediction with safe rollout (convert from casadi DM to np)
                    ctrl_x_pred[:self.cbf_filter.K + 1, :] = x_repaired_dm
                    
                    if self.sim_step % 10 == 0:
                        if cbf_status != 0:
                            self.get_logger().error(f"CBF shield fail: status {cbf_status}! Run fallback option")
                # warmstart regardless of shield
                if self.warmstart: # if warm starting: update safe guess
                    self.u_safe_guess[:-1, :] = u_filtered[1:, :]
                    self.u_safe_guess[-1, :] = u_filtered[-1, :]
            else:
                u_act = u_pred[0, :]
                if self.warmstart:
                    self.u_safe_guess[:-1, :] = u_pred[1:, :]
                    self.u_safe_guess[-1, :] = u_pred[-1, :]
        else:
            u_act = np.array([0.0, self.model.gravity, 0.0])
            if self.sim_step % 10 == 0:
                self.get_logger().error(f"MPC fail: status {mpc_status}. Run fallback option")
                
        if is_x_pred_valid:
            self.last_valid_x_pred = ctrl_x_pred.copy()
            if self.warmstart and not solver_failed:
                # shift the states for the warm start
                if self.x_safe_guess is None: # first time
                    self.x_safe_guess = np.zeros((self.N_horizon + 1, self.nx))
                self.x_safe_guess[:-1, :] = ctrl_x_pred[1:, :]
                self.x_safe_guess[-1, :] = ctrl_x_pred[-1, :]

        self.sim_step += 1 # just for the visualization updates
        
        is_offboard = (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)

        # publish complete diagnostic profile
        self.publish_diagnostics(viz_x_pred, u_act, yref, x0, mpc_status, solver_failed, is_offboard, shield_tripped, penalty, cbf_status)
        
        if self.sim_step % 10 == 0:
            self._publish_global_orbit_path()
        # skip publishing to indi controllers
        if not is_offboard or solver_failed:
            return

        self.publish_control_traj_setpoint(u_act, ctrl_x_pred, timestamp)
        self.publish_control_body(u_act, ctrl_x_pred, timestamp)

    def publish_control_traj_setpoint(self, u_act, x_pred, timestamp):
        """
        Publish kinematic trajectory setpoints: publish acceleration for low-level PX4 controller in NED frame
        Inputs: 
        - u_act = [f_xw, f_zw, roll_rate]: control input array
        - x_pred: planned state trajectory from MPC
        - timestamp: current timestamp synced with PX4
        
        Output: None
        - publish TrajectorySetpoint msg containing acceleration vector mapped to NED frame
        """
        fxw_cmd = np.clip(u_act[0], self.model.min_fxw, self.model.max_fxw)
        fzw_cmd = np.clip(u_act[1], self.model.min_fzw, self.model.max_fzw)

        q_target_enu = x_pred[1, 4:8]
        R_target = self._quat_to_rotmat(q_target_enu)
        
        # map specific force to enu frame
        f_w_body = np.array([fxw_cmd, 0.0, fzw_cmd])
        # kin accel = gravity vector + body forces rotated into ENU frame
        acc_kinematic_enu = R_target @ f_w_body + np.array([0.0, 0.0, -self.model.gravity])
        acc_ned = self._enu_to_ned(acc_kinematic_enu)
        
        # same for velocity if needed
        # speed_planned = max(x_pred[1, 3], self.v_min)
        # v_body_dir = np.array([1.0, 0.0, 0.0])
        # vel_ref_enu = speed_planned * (R_target @ v_body_dir)
        # vel_ned = self._enu_to_ned(vel_ref_enu)

        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = timestamp
        traj_msg.acceleration = acc_ned.tolist() # built-in conversion into list of floats
        traj_msg.velocity = [float('NaN'),float('NaN'),float('NaN')]
        # traj_msg.velocity = vel_ned.tolist()
        self.pub_traj_setpoint.publish(traj_msg)
        
    def publish_control_body(self, u_act, x_pred, timestamp):
        """
        Publish rate setpoints (roll, pitch, yaw rates) and normalized thrust in the body frame,
        for use by inner-loop controllers in PX4.
        
        Inputs:
        - u_act = [f_xw, f_zw, roll_rate]: control input array
        - x_pred: planned state trajectory from MPC
        - timestamp: current timestamp synced with PX4
        
        Output: None
        - publish VehicleRatesSetpoint msg containing, roll, pitch, yaw rates (rad/s) and normalized thrust cmd
        
        """
        fxw_cmd = np.clip(u_act[0], self.model.min_fxw, self.model.max_fxw)
        roll_rate_cmd = np.clip(u_act[2], -self.model.max_roll_rate, self.model.max_roll_rate)

        # extract forward finite difference of planned quaternions
        q0 = x_pred[0, 4:8]
        q1 = x_pred[1, 4:8]
        q0_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
        # rotational difference between current and planned state
        q_diff = self._quat_mul(q0_inv, q1)
        # rotate the shortest way possible
        if q_diff[0] < 0.0:
            q_diff = -q_diff
            
        # map angular velocity from enu rotation to body rates
        omega_enu = (2.0 / self.Ts) * q_diff[1:4]
        pitch_rate_cmd = -omega_enu[1]
        yaw_rate_cmd = -omega_enu[2]
        throttle_offset = 0.3
        throttle_cmd = np.clip( throttle_offset+ (1-throttle_offset)* fxw_cmd / self.model.max_fxw, 0.0, 1.0)
        
        rates_msg = VehicleRatesSetpoint()
        rates_msg.timestamp = timestamp
        rates_msg.roll = roll_rate_cmd
        rates_msg.pitch = pitch_rate_cmd
        rates_msg.yaw = yaw_rate_cmd
        rates_msg.thrust_body = [throttle_cmd, 0.0, 0.0]
        self.pub_rates_setpoint.publish(rates_msg)

    def publish_diagnostics(self, x_pred, u_act, yref, x0, mpc_status, solver_failed, is_offboard, shield_tripped, penalty, cbf_status):
        stamp = self.get_clock().now().to_msg()

        array_msg = Float64MultiArray()
        array_msg.data = x_pred.flatten().tolist()
        self.optimal_path_raw_pub.publish(array_msg)

        actual_pt = PointStamped()
        actual_pt.header.stamp = stamp
        actual_pt.header.frame_id = 'map'
        actual_pt.point.x = self.pos_enu[0]
        actual_pt.point.y = self.pos_enu[1]
        actual_pt.point.z = self.pos_enu[2]
        self.plane_actual_pos_pub.publish(actual_pt)

        solver_pt = PointStamped()
        solver_pt.header.stamp = stamp
        solver_pt.header.frame_id = 'map'
        solver_pt.point.x = x0[0]
        solver_pt.point.y = x0[1]
        solver_pt.point.z = x0[2]
        self.solver_state_pos_pub.publish(solver_pt)

        err_msg = PointStamped()
        err_msg.header.stamp = stamp
        err_msg.header.frame_id = 'map'
        err_msg.point.x = self.pos_enu[0] - yref[0, 0]
        err_msg.point.y = self.pos_enu[1] - yref[0, 1]
        err_msg.point.z = self.pos_enu[2] - yref[0, 2]
        self.error_pub.publish(err_msg)

        effort_msg = PointStamped()
        effort_msg.header.stamp = stamp
        effort_msg.header.frame_id = 'base_link'
        effort_msg.point.x = u_act[0]
        effort_msg.point.y = u_act[1]
        effort_msg.point.z = u_act[2]
        self.effort_pub.publish(effort_msg)

        status_msg = Int32()
        status_msg.data = int(mpc_status) if mpc_status is not None else -1
        self.solver_status_pub.publish(status_msg)

        fail_msg = Int32()
        fail_msg.data = 1 if solver_failed else 0
        self.solver_fail_pub.publish(fail_msg)

        offboard_msg = Int32()
        offboard_msg.data = 1 if is_offboard else 0
        self.detect_offboard_pub.publish(offboard_msg)
        
        # log the safety filter diagnostics
        tripped_msg = Int32()
        tripped_msg.data = 1 if shield_tripped else 0
        self.cbf_tripped_pub.publish(tripped_msg)
        
        pen_msg = Float32()
        pen_msg.data = penalty
        self.cbf_penalty_pub.publish(pen_msg)
        
        cbf_stat_msg = Int32()
        cbf_stat_msg.data = int(cbf_status)
        self.cbf_status_pub.publish(cbf_stat_msg)

        # desired wind forces directly from mpc
        desired_msg = PointStamped()
        desired_msg.header.stamp = stamp
        desired_msg.header.frame_id = 'base_link'
        desired_msg.point.x = u_act[0]
        desired_msg.point.y = 0.0             
        desired_msg.point.z = u_act[1]
        self.f_w_flu_desired_pub.publish(desired_msg)
        
def main(args=None):
    rclpy.init(args=args)
    node = FixedWingMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
