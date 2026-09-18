#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, TransformStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster

from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition

class FixedWingVizNode(Node):
    def __init__(self):
        super().__init__('fw_viz_node')

        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # px4 telemetry
        self.attitude_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.attitude_callback, qos_sub)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.position_callback, qos_sub)

        # subscribe to mpc owned reference path instead of defining locally
        self.orbit_path_sub = self.create_subscription(
            Path, '/px4_mpc/global_reference_path', self.orbit_path_callback, qos_latched)

        # diagnostics subscriptions
        self.optimal_path_sub = self.create_subscription(
            Float64MultiArray, '/mpc/traj_optimal_raw', self.optimal_path_callback, 1)
        self.plane_actual_sub = self.create_subscription(
            PointStamped, '/mpc/state/plane_actual_pos', self.plane_actual_callback, 1)
        self.solver_state_sub = self.create_subscription(
            PointStamped, '/mpc/state/solver_x0_pos', self.solver_state_callback, 1)

        # rviz visual publishers
        self.vehicle_path_pub = self.create_publisher(Path, "px4_viz/vehicle_path", 10)
        self.vehicle_pose_pub = self.create_publisher(MarkerArray, "px4_viz/vehicle_pose", 10)
        self.optimal_path_pub = self.create_publisher(MarkerArray, "px4_viz/optimal_path", 1)
        self.state_comparison_pub = self.create_publisher(MarkerArray, "px4_viz/state_comparison", 1)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.plane_actual_pos = np.array([0.0, 0.0, 0.0])
        self.solver_x0_pos = np.array([0.0, 0.0, 0.0])
        
        self.optimal_trajectory = None
        self.vehicle_path_msg = Path()
        self.trail_size = 2048

        self.timer = self.create_timer(0.033, self.publish_visuals)

    def orbit_path_callback(self, msg):
        # no local math: whatever mpc publishes is adopted
        pass

    def attitude_callback(self, msg):
        q_enu = (1.0 / np.sqrt(2.0)) * np.array([
            msg.q[0] + msg.q[3],
            msg.q[1] + msg.q[2],
            msg.q[1] - msg.q[2],
            msg.q[0] - msg.q[3]
        ])
        norm = np.linalg.norm(q_enu)
        if norm > 1e-6:
            self.vehicle_attitude[:] = q_enu / norm

    def position_callback(self, msg):
        self.vehicle_local_position[:] = [msg.y, msg.x, -msg.z]

    def plane_actual_callback(self, msg):
        self.plane_actual_pos[:] = [msg.point.x, msg.point.y, msg.point.z]

    def solver_state_callback(self, msg):
        self.solver_x0_pos[:] = [msg.point.x, msg.point.y, msg.point.z]

    def optimal_path_callback(self, msg):
        if not msg.data:
            return
        data = np.array(msg.data)
        nx = 8
        horizon = len(data) // nx
        self.optimal_trajectory = data.reshape((horizon, nx))

    def publish_visuals(self):
        stamp = self.get_clock().now().to_msg()

        # broadcast map -> base_link
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(self.vehicle_local_position[0])
        t.transform.translation.y = float(self.vehicle_local_position[1])
        t.transform.translation.z = float(self.vehicle_local_position[2])
        t.transform.rotation.w = float(self.vehicle_attitude[0])
        t.transform.rotation.x = float(self.vehicle_attitude[1])
        t.transform.rotation.y = float(self.vehicle_attitude[2])
        t.transform.rotation.z = float(self.vehicle_attitude[3])
        self.tf_broadcaster.sendTransform(t)

        # vehicle flown trail
        pose_msg = PoseStamped()
        pose_msg.header = t.header
        pose_msg.pose.orientation = t.transform.rotation
        pose_msg.pose.position.x = t.transform.translation.x
        pose_msg.pose.position.y = t.transform.translation.y
        pose_msg.pose.position.z = t.transform.translation.z

        self.vehicle_path_msg.header = pose_msg.header
        self.vehicle_path_msg.poses.append(pose_msg)
        if len(self.vehicle_path_msg.poses) > self.trail_size:
            del self.vehicle_path_msg.poses[0]
        self.vehicle_path_pub.publish(self.vehicle_path_msg)

        self.publish_vehicle_pose(stamp)
        self.publish_state_comparison(stamp)

        if self.optimal_trajectory is not None:
            self.publish_optimal_path(self.optimal_trajectory, stamp)

    def publish_state_comparison(self, stamp):
        marker_array = MarkerArray()

        # plane actual telemetry
        m_actual = Marker()
        m_actual.header.stamp = stamp
        m_actual.header.frame_id = "map"
        m_actual.ns = "state_comparison"
        m_actual.id = 100
        m_actual.type = Marker.SPHERE
        m_actual.action = Marker.ADD
        m_actual.scale.x = m_actual.scale.y = m_actual.scale.z = 1.2
        m_actual.color.a = 0.85
        m_actual.color.r = 0.0
        m_actual.color.g = 0.9
        m_actual.color.b = 1.0
        m_actual.pose.position.x = float(self.plane_actual_pos[0])
        m_actual.pose.position.y = float(self.plane_actual_pos[1])
        m_actual.pose.position.z = float(self.plane_actual_pos[2])
        m_actual.pose.orientation.w = 1.0

        # solver x0 position
        m_solver = Marker()
        m_solver.header.stamp = stamp
        m_solver.header.frame_id = "map"
        m_solver.ns = "state_comparison"
        m_solver.id = 101
        m_solver.type = Marker.SPHERE
        m_solver.action = Marker.ADD
        m_solver.scale.x = m_solver.scale.y = m_solver.scale.z = 0.8
        m_solver.color.a = 0.85
        m_solver.color.r = 1.0
        m_solver.color.g = 0.1
        m_solver.color.b = 0.9
        m_solver.pose.position.x = float(self.solver_x0_pos[0])
        m_solver.pose.position.y = float(self.solver_x0_pos[1])
        m_solver.pose.position.z = float(self.solver_x0_pos[2])
        m_solver.pose.orientation.w = 1.0

        # disparity line
        m_line = Marker()
        m_line.header.stamp = stamp
        m_line.header.frame_id = "map"
        m_line.ns = "state_comparison"
        m_line.id = 102
        m_line.type = Marker.LINE_STRIP
        m_line.action = Marker.ADD
        m_line.scale.x = 0.15
        m_line.color.a = 0.8
        m_line.color.r = 1.0
        m_line.color.g = 1.0
        m_line.color.b = 0.0
        p_act = Point(x=float(self.plane_actual_pos[0]), y=float(self.plane_actual_pos[1]), z=float(self.plane_actual_pos[2]))
        p_sol = Point(x=float(self.solver_x0_pos[0]), y=float(self.solver_x0_pos[1]), z=float(self.solver_x0_pos[2]))
        m_line.points = [p_act, p_sol]

        marker_array.markers.extend([m_actual, m_solver, m_line])
        self.state_comparison_pub.publish(marker_array)

    def publish_vehicle_pose(self, stamp):
        marker_array = MarkerArray()

        mesh_rotation = np.array([np.cos(np.pi/2), 0.0, 0.0, np.sin(np.pi/2)])
        mesh_attitude = self.quat_multiply(self.vehicle_attitude, mesh_rotation)

        mesh_marker = Marker()
        mesh_marker.header.stamp = stamp
        mesh_marker.header.frame_id = "map"
        mesh_marker.ns = "vehicle_mesh"
        mesh_marker.id = 0
        mesh_marker.type = Marker.MESH_RESOURCE
        mesh_marker.mesh_resource = "package://px4_mpc/resource/believer.dae"
        mesh_marker.scale.x = mesh_marker.scale.y = mesh_marker.scale.z = 1.0
        mesh_marker.color.a = 0.8
        mesh_marker.color.r = mesh_marker.color.g = mesh_marker.color.b = 0.7
        mesh_marker.pose.position.x = float(self.vehicle_local_position[0])
        mesh_marker.pose.position.y = float(self.vehicle_local_position[1])
        mesh_marker.pose.position.z = float(self.vehicle_local_position[2])
        mesh_marker.pose.orientation.w = float(mesh_attitude[0])
        mesh_marker.pose.orientation.x = float(mesh_attitude[1])
        mesh_marker.pose.orientation.y = float(mesh_attitude[2])
        mesh_marker.pose.orientation.z = float(mesh_attitude[3])
        mesh_marker.action = Marker.ADD
        marker_array.markers.append(mesh_marker)

        def create_axis_marker(m_id, r, g, b):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "map"
            m.ns = "vehicle_axes"
            m.id = m_id
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.08
            m.color.a = 1.0
            m.color.r, m.color.g, m.color.b = float(r), float(g), float(b)
            return m

        marker_x = create_axis_marker(10, 1.0, 0.0, 0.0)
        marker_y = create_axis_marker(11, 0.0, 1.0, 0.0)
        marker_z = create_axis_marker(12, 0.0, 0.0, 1.0)

        axis_length = 5.0
        r_mat = self.quat_to_rot_matrix(self.vehicle_attitude)
        pos = self.vehicle_local_position
        p_center = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))

        p_x = Point(x=float(pos[0] + r_mat[0, 0] * axis_length), y=float(pos[1] + r_mat[1, 0] * axis_length), z=float(pos[2] + r_mat[2, 0] * axis_length))
        p_y = Point(x=float(pos[0] + r_mat[0, 1] * axis_length), y=float(pos[1] + r_mat[1, 1] * axis_length), z=float(pos[2] + r_mat[2, 1] * axis_length))
        p_z = Point(x=float(pos[0] + r_mat[0, 2] * axis_length), y=float(pos[1] + r_mat[1, 2] * axis_length), z=float(pos[2] + r_mat[2, 2] * axis_length))

        marker_x.points.extend([p_center, p_x])
        marker_y.points.extend([p_center, p_y])
        marker_z.points.extend([p_center, p_z])
        marker_array.markers.extend([marker_x, marker_y, marker_z])

        self.vehicle_pose_pub.publish(marker_array)

    def publish_optimal_path(self, optimal_trajectory, stamp):
        marker_array = MarkerArray()

        def create_triad_marker(marker_id, r, g, b):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "map"
            m.ns = "optimal_path_triads"
            m.id = marker_id
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.25
            m.color.a = 0.8
            m.color.r, m.color.g, m.color.b = float(r), float(g), float(b)
            return m

        marker_x = create_triad_marker(1, 1.0, 0.0, 0.0)
        marker_y = create_triad_marker(2, 0.0, 1.0, 0.0)
        marker_z = create_triad_marker(3, 0.0, 0.0, 1.0)

        axis_length = 2.0
        qs = optimal_trajectory[:, 4:8]
        norms = np.linalg.norm(qs, axis=1, keepdims=True)
        safe_norms = np.where(norms > 1e-6, norms, 1.0)
        qs_norm = qs / safe_norms

        for t in range(0, optimal_trajectory.shape[0], 5):
            pos = optimal_trajectory[t, 0:3]
            r_mat = self.quat_to_rot_matrix(qs_norm[t])
            p_center = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))

            p_x = Point(x=float(pos[0] + r_mat[0, 0] * axis_length), y=float(pos[1] + r_mat[1, 0] * axis_length), z=float(pos[2] + r_mat[2, 0] * axis_length))
            p_y = Point(x=float(pos[0] + r_mat[0, 1] * axis_length), y=float(pos[1] + r_mat[1, 1] * axis_length), z=float(pos[2] + r_mat[2, 1] * axis_length))
            p_z = Point(x=float(pos[0] + r_mat[0, 2] * axis_length), y=float(pos[1] + r_mat[1, 2] * axis_length), z=float(pos[2] + r_mat[2, 2] * axis_length))

            marker_x.points.extend([p_center, p_x])
            marker_y.points.extend([p_center, p_y])
            marker_z.points.extend([p_center, p_z])

        marker_array.markers.extend([marker_x, marker_y, marker_z])
        self.optimal_path_pub.publish(marker_array)

    def quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quat_to_rot_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

def main(args=None):
    rclpy.init(args=args)
    node = FixedWingVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()