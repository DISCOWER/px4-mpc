
import casadi as cs
import numpy as np

def skew_symmetric_cs(v):
    return cs.vertcat(cs.horzcat(0, -v[0], -v[1], -v[2]),
                      cs.horzcat(v[0], 0, v[2], -v[1]),
                      cs.horzcat(v[1], -v[2], 0, v[0]),
                      cs.horzcat(v[2], v[1], -v[0], 0))

def skew_symmetric_np(v):
    return np.array([[0, -v[0], -v[1], -v[2]],
                     [v[0], 0, v[2], -v[1]],
                     [v[1], -v[2], 0, v[0]],
                     [v[2], v[1], -v[0], 0]])

def q_to_rot_mat_cs(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    rot_mat = cs.vertcat(
        cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
        cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))
    return rot_mat

def q_to_rot_mat_np(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    rot_mat = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])
    return rot_mat

def quat_mult_cs(q1, q2):
    return cs.vertcat(
            q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
            q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
            q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
            q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        )

def quat_conj_cs(q):
    """Quaternion conjugate (also inverse for unit quaternion)"""
    return cs.vertcat(q[0], -q[1], -q[2], -q[3])

def quat_error_cs(q_ref, q):
    """Error quaternion: q_e = q_ref^-1 * q"""
    return quat_mult_cs(quat_conj_cs(q_ref), q)

def quat_to_dphi_cs(q_e):
    """Small-angle error vector from error quaternion"""
    return 2 * q_e[1:4]

def quat_mult_np(q1, q2):
    return np.array([
            q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
            q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
            q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
            q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        ])

def v_dot_q_cs(v, q):
    rot_mat = q_to_rot_mat_cs(q)

    return cs.mtimes(rot_mat, v)

def v_dot_q_np(v, q):
    rot_mat = q_to_rot_mat_np(q)

    return np.dot(rot_mat, v)

def quat_error_v_cs(q, q_ref):
    q = q / (cs.norm_2(q) + 1e-8) # Normalize to avoid numerical issues
    q_ref = cs.sign(cs.dot(q, q_ref)) * q_ref # Ensure q_ref has the same sign as q to avoid discontinuity
    q_error_v = cs.vertcat(
        q_ref[0]*q[1] - q_ref[1]*q[0] - q_ref[2]*q[3] + q_ref[3]*q[2],
        q_ref[0]*q[2] + q_ref[1]*q[3] - q_ref[2]*q[0] - q_ref[3]*q[1],
        q_ref[0]*q[3] - q_ref[1]*q[2] + q_ref[2]*q[1] - q_ref[3]*q[0]
    )
    return q_error_v