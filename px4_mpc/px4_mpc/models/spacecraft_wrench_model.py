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

from acados_template import AcadosModel
import casadi as ca
import numpy as np

class SpacecraftWrenchModel():
    def __init__(self):
        self.name = 'spacecraft_wrench_model'

        # constants
        self.mass = 17.8
        self.inertia = np.diag([0.315]*3)
        self.max_thrust = 2 * 1.5
        self.max_torque = 4 * 0.12 * 1.5

        # set linearized symbolic matrices
        # functions: 12x12, 12x6 wrt error state
        # call each with x_err, u, x_ref, u_ref
        self.create_model()
        self.A, self.B = self.sym_linearization()

    def create_model(self):
        def skew_symmetric(v):
            return ca.vertcat(ca.horzcat(0, -v[0], -v[1], -v[2]),
                              ca.horzcat(v[0], 0, v[2], -v[1]),
                              ca.horzcat(v[1], -v[2], 0, v[0]),
                              ca.horzcat(v[2], v[1], -v[0], 0))

        def q_to_rot_mat(q):
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]

            rot_mat = ca.vertcat(
                ca.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
                ca.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
                ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

            return rot_mat

        def v_dot_q(v, q):
            rot_mat = q_to_rot_mat(q)

            return ca.mtimes(rot_mat, v)

        # set up states & controls
        p      = ca.MX.sym('p', 3)
        v      = ca.MX.sym('v', 3)
        q      = ca.MX.sym('q', 4)
        w      = ca.MX.sym('w', 3)

        self.x = ca.vertcat(p, v, q, w)
        self.u = ca.MX.sym('u', 6)

        F = self.u[0:3]
        tau = self.u[3:6]

        # xdot
        p_dot      = ca.MX.sym('p_dot', 3)
        v_dot      = ca.MX.sym('v_dot', 3)
        q_dot      = ca.MX.sym('q_dot', 4)
        w_dot      = ca.MX.sym('w_dot', 3)

        self.xdot = ca.vertcat(p_dot, v_dot, q_dot, w_dot)

        # dynamics
        self.f_expl = ca.vertcat(v,
                                 v_dot_q(F, q) / self.mass,
                                 1.0 / 2 * ca.mtimes(skew_symmetric(w), q),
                                 ca.inv(self.inertia) @ (tau - ca.cross(w, self.inertia @ w))
                                 )
        self.dynamics = ca.Function('f', [self.x, self.u], [self.f_expl])
        return

    def get_acados_model(self) -> AcadosModel:
        model = AcadosModel()
        f_impl = self.xdot - self.f_expl

        model.f_impl_expr = f_impl
        model.f_expl_expr = self.f_expl
        model.x = self.x
        model.xdot = self.xdot
        model.u = self.u
        model.name = self.name
        return model

    def quat_mul(self, q1, q2):
        """Quaternion multiplication q = q1 ⊗ q2"""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        return ca.vertcat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def quat_conj(self, q):
        """Quaternion conjugate (also inverse for unit quaternion)"""
        return ca.vertcat(q[0], -q[1], -q[2], -q[3])

    def quat_error(self, q_ref, q):
        """Error quaternion: q_e = q_ref^-1 ⊗ q"""
        return self.quat_mul(self.quat_conj(q_ref), q)

    def quat_to_dphi(self, qe):
        """Small-angle error vector from error quaternion"""
        # Vector part times 2
        return 2 * qe[1:4]

    def get_error_state(self, x_ref, x):
        dp = x[0:3] - x_ref[0:3]
        dv = x[3:6] - x_ref[3:6]
        qe = self.quat_error(x_ref[6:10], x[6:10])
        dphi = self.quat_to_dphi(qe)
        dw = x[10:13] - x_ref[10:13]
        return ca.vertcat(dp, dv, dphi, dw)

    def sym_linearization(self):
        # --- symbolic variables ---
        x_ref = ca.MX.sym('x_ref', 13)
        u_ref = ca.MX.sym('u_ref', 6)
        x_err = ca.MX.sym('x_err', 12)
        u = ca.MX.sym('u', 6)

        # --- reconstruct full state from error ---
        dp = x_err[0:3]
        dv = x_err[3:6]
        dphi = x_err[6:9]
        dw = x_err[9:12]

        p = x_ref[0:3] + dp
        v = x_ref[3:6] + dv
        w = x_ref[10:13] + dw

        # small-angle quaternion approximation: q ≈ q_ref * [1; 0.5*dphi]
        q_ref = x_ref[6:10]
        q_delta = ca.vertcat(1.0, 0.5 * dphi)
        q = self.quat_mul(q_ref, q_delta)
        q = q / ca.sqrt(ca.dot(q, q))  # normalize

        x_full = ca.vertcat(p, v, q, w)

        # --- dynamics ---
        f_full = self.dynamics(x_full, u)
        f_ref = self.dynamics(x_ref, u_ref)

        # --- error dynamics ---
        dpdot = f_full[0:3] - f_ref[0:3]
        dvdot = f_full[3:6] - f_ref[3:6]
        q_e_dot = self.quat_mul(self.quat_conj(q_ref), f_full[6:10])
        dphi_dot = 2 * q_e_dot[1:4]
        dwdot = f_full[10:13] - f_ref[10:13]

        xerr_dot = ca.vertcat(dpdot, dvdot, dphi_dot, dwdot)

        # --- Jacobians ---
        A_fun = ca.Function('A_fun', [x_err, u, x_ref, u_ref],
                            [ca.jacobian(xerr_dot, x_err)])
        B_fun = ca.Function('B_fun', [x_err, u, x_ref, u_ref],
                            [ca.jacobian(xerr_dot, u)])

        return A_fun, B_fun
