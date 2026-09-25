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
import casadi as cs
import numpy as np
import px4_mpc.utils.rotations as R

class SpacecraftWrenchModel():
    def __init__(self):
        self.name = 'spacecraft_wrench_model'

        # constants
        self.mass = 17.8
        self.inertia = np.diag([0.315]*3)
        self.max_thrust = 2 * 1.4 * 2/3
        self.max_torque = 4 * 0.12 * 1.4 * 1/3

    def get_acados_model(self) -> AcadosModel:
        model = AcadosModel()

        # set up states & controls
        p      = cs.MX.sym('p', 3)
        v      = cs.MX.sym('v', 3)
        q      = cs.MX.sym('q', 4)
        w      = cs.MX.sym('w', 3)

        x = cs.vertcat(p, v, q, w)

        u = cs.MX.sym('u', 6)

        F = u[0:3]
        tau = u[3:6]

        # xdot
        p_dot      = cs.MX.sym('p_dot', 3)
        v_dot      = cs.MX.sym('v_dot', 3)
        q_dot      = cs.MX.sym('q_dot', 4)
        w_dot      = cs.MX.sym('w_dot', 3)

        xdot = cs.vertcat(p_dot, v_dot, q_dot, w_dot)

        q_normalized = q / cs.norm_2(q)
        a_thrust = R.v_dot_q_cs(F, q_normalized)/self.mass

        # dynamics
        f_expl = cs.vertcat(v,
                            a_thrust,
                            1 / 2 * cs.mtimes(R.skew_symmetric_cs(w), q_normalized),
                            np.linalg.inv(self.inertia) @ (tau - cs.cross(w, self.inertia @ w))
                            )

        f_impl = xdot - f_expl

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = self.name

        return model

    def get_error_state(self, x_ref, x):
        """Error state [dp, dv, dphi, dw] (12) between full states x and x_ref (13), used by LQR"""
        dp = x[0:3] - x_ref[0:3]
        dv = x[3:6] - x_ref[3:6]
        dphi = R.quat_to_dphi_cs(R.quat_error_cs(x_ref[6:10], x[6:10]))
        dw = x[10:13] - x_ref[10:13]
        return cs.vertcat(dp, dv, dphi, dw)

    def sym_linearization(self):
        """Linearized error dynamics for LQR
        Returns functions A_fun (12x12) and B_fun (12x6), each called with (x_err, u, x_ref, u_ref)
        """
        model = self.get_acados_model()
        dynamics = cs.Function('f', [model.x, model.u], [model.f_expl_expr])

        # --- symbolic variables ---
        x_ref = cs.MX.sym('x_ref', 13)
        u_ref = cs.MX.sym('u_ref', 6)
        x_err = cs.MX.sym('x_err', 12)
        u = cs.MX.sym('u', 6)

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
        q_delta = cs.vertcat(1.0, 0.5 * dphi)
        q = R.quat_mult_cs(q_ref, q_delta)
        q = q / cs.sqrt(cs.dot(q, q))  # normalize

        x_full = cs.vertcat(p, v, q, w)

        # --- dynamics ---
        f_full = dynamics(x_full, u)
        f_ref = dynamics(x_ref, u_ref)

        # --- error dynamics ---
        dpdot = f_full[0:3] - f_ref[0:3]
        dvdot = f_full[3:6] - f_ref[3:6]
        q_normalized_full = f_full[6:10] / cs.norm_2(f_full[6:10])
        q_e_dot = R.quat_mult_cs(R.quat_conj_cs(q_ref), q_normalized_full)
        dphi_dot = 2 * q_e_dot[1:4]
        dwdot = f_full[10:13] - f_ref[10:13]

        xerr_dot = cs.vertcat(dpdot, dvdot, dphi_dot, dwdot)

        # --- Jacobians ---
        A_fun = cs.Function('A_fun', [x_err, u, x_ref, u_ref],
                            [cs.jacobian(xerr_dot, x_err)])
        B_fun = cs.Function('B_fun', [x_err, u, x_ref, u_ref],
                            [cs.jacobian(xerr_dot, u)])

        return A_fun, B_fun
