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

class SpacecraftPropellerModel():
    def __init__(self):
        """
        Planar ATMOS dynamics actuated by four propellers, with a per-thruster
        slew-rate limit built into the model: the thruster forces are STATES and
        the control is their rate of change (df/dt), so the MPC can see and plan
        around the rate limit.

            state    x = [ p(3), v(3), q(4), w(3), f(4) ] -> nx = 17
            control  u = df/dt(4)                         -> nu = 4
            params   p = [ x_ref(13), f_ref(4) ]          -> np = 17
        """
        self.name = 'spacecraft_propeller_model'

        # ---- constants -------------------------------------------------------
        self.mass = 19.0
        self.inertia_zz = 0.34
        self.max_thrust = 1.5
        self.max_thrust_rate = 0.75
        self.torque_arm_length = 0.105
        self.min_thrust     = -self.max_thrust
        self.mass_inv       = 1.0 / self.mass
        self.inertia_zz_inv = 1.0 / self.inertia_zz

    def get_acados_model(self) -> AcadosModel:
        model = AcadosModel()
        # ---- states ----------------------------------------------------------
        x = cs.MX.sym('x', 17)
        u = cs.MX.sym('u', 4)

        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:13]
        F = x[13:17]

        # Normalise the quaternion before use so integrator drift in q does not
        # leak into the rotation / kinematics.
        q_n = q / cs.norm_2(q)

        # ---- allocation acting on the thrust STATE f ------------------------
        D_mat = cs.MX.zeros(2, 4)
        D_mat[0, 0] = -1
        D_mat[0, 1] = 1
        D_mat[1, 2] = -1
        D_mat[1, 3] = 1

        L_mat = cs.MX.zeros(1, 4)
        L_mat[0, 0] = 1
        L_mat[0, 1] = 1
        L_mat[0, 2] = 1
        L_mat[0, 3] = 1
        L_mat = L_mat * self.torque_arm_length

        F_2d   = cs.mtimes(D_mat, F)           # force from thrust STATE
        tau_1d = cs.mtimes(L_mat, F)           # torque from thrust STATE
        F   = cs.vertcat(F_2d[0, 0], F_2d[1, 0], 0.0)
        tau = cs.vertcat(0.0, 0.0, tau_1d)

        # xdot
        p_dot      = cs.MX.sym('p_dot', 3)
        v_dot      = cs.MX.sym('v_dot', 3)
        q_dot      = cs.MX.sym('q_dot', 4)
        w_dot      = cs.MX.sym('w_dot', 3)
        f_dot      = cs.MX.sym('f_dot', 4)

        xdot = cs.vertcat(p_dot, v_dot, q_dot, w_dot, f_dot)

        # ---- planar dynamics: v_z = 0, w_x = w_y = 0, rotation about z only --
        v_planar            = cs.vertcat(v[0], v[1], 0.0)
        a_thrust_planar     = R.v_dot_q_cs(F, q_n) * self.mass_inv
        a_thrust_planar[2]  = 0.0
        w_planar            = cs.vertcat(0.0, 0.0, w[2])
        w_dot_planar        = cs.vertcat(0.0, 0.0, self.inertia_zz_inv * tau[2])
        q_dot_planar        = 0.5 * cs.mtimes(R.skew_symmetric_cs(w_planar), q_n)

        # ---- Assign dynamics -------------------------------------------------
        f_expl = cs.vertcat(
            v_planar,            # p_dot
            a_thrust_planar,     # v_dot
            q_dot_planar,        # q_dot
            w_dot_planar,        # w_dot
            u                    # f_dot = commanded thrust rate
        )
        f_impl = xdot - f_expl

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = self.name

        # ---- parameters: tracking references, defined with the symbols -------
        x_ref = cs.MX.sym('x_ref', 13)
        u_ref = cs.MX.sym('u_ref', 4)
        model.p = cs.vertcat(x_ref, u_ref)

        # ---- actuator limits carried on the model ----------------------------
        model.u_max = np.array([self.max_thrust_rate] * 4)
        model.u_min = -model.u_max
        # thrust-state magnitude limits -> state box on f (x[13:17])
        model.f_max = np.array([self.max_thrust] * 4)
        model.f_min = np.array([self.min_thrust] * 4)

        return model
