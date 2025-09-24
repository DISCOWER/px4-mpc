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
        self.name = 'spacecraft_direct_allocation_model'

        # constants
        self.mass = 17.8
        self.inertia = np.diag([0.315]*3)
        self.max_thrust = 2 * 1.5 * 2/3
        self.max_torque = 4 * 0.12 * 1.5 * 1/3

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
