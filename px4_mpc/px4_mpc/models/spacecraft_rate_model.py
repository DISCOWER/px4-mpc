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
from px4_mpc.utils.rotation_utils import skew_symmetric, v_dot_q
import casadi as cs

class SpacecraftRateModel():
    def __init__(self):
        self.name = 'spacecraft_rate_model'

        # constants
        self.mass = 15.0
        self.max_thrust = 1.5
        self.max_rate = 0.5

        self.nx = 10
        self.nu = 6

    def get_dx(self,x,u):
        p,v,q = x[0:3], x[3:6], x[6:10]
        F,w = u[0:3], u[3:6]
        return cs.vertcat(v,
                          v_dot_q(F,q)/self.mass,
                          1/2*cs.mtimes(skew_symmetric(w),q))
    
    def get_euler_integration(self,x,u,dt):
        return x + self.get_dx(x,u)*dt
    
    def get_rk4_integration(self,x,u,dt):
        k1 = self.get_dx(x, u)
        k2 = self.get_dx(x + dt / 2 * k1, u)
        k3 = self.get_dx(x + dt / 2 * k2, u)
        k4 = self.get_dx(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    def get_acados_model(self) -> AcadosModel:
        model = AcadosModel()

        # set up states & controls
        p      = cs.MX.sym('p', 3)
        v      = cs.MX.sym('v', 3)
        q = cs.MX.sym('q', 4)

        x = cs.vertcat(p, v, q)

        F = cs.MX.sym('F', 3)
        w = cs.MX.sym('w', 3)
        u = cs.vertcat(F, w)

        # xdot
        p_dot      = cs.MX.sym('p_dot', 3)
        v_dot      = cs.MX.sym('v_dot', 3)
        q_dot      = cs.MX.sym('q_dot', 4)

        xdot = cs.vertcat(p_dot, v_dot, q_dot)

        a_thrust = v_dot_q(F, q)/self.mass

        # dynamics
        f_expl = cs.vertcat(v,
                        a_thrust,
                        1 / 2 * cs.mtimes(skew_symmetric(w), q)
                        )

        f_impl = xdot - f_expl


        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = self.name

        return model
