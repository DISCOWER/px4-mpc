############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
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

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
import scipy.linalg
import casadi as cs

class MultirotorRateMPC():
    def __init__(self, model):
        self.model = model
        self.Tf = 1.0
        self.N = 50

        self.x0 = np.array([0.01, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        self.ocp_solver, self.integrator = self.setup(self.x0, self.N , self.Tf)

    def setup(self, x0, N_horizon, Tf):
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # set model
        model = self.model.get_acados_model()
        Fmax = self.model.max_thrust
        wmax = self.model.max_rate

        ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        # set dimensions
        ocp.dims.N = N_horizon

        # set cost
        Q_mat = 2*np.diag([1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 0.0, 0.1, 0.1, 0.1])
        Q_e = 2*np.diag([3e2, 3e2, 3e2, 1e2, 1e2, 1e2, 0.0, 0.0, 0.0, 0.0])
        R_mat = 2*np.diag([1e1, 5e2, 5e2, 5e2])

        # TODO: How do you add terminal costs?

        # the 'EXTERNAL' cost type can be used to define general cost terms
        # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
        # ocp.cost.cost_type = 'EXTERNAL'
        # ocp.cost.cost_type_e = 'EXTERNAL'
        # ocp.model.cost_expr_ext_cost = model.x.T @ Q_mat @ model.x + model.u.T @ R_mat @ model.u
        # ocp.model.cost_expr_ext_cost_e = model.x.T @ Q_e @ model.x

        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.W_e = scipy.linalg.block_diag(Q_e)


        ocp.model.cost_y_expr = cs.vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x
        ocp.cost.yref  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ocp.cost.yref_e = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # set constraints
        ocp.constraints.lbu = np.array([0.0, -wmax, -wmax, -0.5*wmax])
        ocp.constraints.ubu = np.array([+Fmax,  wmax, wmax, 0.5*wmax])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        ocp.constraints.x0 = x0

        # set options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.print_level = 1
        use_RTI=True
        if use_RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
            ocp.solver_options.sim_method_num_stages = 4
            ocp.solver_options.sim_method_num_steps = 3
        else:
            ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP

        ocp.solver_options.qp_solver_cond_N = N_horizon

        # set prediction horizon
        ocp.solver_options.tf = Tf

        ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')
        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp.json')

        return ocp_solver, acados_integrator
    def solve(self, x0, verbose=False):

        # preparation phase
        ocp_solver = self.ocp_solver

        # set initial state
        ocp_solver.set(0, "lbx", x0)
        ocp_solver.set(0, "ubx", x0)

        status = ocp_solver.solve()
        if verbose:
            self.ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

        if status != 0:
            raise Exception(f'acados returned status {status}.')

        N = self.N
        nx = self.model.get_acados_model().x.size()[0]
        nu = self.model.get_acados_model().u.size()[0]

        simX = np.ndarray((N+1, nx))
        simU = np.ndarray((N, nu))

        # get solution
        for i in range(N):
            simX[i,:] = self.ocp_solver.get(i, "x")
            simU[i,:] = self.ocp_solver.get(i, "u")
        simX[N,:] = self.ocp_solver.get(N, "x")

        return simU, simX
