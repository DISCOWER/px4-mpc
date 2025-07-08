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

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
import casadi as cs
import os

class SpacecraftDirectAllocationMPC():
    def __init__(self, model):
        self.model = model
        self.Tf = 5.0
        self.N = 49

        self.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.ocp_solver, self.integrator = self.setup(self.x0, self.N, self.Tf)

    def setup(self, x0, N_horizon, Tf):
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # Set directory for code generation and json file
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.abspath(os.path.join(this_file_dir, '..'))
        codegen_dir = os.path.join(package_root, 'mpc_codegen')
        json_path = os.path.join(codegen_dir, 'acados_ocp.json')
        os.makedirs(codegen_dir, exist_ok=True)
        ocp.code_export_directory = codegen_dir

        # set model
        model = self.model.get_acados_model()
        Fmax = self.model.max_thrust

        ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        print(f"nx: {nx}, nu: {nu}")

        # set dimensions
        ocp.dims.N = N_horizon
        ocp.solver_options.N_horizon = N_horizon

        # set cost
        Q_mat = [5e1, 5e1, 5e1,
                 1e1, 1e1, 1e1,
                 8e3,
                 1e1, 1e1, 1e1]
        R_mat = [1e1] * 4

        ocp.cost.W_0 = np.diag(Q_mat + R_mat)
        ocp.cost.W = np.diag(Q_mat + R_mat)
        ocp.cost.W_e = 20 * np.diag(Q_mat)

        # References:
        x_ref = cs.MX.sym('x_ref', (13, 1))
        u_ref = cs.MX.sym('u_ref', (4, 1))

        # Calculate errors
        # x : p,v,q,w               , R9 x SO(3)
        # u : Fx,Fy,Fz,Mx,My,Mz     , R6
        x = ocp.model.x
        u = ocp.model.u

        x_error = x[0:3] - x_ref[0:3]
        x_error = cs.vertcat(x_error, x[3:6] - x_ref[3:6])
        x_error = cs.vertcat(x_error, 1 - (x[6:10].T @ x_ref[6:10])**2)
        x_error = cs.vertcat(x_error, x[10:13] - x_ref[10:13])
        u_error = u - u_ref

        ocp.model.p = cs.vertcat(x_ref, u_ref)

        # define cost with parametric reference
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.cost_type_0 = 'NONLINEAR_LS'

        ocp.model.cost_y_expr_0 = cs.vertcat(x_error, u_error)
        ocp.model.cost_y_expr = cs.vertcat(x_error, u_error)
        ocp.model.cost_y_expr = cs.vertcat(x_error, u_error)
        ocp.model.cost_y_expr_e = x_error

        ocp.cost.yref_0 = np.array([0.0] * (nx - 3 + nu))
        ocp.cost.yref = np.array([0.0] * (nx - 3 + nu))
        ocp.cost.yref_e = np.array([0.0] * (nx - 3))

        # Initialize parameters
        p_0 = np.concatenate((x0, np.zeros(nu)))  # First step is error 0 since x_ref = x0
        ocp.parameter_values = p_0

        # set constraints on U
        ocp.constraints.lbu = np.array([-Fmax, -Fmax, -Fmax, -Fmax])
        ocp.constraints.ubu = np.array([+Fmax, +Fmax, +Fmax, +Fmax])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # set constraints on X
        ocp.constraints.lbx = np.array([-5, -5, -5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ocp.constraints.ubx = np.array([+5, +5, +5, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1])
        ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        # set constraints on X at the end of the horizon
        ocp.constraints.lbx_e = np.array([-5, -5, -5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ocp.constraints.ubx_e = np.array([+5, +5, +5, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1])
        ocp.constraints.idxbx_e = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        # set initial state
        ocp.constraints.x0 = x0

        # set options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
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

        # ocp.solver_options.qp_solver_cond_N = N_horizon
        # ocp.solver_options.print_level = 6

        # set prediction horizon
        ocp.solver_options.tf = Tf

        ocp_solver = AcadosOcpSolver(ocp, json_file=json_path)
        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(ocp, json_file=json_path)

        return ocp_solver, acados_integrator

    def solve(self, x0, verbose=False, ref=None):

        # preparation phase
        ocp_solver = self.ocp_solver

        # Set reference, create zero reference
        if ref is None:
            zero_ref = np.zeros(self.model.get_acados_model().x.size()[0] + self.model.get_acados_model().u.size()[0])
            zero_ref[6] = 1.0

        for i in range(self.N+1):
            if ref is not None:
                # Assumed ref structure: (nx+nu) x N+1
                # NOTE: last u_ref is not used
                p_i = ref[:, i]
                ocp_solver.set(i, "p", p_i)
            else:
                # set all references to 0
                ocp_solver.set(i, "p", zero_ref)

        # set initial state
        ocp_solver.set(0, "lbx", x0.flatten())
        ocp_solver.set(0, "ubx", x0.flatten())

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
