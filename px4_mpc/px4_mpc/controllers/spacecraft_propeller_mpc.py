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

from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
import casadi as cs
import os
from px4_mpc.utils.rotations import quat_error_v_cs

class SpacecraftPropellerMPC():
    def __init__(self, model, skip_build=False, kthspace_constraints=False):
        self.model = model
        self.skip_build = skip_build
        self.kthspace_constraints = kthspace_constraints

        self.Tf = 3.0
        self.N = 30
        self.x0 = np.array([0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            1.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0])
        self.f_applied = np.zeros(4)   # last commanded thruster forces

        self.ocp_solver = self.setup(self.x0, self.N, self.Tf)

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

        ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        self.nx = nx
        self.nu = nu
        p0 = np.concatenate((self.x0[:nx-nu], np.zeros(nu)))
        p0[6] = 1.0
        ocp.parameter_values = p0
        self.np_ = p0.size

        # set dimensions
        ocp.dims.N = N_horizon
        ocp.solver_options.N_horizon = N_horizon

        # set cost
        Q_mat = [4e1, 4e1, 4e1,     # position
                2e2, 2e2, 2e2,      # velocity
                4e1, 4e1, 4e1,      # attitude (q_error_v)
                6e0, 6e0, 6e0]      # angular velocity
        R_mat = [5e0] * 4           # propeller forces

        ocp.cost.W_0 = np.diag(Q_mat + R_mat)
        ocp.cost.W = np.diag(Q_mat + R_mat)
        ocp.cost.W_e = 10 * np.diag(Q_mat)

        # Get variables
        x = ocp.model.x[:nx-nu]    # p, v, q, w   (13)
        u = ocp.model.x[nx-nu:]    # thrust states (4)
        x_ref = ocp.model.p[:nx-nu]     # physical-state reference (13)
        u_ref = ocp.model.p[nx-nu:]     # thrust-force reference (4)

        # Calculate errors
        q_error_v = quat_error_v_cs(x[6:10], x_ref[6:10])

        x_error = x[0:3] - x_ref[0:3]
        x_error = cs.vertcat(x_error, x[3:6] - x_ref[3:6])
        x_error = cs.vertcat(x_error, q_error_v)
        x_error = cs.vertcat(x_error, x[10:13] - x_ref[10:13])
        u_error = u - u_ref

        # define cost with parametric reference
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.cost_type_0 = 'NONLINEAR_LS'

        ocp.model.cost_y_expr_0 = cs.vertcat(x_error, u_error)
        ocp.model.cost_y_expr = cs.vertcat(x_error, u_error)
        ocp.model.cost_y_expr_e = x_error

        ocp.cost.yref_0 = np.zeros(ocp.model.cost_y_expr_0.shape[0])
        ocp.cost.yref = np.zeros(ocp.model.cost_y_expr.shape[0])
        ocp.cost.yref_e = np.zeros(ocp.model.cost_y_expr_e.shape[0])

        # set constraints on U
        umin = ocp.model.u_min.copy()
        umax = ocp.model.u_max.copy()
        ocp.constraints.lbu = np.array([umin[0], umin[1], umin[2], umin[3]])
        ocp.constraints.ubu = np.array([umax[0], umax[1], umax[2], umax[3]])
        ocp.constraints.idxbu = np.arange(nu)

        # set constraints on X
        fmin = ocp.model.f_min.copy()
        fmax = ocp.model.f_max.copy()
        if self.kthspace_constraints:
            ocp.constraints.lbx = np.concatenate(([0.3, -1.28, -0.5, -0.5, -0.5], fmin))
            ocp.constraints.ubx = np.concatenate(([+3.8, +1.44, +0.5, +0.5, +0.5], fmax))
            ocp.constraints.idxbx = np.array([0, 1, 3, 4, 12, 13, 14, 15, 16])
        else:
            ocp.constraints.lbx = np.concatenate(([-0.5, -0.5, -0.5], fmin))
            ocp.constraints.ubx = np.concatenate(([+0.5, +0.5, +0.5], fmax))
            ocp.constraints.idxbx = np.array([3, 4, 12, 13, 14, 15, 16])

        # To constrain quaternion states, add indices 6–9 to idxbx/idxbx_e and set their bounds in lbx/ubx.
        # Usually not needed. Valid quaternions stay in [-1, 1], and drift is better fixed by renormalising.

        # Soft constraints are turned on by setting weights for slack variables
        use_soft_constraints = True
        if use_soft_constraints:
            # set weights slack variables for X constraints
            ocp.constraints.idxsbx = np.arange(len(ocp.constraints.idxbx))
            ocp.cost.Zl = np.array([1e6]*len(ocp.constraints.idxsbx))
            ocp.cost.Zu = np.array([1e6]*len(ocp.constraints.idxsbx))
            ocp.cost.zl = np.array([0.0]*len(ocp.constraints.idxsbx))
            ocp.cost.zu = np.array([0.0]*len(ocp.constraints.idxsbx))

            # set weights slack variables for X_e constraints
            ocp.constraints.idxsbx_e = np.arange(len(ocp.constraints.idxbx_e))
            ocp.cost.Zl_e = np.array([1e6]*len(ocp.constraints.idxsbx_e))
            ocp.cost.Zu_e = np.array([1e6]*len(ocp.constraints.idxsbx_e))
            ocp.cost.zl_e = np.array([0.0]*len(ocp.constraints.idxsbx_e))
            ocp.cost.zu_e = np.array([0.0]*len(ocp.constraints.idxsbx_e))

        # set initial state
        ocp.constraints.x0 = x0

        # set options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.levenberg_marquardt = 1e-4  # regularises the un-penalised rate control
        # ocp.solver_options.print_level = 1
        use_RTI=True
        if use_RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
            ocp.solver_options.sim_method_num_stages = 4
            ocp.solver_options.sim_method_num_steps = 3
        else:
            ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP

        # set prediction horizon
        ocp.solver_options.tf = Tf

        # create ocp solver
        ocp_solver = AcadosOcpSolver(ocp, json_file=json_path,
                                    generate=not self.skip_build,
                                    build=not self.skip_build)
        return ocp_solver

    def _set_references(self, ref):
        """Set the parameter (reference) at every stage."""
        if ref is None:
            zero_ref = np.zeros(self.np_)                  # 17 = x_ref(13) + f_ref(4)
            zero_ref[6] = 1.0                              # identity quaternion (qw)
        for i in range(self.N + 1):
            self.ocp_solver.set(i, "p", ref[:, i] if ref is not None else zero_ref)

    def reset_solver(self, x0_full):
        """Cold-start: clear solver memory and re-seed every stage from the
        augmented state (thrust included)."""
        x0_full = np.asarray(x0_full).flatten()
        self.ocp_solver.reset()
        for i in range(self.N + 1):
            self.ocp_solver.set(i, "x", x0_full)
        for i in range(self.N):
            self.ocp_solver.set(i, "u", np.zeros(self.nu))   # zero rate

    def solve(self, x0, ref=None):
        ocp_solver = self.ocp_solver
        nx, nu, N = self.nx, self.nu, self.N
        nx_phys = nx - nu
        
        # build the full (augmented) initial state: append the fed-back thrust
        x0 = np.asarray(x0).flatten()
        if x0.size == nx_phys:
            x0_full = np.concatenate((x0, self.f_applied))
        elif x0.size == nx:
            x0_full = x0
        else:
            raise ValueError(f"x0 has size {x0.size}, expected {nx_phys} or {nx}")
        
        # Use dimensionless initial state and reference for the solver
        x = x0_full
        ref = ref if ref is not None else None

        self._set_references(ref)
        ocp_solver.set(0, "lbx", x)
        ocp_solver.set(0, "ubx", x)
        status = ocp_solver.solve()

        # Recovery: cold-start and retry a couple of times on failure
        max_retries = 2
        for attempt in range(1, max_retries + 1):
            if status == 0:
                break
            print(f"[PropellerMPC] acados status {status}; cold-start retry {attempt}/{max_retries}")
            self.reset_solver(x)
            self._set_references(ref)
            ocp_solver.set(0, "lbx", x)
            ocp_solver.set(0, "ubx", x)
            status = ocp_solver.solve()

        if status != 0:
            print(f"[PropellerMPC] still failing after {max_retries} retries; commanding zero input")
            self.f_applied = np.zeros(nu)
            u = np.zeros((N, nu))
            x = np.tile(x[:nx_phys], (N + 1, 1))
            return u, x

        # extract solution
        x_pred = np.zeros((N + 1, nx))
        x = np.zeros((N + 1, nx_phys))
        u = np.zeros((N, nu))
        for i in range(N + 1):
            x_pred[i, :] = ocp_solver.get(i, "x")
            x[i, :]    = x_pred[i, :nx_phys]
        for i in range(N):
            u[i, :] = x_pred[i + 1, nx_phys:]

        self.f_applied = u[0, :].copy()
        return u, x
