import numpy as np
import casadi as cs
import os

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
# from models.fixedwing_model import FixedWingModel

class FixedWingMPC:
    def __init__(self, model,cbf_filter=None, x0_init = None, N=40, Ts=.05,trackingAttitude=False):
        self.N = N
        self.Ts = Ts
        self.Tf = N*Ts
        self.model = model # this is an instance of FixedWingModel, not the AcadosModel
        self.acados_model = self.model.get_acados_model()
        self.nx = self.model.get_acados_model().x.size()[0] # 8
        self.nu = self.model.get_acados_model().u.size()[0] # 3
        self.cbf_filter = cbf_filter

        # need to initialize x0_init if not provided
        if x0_init is None:
            print("Warning: No initial state provided, using default initialization.")
            x0_init = np.zeros(self.nx)
            x0_init[3] = 10.0  # Initialize speed to 10 m/s
            x0_init[4] = 1.0  # Initialize quaternion w=1
        
        self.ocp_solver, self.integrator = self.setup(x0_init, trackingAttitude=trackingAttitude)
        
        for i in range(self.N + 1):
            self.ocp_solver.set(i, "x", x0_init)
            if i < self.N:
                # Give it a realistic hover/flight guess for controls (1g lift)
                self.ocp_solver.set(i, "u", np.array([0.0, 9.81, 0.0]))

    def setup(self, x0,trackingAttitude=False):
        ocp = AcadosOcp()
        ocp.model = self.acados_model
        
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.abspath(os.path.join(this_file_dir, '..'))
        codegen_dir = os.path.join(package_root, 'mpc_codegen')
        json_path = os.path.join(codegen_dir, 'acados_ocp.json')
        os.makedirs(codegen_dir, exist_ok=True)
        ocp.code_export_directory = codegen_dir
        
        # COST FORMULATION
        # x: [px, py, pz, speed, qw, qx, qy, qz]
        # TODO: change to cs.diag(SX([elements]))
        # big number look bad, but technically still take the same fp64 / IEE 754? standard
        Q_all = np.diag([1e3, 1e3, 2e3, 10.0, 50.0, 50.0, 50.0, 50.0])
        Q_pos = np.diag([5e2, 5e2, 4e3, 0.0, 0, 0, 0, 0])
        Q_mat = Q_all if trackingAttitude else Q_pos
        # u: [thrust, lift, roll_rate]
        # R_mat = np.diag([5e2, 5e2, 2e4])    # Control effort penalties
        R_mat = np.diag([1e2, 1e2, 2e4])    # Control effort penalties
        # weights in a block diagonal matrix
        ocp.cost.W = np.block([ [Q_mat, np.zeros((Q_mat.shape[0], R_mat.shape[1]))],
                                [np.zeros((R_mat.shape[0], Q_mat.shape[1])), R_mat] ])
        # TODO: add different terminal cost weights for better convergence
        Q_e = None
        # Q_e = np.diag([1e3, 1e3, 2e3, 0.0, 0.0, 0.0, 0.0, 0.0])
        ocp.cost.W_e = Q_mat if Q_e is None else Q_e
        
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        
        # modify if needed
        ocp.model.cost_y_expr = cs.vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr_e = ocp.model.x
        
        ocp.cost.yref = np.zeros(self.nx+self.nu)
        ocp.cost.yref_e = np.zeros(self.nx)
        
        # ctrl constraints, self.model: FixedWingReducedModel
        ocp.constraints.lbu = np.array([self.model.min_fxw, self.model.min_fzw, -self.model.max_roll_rate])
        ocp.constraints.ubu = np.array([self.model.max_fxw, self.model.max_fzw, self.model.max_roll_rate])
        ocp.constraints.idxbu = np.array([0, 1, 2])
        
        ocp.constraints.idxbx = np.array([3]) # Index 3 is speed
        ocp.constraints.lbx = np.array([0.1]) # Minimum airspeed (stall limit)
        ocp.constraints.ubx = np.array([70.0])# Max structural airspeed
        
        # make airspeed a soft constraint with slack
        # ocp.constraints.idxsbx = np.array([0]) # speed is now first item in idxbx
        ocp.constraints.x0 = x0 # np.zeros
        ocp.constraints.lbu_0 = ocp.constraints.lbu
        ocp.constraints.ubu_0 = ocp.constraints.ubu
        ocp.constraints.idxbu_0 = ocp.constraints.idxbu
        
        # this is if i implemented cbf into the cost function
        
        if self.cbf_filter is not None:
            cbf_expr = self.cbf_filter.get_cbf_expr(ocp.model.x, ocp.model.u)
            ocp.model.con_h_expr = cs.vertcat(cbf_expr)  # Add the CBF constraint to the model
            ocp.constraints.lh = np.array([self.cbf_filter.lh])
            ocp.constraints.uh = np.array([self.cbf_filter.uh])
            # uncomment l96~107 to make h and speed a soft constraint (for debugging only)
            # ocp.constraints.idxsh = np.array([0])
            # in this case we have two soft contraints
            # ocp.cost.zl = np.array([1e5, 1e5]) # linear penalty
            # ocp.cost.zu = np.array([1e5, 1e5])
            # ocp.cost.Zl = np.array([1e6, 1e6]) # quadratic penalty
            # ocp.cost.Zu = np.array([1e6, 1e6])
        else:
            print("Warning: No CBF safety filter provided to MPC cost.")
            # ocp.cost.zl = np.array([1e5]) # linear penalty (lower bound)
            # ocp.cost.zu = np.array([1e5]) # linear penalty (upper bound)
            # ocp.cost.Zl = np.array([1e5]) # quadratic penalty
            # ocp.cost.Zu = np.array([1e5])

        # SOLVER OPTIONS
        ocp.solver_options.tf = self.Tf
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # regularization of hessian matrix for numerical stability/ avoid det = 0
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.levenberg_marquardt = 1e-4
        # ocp.solver_options.sim_method_jac_reuse = 1
        ocp.solver_options.qp_solver_cond_N = self.N
        ocp_solver = AcadosOcpSolver(ocp, json_file=json_path)
        
        ocp.solver_options.qp_solver_warm_start = 2
        # ocp.solver_options.nlp_solver_warm_start_first_qp = True
        
        acados_integrator = AcadosSimSolver(ocp, json_file=json_path)

        return ocp_solver, acados_integrator
    
    def solve(self, x0, yref_trajectory, x_warm_start=None, u_warm_start=None,verbose=True):
        """
        Receives the receding horizon trajectory (N+1 points) and updates the solver.
        """
        ocp_solver = self.ocp_solver
        # normalize quaternion here if i didnt do it in *_model.py
        ocp_solver.set(0, "lbx", x0)
        ocp_solver.set(0, "ubx", x0)

        # inject lookahead trajectory
        if u_warm_start is not None and x_warm_start is not None:
            for i in range(self.N):
                u_val = u_warm_start[i, :]
                x_val = x_warm_start[i,:]
                # gotta do this to keep dynamics in solver clean + not explode
                x_val[4:8] /= (np.linalg.norm(x_val[4:8]) + 1e-8)
                ocp_solver.set(i, "u", u_val)
                ocp_solver.set(i, "x", x_val)
                ocp_solver.set(i, "yref", np.concatenate([yref_trajectory[i], u_val]))
        else:
            u_fallback = np.array([0.0, 9.81, 0.0])
            for i in range(self.N):
                ocp_solver.set(i, "yref", np.concatenate([yref_trajectory[i], u_fallback]))
        ocp_solver.set(self.N, "yref", yref_trajectory[self.N])
        
        status = ocp_solver.solve()

        if status != 0:
            print(f"\n[!] MPC ACADOS SOLVER FAILED (Status {status})")
            ocp_solver.print_statistics()
            if verbose:
                print(f"-> Initial State (x0): {np.round(x0, 3)}")
                
                # find stage w/ highest error btwn internal state and target
                print(f"\t\tquaternion check: {np.linalg.norm(x0[4:8])}")
                max_error = -1
                worst_stage = 0
                worst_x = np.zeros(self.nx)
                worst_target = np.zeros(self.nx)
                
                for k in range(self.N):
                    x_k = ocp_solver.get(k, "x")
                    target_k = yref_trajectory[k, :self.nx] 
                    
                    # euclidean distance for position (assuming px, py, pz are indices 0, 1, 2)
                    error = np.linalg.norm(x_k[0:3] - target_k[0:3]) 
                    
                    if error > max_error:
                        max_error = error
                        worst_stage = k
                        worst_x = x_k
                        worst_target = target_k
                        
                print(f"-> Highest divergence detected at stage {worst_stage} / {self.N}")
                print(f"   Solver's state guess: {np.round(worst_x, 3)}")
                print(f"\t\tquaternion check: {np.linalg.norm(worst_x[4:8])}")
                print(f"   Target reference:     {np.round(worst_target, 3)}")
                print("-" * 50)
            
            # reset the solver with a fallback control input (hover/flight) to avoid endless solver fail
            u_fallback = np.array([0.0, 9.81, 0.0])
            for i in range(self.N + 1):
                ocp_solver.set(i, "x", x0)
                if i < self.N:
                    ocp_solver.set(i, "u", u_fallback)
        simX = np.zeros((self.N+1, self.nx))
        simU = np.zeros((self.N, self.nu))

        for i in range(self.N):
            simX[i, :] = ocp_solver.get(i, "x")
            simU[i, :] = ocp_solver.get(i, "u")
        simX[self.N, :] = ocp_solver.get(self.N, "x")

        return simU, simX, status

    def solve_null(self, x0, yref_trajectory, x_warm_start=None, u_warm_start=None,verbose=True):
            """
            empty solve function
            """
            simX = np.zeros((self.N+1, self.nx))
            simU = np.zeros((self.N, self.nu))
            
            return simU, simX, 0
