import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
import time

class CBFSafetyFilter:

    def __init__(self, model, N, repair_horizon, Ts, filter_mode="HOCBF", solver_mode="custom", vmin=1.0, gamma1=2.0, gamma2=2.0, beta=5.0, max_iters=10):
        self.model = model
        self.Ts = Ts
        self.N = N
        self.K = repair_horizon
        self.vmin = vmin
        self.filter_mode = filter_mode
        self.solver_mode = solver_mode
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.max_iters = max_iters
        self.beta = beta
        
        self.lh = 1e-5
        self.uh = 1e5
        
        self.last_perf_breakdown = {
            "ms_format": 0.0,
            "ms_init_eval": 0.0,
            "ms_loop": 0.0,
            "iters": 0
        }
        
        self._X_eval = cs.DM.zeros(8, self.K + 1)
        self._U_eval = cs.DM.zeros(3, self.K)
        self._last_u_sol = np.zeros((3, self.K))
        
        # used for both solvers
        self.u_min_np = np.array([self.model.min_fxw, self.model.min_fzw, -self.model.max_roll_rate])
        self.u_max_np = np.array([self.model.max_fxw, self.model.max_fzw, self.model.max_roll_rate])
        # for custom solver
        self._min_bounds = cs.repmat(cs.DM(self.u_min_np), 1, self.K)
        self._max_bounds = cs.repmat(cs.DM(self.u_max_np), 1, self.K)
        
        self._setup_symbolics()
        self._setup_shared_horizon_evaluators()
        self._setup_solver_routing()

    def _get_cbf_components(self, x, u):
        """this was for CT-CBFs"""
        speed = x[3]
        q = x[4:8]
        f_xw, f_zw, roll_r = u[0], u[1], u[2]
        
        R = cs.vertcat(
            cs.horzcat(1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])),
            cs.horzcat(2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[1]**2 + q[3]**2), 2*(q[2]*q[3] - q[0]*q[1])),
            cs.horzcat(2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2))
        )
        
        speed_safe = cs.sqrt(speed**2 + 1e-8)
        # gravity projected into the wind frame (z-up)
        g_inertial = cs.vertcat(0, 0, self.model.gravity)
        g_wind = cs.mtimes(R.T, g_inertial)
        
        # CBF Formulation; psi0 = h = V-Vmin
        psi0 = speed - self.vmin
        h_dot = f_xw + g_wind[0] # g_xw
        
        if self.filter_mode == "FIRST_ORDER":
            cbf_val = h_dot + self.gamma1 * psi0
        elif self.filter_mode == "HOCBF":
            psi1 = h_dot + self.gamma1 * psi0
            q_wind = -(f_zw + g_wind[2]) / speed_safe
            r_wind = g_wind[1] / speed_safe
            h_ddot = r_wind * g_wind[1] - q_wind * g_wind[2] + self.gamma1 * h_dot
            cbf_val = h_ddot + self.gamma2 * psi1
        else:
            raise ValueError(f"bad filter mode: {self.filter_mode}")
            
        return psi0, h_dot, cbf_val

    def _get_dcbf_components(self, x, u):
        """
        Evaluate DT CBF components
        
        Returns:
            h_k: h(x_k) = V_a - vmin, value of the value function
            d_h: d_h = h(x_k+1) - h(x_k), finite difference btwn steps
            cbf_val: cbf condition bound
            - 1st order CBF: h_dot + gamma h(x) \ge 0
            - HOCBF: h_ddot + (gamma1 + gamma2) h_dot + gamma1*gamma2*h \ge 0
        CT: 
        h = V - Vmin
        h_dot  = g_xw + f_xw/m
        psi0 = h(x) = V - Vmin (for HOCBF)
        1st order CBF cond: h_dot + gamma1*h(x) \ge 0
        psi1 = psi0_dot + alpha1(psi0) = h_dot + gamma1*h
        HOCBF condition: psi1_dot + alpha2(psi1) \ge 0
        
        class k functions:
        - alpha1(psi0) = gamma1 * h(x), psi0 = h
        - alpha2(psi1) = gamma2 * psi1
        """
        # map gamma inputs to discrete class-k decay parameters beta / alpha in (0, 0.95)
        # \dot h(x) \ge -gamma h(x) ⇒ \dot h(x) \approx (h(x_{k+1}) - h(x_k))/Ts \ge -gamma h(x_k)
        # h(x_{k+1}) - h(x_k) \ge -gamma*T_s*h(x_k) ⇒ h(x_k) \ge (I - beta) * h(x_{k-1}), beta= gamma*T_s
        # alpha = 1-beta. beta \in (0,1) 
        alpha1 = cs.fmin(cs.fmax(1.0 - self.gamma1 * self.Ts, 1e-4), 1-1e-4)
        alpha2 = cs.fmin(cs.fmax(1.0 - self.gamma2 * self.Ts, 1e-4), 1-1e-4)
        
        h_k = x[3] - self.vmin
        # 1 step with rk4
        x_next = self._step_func(x, u)
        h_next = x_next[3] - self.vmin
        d_h = h_next - h_k # finite difference
        # DT 1st order condition: h(x_k+1) - (1 - beta1)*h(x_k) >= 0 -> equivalent to d_h + beta1*h_k >= 0
        psi1_k = h_next+ alpha1 * h_k
        
        if self.filter_mode == "FIRST_ORDER":
            cbf_val = psi1_k # compare (18) vs (24) in paper
        elif self.filter_mode == "HOCBF":
            # predict 2 steps into the future for higher relative degree
            x_next2 = self._step_func(x_next, u)
            h_next2 = x_next2[3] - self.vmin
            # d_h2 = h_next2 - h_next
            # evaluate psi1 one step ahead
            psi1_next = h_next2 + alpha1 * h_next
            # discrete hocbf condition bound
            cbf_val = psi1_next - alpha2 * psi1_k
        else:
            raise ValueError(f"bad filter mode: {self.filter_mode}")
            
        return h_k, d_h, cbf_val

    def get_cbf_expr(self, x, u):
        return self._get_dcbf_components(x, u)[2]

    def _setup_symbolics(self):
        # build the rk4 integrator first so it can be utilized by the dcbf evaluator
        acados_model = self.model.get_acados_model()
        x_mx = acados_model.x
        u_mx = acados_model.u
        xdot_mx = acados_model.f_expl_expr
        
        f_ode = cs.Function('f_ode', [x_mx, u_mx], [xdot_mx])
        
        dt = self.Ts
        k1 = f_ode(x_mx, u_mx)
        k2 = f_ode(x_mx + 0.5 * dt * k1, u_mx)
        k3 = f_ode(x_mx + 0.5 * dt * k2, u_mx)
        k4 = f_ode(x_mx + dt * k3, u_mx)
        x_next = x_mx + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        q_next = x_next[4:8]
        q_next_norm = q_next / cs.sqrt(cs.sumsqr(q_next) + 1e-6)
        x_next_safe = cs.vertcat(x_next[0:4], q_next_norm)
        
        self._step_func = cs.Function('step_dyn', [x_mx, u_mx], [x_next_safe]).expand()
        
        # use mx symbolics to avoid casadi type mismatches with the integrator
        self._x_sym = cs.MX.sym('x_sym', 8)
        self._u_sym = cs.MX.sym('u_sym', 3)
        
        h_mx, h_next_mx, cbf_val_mx = self._get_dcbf_components(self._x_sym, self._u_sym)
        self._eval_func = cs.Function('eval_cbf_logged', [self._x_sym, self._u_sym], [h_mx, h_next_mx, cbf_val_mx]).expand()
        
        u_single = cs.MX.sym('u_single', 3, 1)
        self._hold_input = cs.Function('hold_input', [u_single], [cs.repmat(u_single, 1, self.N)]).expand()

    def _setup_shared_horizon_evaluators(self):
        K = self.K
        x0_sym = cs.MX.sym('x0_full', 8)
        U_seq_sym = cs.MX.sym('U_seq_full', 3, K)
        
        X_out = [x0_sym]
        x_curr = x0_sym
        total_pen_sym = 0
        
        # CRITICAL FIX: Evaluate penalty on the current state BEFORE stepping kinematics
        for k in range(K):
            cbf_val_k = self.get_cbf_expr(x_curr, U_seq_sym[:, k])
            total_pen_sym += 0.5 * cs.fmax(0, -cbf_val_k)**2
            
            x_curr = self._step_func(x_curr, U_seq_sym[:, k])
            X_out.append(x_curr)
            
        X_sym = cs.horzcat(*X_out)
        self._rollout_func = cs.Function('rollout', [x0_sym, U_seq_sym], [X_sym]).expand()
        self._eval_nominal_penalty = cs.Function('eval_nom_pen', [x0_sym, U_seq_sym], [total_pen_sym]).expand()
        
        self._x0_sym_mx = x0_sym
        self._U_seq_sym_mx = U_seq_sym
        self._total_pen_sym_mx = total_pen_sym

    def _setup_solver_routing(self):
        if self.solver_mode == "custom":
            self._setup_custom_backend()
        elif self.solver_mode == "acados":
            self._setup_acados_backend()
        else:
            raise ValueError(f"unknown solver mode: {self.solver_mode}")

    def _setup_custom_backend(self):
        grad_U_sym = cs.gradient(self._total_pen_sym_mx, self._U_seq_sym_mx)
        grad_norm_sq = cs.sumsqr(grad_U_sym) + 1e-8
        grad_step_sym = 2.0 * self._total_pen_sym_mx / grad_norm_sq

        self._total_grad_func = cs.Function(
            'total_grad',
            [self._x0_sym_mx, self._U_seq_sym_mx],
            [self._total_pen_sym_mx, grad_U_sym, grad_step_sym]
        ).expand()

    def _setup_acados_backend(self):
        ocp = AcadosOcp()
        ocp.model = self.model.get_acados_model()
        
        ocp.solver_options.N_horizon = self.K
        ocp.solver_options.tf = self.K * self.Ts
        
        # Optimization
        # The filter seeks the minimum deviation from the unsafe nominal controls that is safe
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.model.cost_y_expr = ocp.model.u
        ocp.model.cost_y_expr_e = cs.MX.sym('y_e', 0, 1) # No terminal cost
        
        # # Enforce severe penalty on changing roll rate (1e4) to mirror the "masking" behavior
        ocp.cost.W = np.diag([1.0, 1.0, 1.0]) 
        ocp.cost.W_e = np.zeros((0, 0))
        ocp.cost.yref = np.zeros(3) 
        ocp.cost.yref_e = np.zeros(0)
        
        # hard constraints for hardware
        ocp.constraints.lbu = self.u_min_np
        ocp.constraints.ubu = self.u_max_np
        ocp.constraints.idxbu = np.array([0, 1, 2])
        ocp.constraints.x0 = np.zeros(8)
        
        # hard constraint for CBF
        cbf_expr = self.get_cbf_expr(ocp.model.x, ocp.model.u)
        ocp.model.con_h_expr = cs.vertcat(cbf_expr)
        ocp.constraints.lh = np.array([0.0]) # Hard boundary. >= 0
        ocp.constraints.uh = np.array([self.uh])
        
        #no idxsh slack block here for absolute safety

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'# 'SQP_RTI'
        # ocp.solver_options.nlp_solver_max_iter = 5 # self.max_iters
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.qp_solver_warm_start = 2
        
        ocp.code_export_directory = 'cbf_shield_c_generated_code'
        self.qp_solver = AcadosOcpSolver(ocp, json_file='cbf_shield_acados_ocp.json')

    def evaluate_cbf(self, x_val, u_val):
        h, h_dot, val = self._eval_func(x_val, u_val)
        return float(h), float(h_dot), float(val)

    def _run_custom_optimizer(self, x0_dm, total_penalty, sim_step):
        # calculate initial gradients before iterating
        total_penalty_mx, grad_U_mx, grad_step_mx = self._total_grad_func(x0_dm, self._U_eval)
        total_penalty = float(total_penalty_mx)
        
        if total_penalty == 0.0:
            return 0, 0.0
            
        prev_penalty = float('inf')
        iteration = 0
        
        for iteration in range(self.max_iters):
            if abs(prev_penalty - total_penalty) < 1e-9:
                break
            prev_penalty = total_penalty
            
            self._U_eval -= grad_U_mx * (grad_step_mx * self.beta)
            self._U_eval = cs.fmin(cs.fmax(self._U_eval, self._min_bounds), self._max_bounds)
            
            total_penalty_mx, grad_U_mx, grad_step_mx = self._total_grad_func(x0_dm, self._U_eval)
            total_penalty = float(total_penalty_mx)
            
            if total_penalty == 0.0:
                break
                
        return iteration + 1, total_penalty, total_penalty!=0

    def _run_acados_qp_optimizer(self, x0_flat, x0_dm, sim_step):
        # initialize solver to x0
        self.qp_solver.set(0, "lbx", x0_flat)
        self.qp_solver.set(0, "ubx", x0_flat)
        
        # Roll out the current inputs to warm-start the state guesses
        x_guess = self._rollout_func(x0_dm, self._U_eval)
        
        # Push controls back 1 index and duplicate the last stage to warm-start inputs
        shifted_u = np.hstack([self._last_u_sol[:, 1:], self._last_u_sol[:, -1:]])
        
        for k in range(self.K):
            # target reference is the nominal control sequence
            u_nom_k = np.array(self._U_eval[:, k]).flatten()
            u_warm_k = shifted_u[:, k]
            x_guess_k = np.array(x_guess[:, k]).flatten()
            x_guess_k[4:8] /= np.linalg.norm(x_guess_k[4:8])
            self.qp_solver.set(k, "x", x_guess_k)
            self.qp_solver.set(k, "u", u_warm_k)
            self.qp_solver.set(k, "yref", u_nom_k)
            
        self.qp_solver.set(self.K, "x", np.array(x_guess[:, self.K]).flatten())
        status = self.qp_solver.solve()
        
        if status != 0:
            # hard constraint proved infeasible.
            print(f"[t={sim_step*self.Ts:4.2f}]\tCBF Acados Failed (Status {status}). Running last accepted input")
            self._U_eval = cs.DM(shifted_u)
        else:
            # Map the corrected controls back to the internal tensor
            for k in range(self.K):
                solved_u = self.qp_solver.get(k, "u")
                self._U_eval[:, k] = solved_u
                self._last_u_sol[:, k] = solved_u
                
        # Calculate trailing penalty purely for logging accuracy
        total_penalty = float(self._eval_nominal_penalty(x0_dm, self._U_eval))
        
        self._U_eval = cs.fmin(cs.fmax(self._U_eval, self._min_bounds), self._max_bounds)
        return 1, total_penalty, status

    def filter(self, x0, U_seq, sim_step=-1):
        t = time.perf_counter()
        diagnostics=None
        x0_dm = cs.DM(x0) if not isinstance(x0, cs.DM) else x0
        
        if not isinstance(U_seq, cs.DM):
            U_seq = cs.DM(np.atleast_2d(U_seq))
            
        if U_seq.numel() == 3:
            U_seq = self._hold_input(cs.reshape(U_seq, 3, 1))
        else:
            U_seq = U_seq.T if (U_seq.shape == (self.N, 3)) else cs.reshape(U_seq, 3, self.N)
        
        self._U_eval = U_seq[:, :self.K]
        t1 = time.perf_counter()
        self.last_perf_breakdown["ms_format"] = (t1 - t) * 1000.0
        
        # fast pre-check without computing unused gradients
        total_penalty = float(self._eval_nominal_penalty(x0_dm, self._U_eval))
        
        t2 = time.perf_counter()
        self.last_perf_breakdown["ms_init_eval"] = (t2 - t1) * 1000.0
        self.last_perf_breakdown["ms_loop"] = 0.0
        self.last_perf_breakdown["iters"] = 0
        
        if total_penalty == 0:
            self._U_eval = cs.fmin(cs.fmax(self._U_eval, self._min_bounds), self._max_bounds)
            U_seq[:, :self.K] = self._U_eval
            self.last_perf_breakdown["ms_format"] += (time.perf_counter() - t2) * 1000.0
            return np.array(U_seq.T), False, total_penalty, None

        if self.solver_mode == "custom":
            iters_run, total_penalty, solver_status = self._run_custom_optimizer(x0_dm, total_penalty, sim_step)
        elif self.solver_mode == "acados":
            x0_flat = np.array(x0_dm).flatten()
            iters_run, total_penalty, solver_status = self._run_acados_qp_optimizer(x0_flat, x0_dm, sim_step)
        else:
            raise NotImplementedError(f"bad solver mode: {self.solver_mode}")
            
        t4 = time.perf_counter()
        self.last_perf_breakdown["ms_loop"] = (t4 - t2) * 1000.0
        self.last_perf_breakdown["iters"] = iters_run
        
        if solver_status != 0:
                    diagnostics = {
                        'sim_step': sim_step,
                        'solver_mode': self.solver_mode,
                        'status_code': solver_status,
                        'x0': np.array(x0_dm).flatten().tolist(),
                        'U_nom': np.array(U_seq).tolist(),
                        'U_failed': np.array(self._U_eval).tolist(),
                        'final_penalty': total_penalty
                    }
        
        self._U_eval = cs.fmin(cs.fmax(self._U_eval, self._min_bounds), self._max_bounds)
        U_seq[:, :self.K] = self._U_eval
        self.last_perf_breakdown["ms_format"] += (time.perf_counter() - t4) * 1000.0
        
        return np.array(U_seq.T), True, total_penalty, diagnostics

class CompositeCBFSafetyFilter:

    def __init__(self, model, N, repair_horizon, Ts, solver_mode="custom", 
                 vmin=10.0, zmin=10.0, gamma=20.0, kappa=10.0, p0=-1.0, alpha=2.0, 
                 beta=5.0, max_iters=10):
        self.model = model
        self.Ts = Ts
        self.N = N
        self.K = repair_horizon
        self.vmin = vmin
        self.zmin = zmin
        self.solver_mode = solver_mode
        
        self.gamma = gamma 
        self.kappa = kappa 
        self.p0 = p0 
        self.alpha_dt = cs.fmin(cs.fmax(alpha, 1e-4), 1-1e-4) 
        
        self.max_iters = max_iters
        self.beta = beta
        
        self.lh = 1e-5
        self.uh = 1e9
        
        self.last_perf_breakdown = {
            "ms_format": 0.0,
            "ms_init_eval": 0.0,
            "ms_loop": 0.0,
            "iters": 0
        }
        
        self._X_eval = cs.DM.zeros(8, self.K + 1)
        self._U_eval = cs.DM.zeros(3, self.K)
        self._last_u_sol = np.zeros((3, self.K))
        
        self.u_min_np = np.array([self.model.min_fxw, self.model.min_fzw, -self.model.max_roll_rate])
        self.u_max_np = np.array([self.model.max_fxw, self.model.max_fzw, self.model.max_roll_rate])
        self._min_bounds = cs.repmat(cs.DM(self.u_min_np), 1, self.K)
        self._max_bounds = cs.repmat(cs.DM(self.u_max_np), 1, self.K)
        
        self._setup_symbolics()
        self._setup_shared_horizon_evaluators()
        self._setup_solver_routing()

    def _build_cbf_expr(self, x, u, f_expl):
        speed = x[3]
        nu_speed = speed - self.vmin

        alt = x[2]
        h_alt = alt - self.zmin

        q = x[4:8]
        R31 = 2*(q[1]*q[3] - q[0]*q[2])
        h_alt_dot = R31 * speed 

        nu_alt = h_alt_dot - self.p0 * h_alt

        s_speed = cs.tanh(nu_speed / self.gamma)
        s_alt = cs.tanh(nu_alt / self.gamma)

        val1 = -self.kappa * s_speed
        val2 = -self.kappa * s_alt
        max_val = cs.fmax(val1, val2)

        inner_sum = cs.exp(val1 - max_val) + cs.exp(val2 - max_val)
        h_comp = -(self.gamma / self.kappa) * (max_val + cs.log(inner_sum))

        # explicitly compute partial derivatives to guarantee non-zero gradient transmission to HPIPM
        dh_dx = cs.jacobian(h_comp, x)
        
        # extract the drift component f(x) and control matrix g(x) parts from f_expl
        # f_expl = f(x) + g(x)*u. let's evaluate drift by setting u=0 or using explicit extraction
        # alternatively, compute the direct jacobian of the time derivative with respect to u:
        h_comp_dot = cs.mtimes(dh_dx, f_expl)
        
        # explicit control gradient term (L_g h) to verify active-set sensitivity
        dh_du = cs.jacobian(h_comp_dot, u)

        alpha_ct = 1.0
        cbf_val = h_comp_dot + alpha_ct * h_comp
        
        # force the expression to explicitly retain the linear control dependence vector
        return cbf_val, h_comp, h_comp_dot

    def _setup_symbolics(self):
        acados_model = self.model.get_acados_model()
        x_mx = acados_model.x
        u_mx = acados_model.u
        xdot_mx = acados_model.f_expl_expr
        
        self._f_expl = cs.Function('f_expl', [x_mx, u_mx], [xdot_mx]).expand()
        
        dt = self.Ts
        k1 = self._f_expl(x_mx, u_mx)
        k2 = self._f_expl(x_mx + 0.5 * dt * k1, u_mx)
        k3 = self._f_expl(x_mx + 0.5 * dt * k2, u_mx)
        k4 = self._f_expl(x_mx + dt * k3, u_mx)
        x_next = x_mx + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        q_next = x_next[4:8]
        q_next_norm = q_next / cs.sqrt(cs.sumsqr(q_next) + 1e-6)
        x_next_safe = cs.vertcat(x_next[0:4], q_next_norm)
        
        self._step_func = cs.Function('step_dyn', [x_mx, u_mx], [x_next_safe]).expand()
        
        self._x_sym = cs.MX.sym('x_sym', 8)
        self._u_sym = cs.MX.sym('u_sym', 3)
        
        cbf_val_mx, h_comp_mx, h_comp_dot_mx = self._build_cbf_expr(self._x_sym, self._u_sym, self._f_expl(self._x_sym, self._u_sym))
        self._eval_func = cs.Function('eval_cbf_logged', [self._x_sym, self._u_sym], [h_comp_mx, h_comp_dot_mx, cbf_val_mx]).expand()
        
        u_single = cs.MX.sym('u_single', 3, 1)
        self._hold_input = cs.Function('hold_input', [u_single], [cs.repmat(u_single, 1, self.N)]).expand()

    def _get_dcbf_components(self, x, u):
        return self._eval_func(x, u)

    def get_cbf_expr(self, x, u):
        return self._eval_func(x, u)[2]

    def evaluate_cbf(self, x_val, u_val):
        h, h_dot, val = self._eval_func(x_val, u_val)
        return float(h), float(h_dot), float(val)

    def _setup_shared_horizon_evaluators(self):
        K = self.K
        x0_sym = cs.MX.sym('x0_full', 8)
        U_seq_sym = cs.MX.sym('U_seq_full', 3, K)
        
        X_out = [x0_sym]
        x_curr = x0_sym
        total_pen_sym = 0
        
        for k in range(K):
            cbf_val_k = self.get_cbf_expr(x_curr, U_seq_sym[:, k])
            total_pen_sym += 0.5 * cs.fmax(0, -cbf_val_k)**2
            
            x_curr = self._step_func(x_curr, U_seq_sym[:, k])
            X_out.append(x_curr)
            
        X_sym = cs.horzcat(*X_out)
        self._rollout_func = cs.Function('rollout', [x0_sym, U_seq_sym], [X_sym]).expand()
        self._eval_nominal_penalty = cs.Function('eval_nom_pen', [x0_sym, U_seq_sym], [total_pen_sym]).expand()
        
        self._x0_sym_mx = x0_sym
        self._U_seq_sym_mx = U_seq_sym
        self._total_pen_sym_mx = total_pen_sym

    def _setup_solver_routing(self):
        if self.solver_mode == "custom":
            self._setup_custom_backend()
        elif self.solver_mode == "acados":
            self._setup_acados_backend()
        else:
            raise ValueError(f"unknown solver mode: {self.solver_mode}")

    def _setup_custom_backend(self):
        grad_U_sym = cs.gradient(self._total_pen_sym_mx, self._U_seq_sym_mx)
        grad_norm_sq = cs.sumsqr(grad_U_sym) + 1e-8
        grad_step_sym = 2.0 * self._total_pen_sym_mx / grad_norm_sq

        self._total_grad_func = cs.Function(
            'total_grad',
            [self._x0_sym_mx, self._U_seq_sym_mx],
            [self._total_pen_sym_mx, grad_U_sym, grad_step_sym]
        ).expand()

    def _setup_acados_backend(self):
        ocp = AcadosOcp()
        ocp.model = self.model.get_acados_model()
        
        ocp.solver_options.N_horizon = self.K
        ocp.solver_options.tf = self.K * self.Ts
        
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.model.cost_y_expr = ocp.model.u
        ocp.model.cost_y_expr_e = cs.MX.sym('y_e', 0, 1) 
        
        ocp.cost.W = np.diag([20.0, 20.0, 1e2]) 
        ocp.cost.W_e = np.zeros((0, 0))
        ocp.cost.yref = np.zeros(3) 
        ocp.cost.yref_e = np.zeros(0)
        
        ocp.constraints.lbu = self.u_min_np
        ocp.constraints.ubu = self.u_max_np
        ocp.constraints.idxbu = np.array([0, 1, 2])
        ocp.constraints.x0 = np.zeros(8)
        
        cbf_expr, _, _ = self._build_cbf_expr(ocp.model.x, ocp.model.u, ocp.model.f_expl_expr)
        ocp.model.con_h_expr = cs.vertcat(cbf_expr)
        
        n_cbf = 1
        ocp.constraints.lh = np.zeros(n_cbf) 
        ocp.constraints.uh = np.ones(n_cbf) * self.uh
        
        ocp.constraints.idxsh_h = np.array([0], dtype=np.int32)
        ocp.constraints.Jsh = np.eye(n_cbf)
        ocp.constraints.lsh = np.zeros(n_cbf)
        ocp.constraints.ush = np.zeros(n_cbf)
        
        # reduced to 1e5 to prevent hpipm matrix freezing while retaining high penalty priority
        ocp.cost.zl = np.ones(n_cbf) * 1e7
        ocp.cost.Zl = np.ones(n_cbf) * 1e7
        ocp.cost.zu = np.ones(n_cbf) * 1e7
        ocp.cost.Zu = np.ones(n_cbf) * 1e7

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.qp_solver_warm_start = 2
        
        ocp.code_export_directory = 'cbf_shield_c_generated_code'
        self.qp_solver = AcadosOcpSolver(ocp, json_file='cbf_shield_acados_ocp.json')

    def _run_custom_optimizer(self, x0_dm, total_penalty, sim_step):
        total_penalty_mx, grad_U_mx, grad_step_mx = self._total_grad_func(x0_dm, self._U_eval)
        total_penalty = float(total_penalty_mx)
        
        if total_penalty == 0.0:
            return 0, 0.0, 0
            
        prev_penalty = float('inf')
        iteration = 0
        
        u_nom_0 = np.array(self._U_eval[:, 0]).flatten()
        
        for iteration in range(self.max_iters):
            if abs(prev_penalty - total_penalty) < 1e-9:
                break
            prev_penalty = total_penalty
            
            self._U_eval -= grad_U_mx * (grad_step_mx * self.beta)
            self._U_eval = cs.fmin(cs.fmax(self._U_eval, self._min_bounds), self._max_bounds)
            
            total_penalty_mx, grad_U_mx, grad_step_mx = self._total_grad_func(x0_dm, self._U_eval)
            total_penalty = float(total_penalty_mx)
            
            if total_penalty == 0.0:
                break

        u_opt_0 = np.array(self._U_eval[:, 0]).flatten()
        u_nom_str = np.array2string(u_nom_0, formatter={'float_kind':lambda x: "%+6.3e" % x})
        u_opt_str = np.array2string(u_opt_0, formatter={'float_kind':lambda x: "%+6.3e" % x})
        print(f"[CBF Custom Debug] Step {sim_step:>4d} | U_nom: {u_nom_str} | U_opt: {u_opt_str} | Total Penalty: {total_penalty:+6.3e}")

        return iteration + 1, total_penalty, total_penalty !=0

    def _run_acados_qp_optimizer(self, x0_flat, x0_dm, sim_step):
        self.qp_solver.set(0, "lbx", x0_flat)
        self.qp_solver.set(0, "ubx", x0_flat)
        
        x_guess = self._rollout_func(x0_dm, self._U_eval)
        shifted_u = np.hstack([self._last_u_sol[:, 1:], self._last_u_sol[:, -1:]])
        
        for k in range(self.K):
            u_nom_k = np.array(self._U_eval[:, k]).flatten()
            u_warm_k = shifted_u[:, k]
            x_guess_k = np.array(x_guess[:, k]).flatten()
            x_guess_k[4:8] /= np.linalg.norm(x_guess_k[4:8])
            self.qp_solver.set(k, "x", x_guess_k)
            self.qp_solver.set(k, "u", u_warm_k)
            self.qp_solver.set(k, "yref", u_nom_k)
            
        self.qp_solver.set(self.K, "x", np.array(x_guess[:, self.K]).flatten())
        status = self.qp_solver.solve()
        
        u_opt_0 = self.qp_solver.get(0, "u")
        u_nom_0 = np.array(self._U_eval[:, 0]).flatten()
        
        total_penalty = float(self._eval_nominal_penalty(x0_dm, self._U_eval))
        
        u_nom_str = np.array2string(u_nom_0, formatter={'float_kind':lambda x: "%+6.3e" % x})
        u_opt_str = np.array2string(u_opt_0, formatter={'float_kind':lambda x: "%+6.3e" % x})
        # print(f"[CBF Acados Debug] Step {sim_step:>4d} | Status: {status} | U_nom: {u_nom_str} | U_opt: {u_opt_str} | Total Penalty: {total_penalty:+6.3e}")
        
        if status != 0:
            self._U_eval = cs.DM(shifted_u)
        else:
            for k in range(self.K):
                solved_u = self.qp_solver.get(k, "u")
                self._U_eval[:, k] = solved_u
                self._last_u_sol[:, k] = solved_u
                
        self._U_eval = cs.fmin(cs.fmax(self._U_eval, self._min_bounds), self._max_bounds)
        return 1, total_penalty, status

    def filter(self, x0, U_seq, sim_step=-1):
        t = time.perf_counter()
        diagnostics = None
        
        x0_dm = cs.DM(x0) if not isinstance(x0, cs.DM) else x0
        
        if not isinstance(U_seq, cs.DM):
            U_seq = cs.DM(np.atleast_2d(U_seq))
            
        if U_seq.numel() == 3:
            U_seq = self._hold_input(cs.reshape(U_seq, 3, 1))
        else:
            U_seq = U_seq.T if (U_seq.shape == (self.N, 3)) else cs.reshape(U_seq, 3, self.N)
            
        U_nom_cache = np.array(U_seq).copy()
        
        self._U_eval = U_seq[:, :self.K]
        t1 = time.perf_counter()
        self.last_perf_breakdown["ms_format"] = (t1 - t) * 1000.0
        
        total_penalty = float(self._eval_nominal_penalty(x0_dm, self._U_eval))
        
        t2 = time.perf_counter()
        self.last_perf_breakdown["ms_init_eval"] = (t2 - t1) * 1000.0
        self.last_perf_breakdown["ms_loop"] = 0.0
        self.last_perf_breakdown["iters"] = 0
        
        solver_status = 0
        
        if total_penalty == 0:
            self._U_eval = cs.fmin(cs.fmax(self._U_eval, self._min_bounds), self._max_bounds)
            U_seq[:, :self.K] = self._U_eval
            self.last_perf_breakdown["ms_format"] += (time.perf_counter() - t2) * 1000.0
            return np.array(U_seq.T), False, total_penalty, diagnostics

        if self.solver_mode == "custom":
            iters_run, total_penalty, solver_status = self._run_custom_optimizer(x0_dm, total_penalty, sim_step)
        elif self.solver_mode == "acados":
            x0_flat = np.array(x0_dm).flatten()
            iters_run, total_penalty, solver_status = self._run_acados_qp_optimizer(x0_flat, x0_dm, sim_step)
            
        if solver_status != 0:
            diagnostics = {
                'sim_step': sim_step,
                'solver_mode': self.solver_mode,
                'status_code': solver_status,
                'x0': np.array(x0_dm).flatten().tolist(),
                'U_nom': np.array(U_nom_cache).tolist(),
                'U_failed': np.array(self._U_eval).tolist(),
                'final_penalty': total_penalty
            }
            
        t4 = time.perf_counter()
        self.last_perf_breakdown["ms_loop"] = (t4 - t2) * 1000.0
        self.last_perf_breakdown["iters"] = iters_run
        
        self._U_eval = cs.fmin(cs.fmax(self._U_eval, self._min_bounds), self._max_bounds)
        U_seq[:, :self.K] = self._U_eval
        self.last_perf_breakdown["ms_format"] += (time.perf_counter() - t4) * 1000.0
        
        return np.array(U_seq.T), True, total_penalty, diagnostics
    