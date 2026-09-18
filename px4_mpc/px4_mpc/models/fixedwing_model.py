import casadi as cs
from acados_template import AcadosModel

class FixedWingModel():
    def __init__(self):
        self.name = 'fixedwing_red_model'
        self.gravity = -9.81
        self.max_fxw = 6.0
        self.min_fxw = -2.0
        self.max_fzw = 20.0
        self.min_fzw = 0.1
        self.max_roll_rate = 1.0
        # from config.sdf in PX4:
        """ # these are not physically realistic; gives ubounds of ~10g thrust, ~40g lift
        mass = 1.0 # kg
        wing_A = 0.35 #area,  m^2 
        aspect_ratio = 6.5 # aspect ratio
        rho = 1.2041 # air density, kg/m^3
        c_l_0 = 0.15188
        c_l_a = 5.015
        alpha_stall = 0.3391428111
        c_d_0 = 0.029
        # defines lift drag curves when it exceeds stall angle
        c_l_a_stall = -3.85
        c_d_a_stall = -0.9233984055

        c_l_max = c_l_0+ (c_l_a* alpha_stall)
        # motorconstant and max rotational velocity from .sdf
        max_thrust = 8.54858e-06 * 3500.0**2
        
        # dynamic pressure differences
        q_max= 0.5*rho*v_max**2
        q_min= 0.5*rho*v_min**2
        min_drag = q_min * wing_A * c_d_0
        # max lift: max speed and max alpha before stall and flow separation
        self.max_fzw = q_max * wing_A * c_l_max / mass
        self.min_fzw = 0.1
        # lowkey just ask for it or define as safety constraint
        self.max_roll_rate = 1.0
        # max fwd accel at full throttle + min drag
        self.max_fxw = (max_thrust - min_drag) / mass
        # min fwd accel at zero throttle + max drag
        oswald_eff = 0.97
        c_d_max = c_d_0 + (c_l_max**2) / (cs.pi * oswald_eff * aspect_ratio)
        max_drag = q_max * wing_A* c_d_max
        self.min_fxw = -max_drag / mass
        """

    def get_acados_model(self) -> AcadosModel:
        # helper functions
        def skew_symmetric(v):
            return cs.vertcat(
                cs.horzcat(0, -v[0], -v[1], -v[2]),
                cs.horzcat(v[0], 0, v[2], -v[1]),
                cs.horzcat(v[1], -v[2], 0, v[0]),
                cs.horzcat(v[2], v[1], -v[0], 0)
            )
        # converts from quaternion in wxyz to rotation matrix
        def q_to_rot_mat(q):
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]
            return cs.vertcat(
                cs.horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
                cs.horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
                cs.horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2))
            )

        model = AcadosModel()
        
        # states
        p      = cs.MX.sym('p', 3)
        speed  = cs.MX.sym('speed', 1)
        q      = cs.MX.sym('q', 4)
        # controls [f_xw, f_zw, q (roll rate)] 
        f_xw = cs.MX.sym('f_xw', 1)
        f_zw = cs.MX.sym('f_zw', 1)
        roll_r = cs.MX.sym('roll_rate', 1)
        
        x = cs.vertcat(p, speed, q)
        u = cs.vertcat(f_xw, f_zw, roll_r)
        
        xdot = cs.MX.sym('xdot', 8) # p_dot(3), s_dot(1), q_dot(4)
        # the paper has wind frame set up as FLU (x-fwd y-left z-up)
        # rotate velocity: W → I frame
        q_safe = q/cs.sqrt(cs.sumsqr(q)+1e-6)
        R = q_to_rot_mat(q_safe)
        velocity_i = cs.mtimes(R, cs.vertcat(speed, 0, 0))

        # safe inverse to avoid div 0
        speed_safe = cs.sqrt(speed**2 + 1e-6)
        
        # rotate gravity vector from inertial to WIND frame
        g_inertial = cs.vertcat(0, 0, self.gravity)
        g_wind = cs.mtimes(R.T, g_inertial) # 

        # describe gravity vector in terms of angular rates in wind frame (pitch q, yaw r)
        q_wind = -(f_zw + g_wind[2]) / speed_safe
        r_wind = g_wind[1] / speed_safe
        # fwd accel with gravity; THIS g term was the only (major) part missing in fixedwing_model.py
        accel  = f_xw + g_wind[0] # \dot V_a = g_xw + f_xw / m

        # quaternion kinematics based on angular rates
        omega = cs.vertcat(roll_r, q_wind, r_wind)
        
        q_dot = 0.5 * cs.mtimes(skew_symmetric(omega), q)
        f_expl = cs.vertcat(velocity_i, accel, q_dot)
        
        model.f_impl_expr = xdot - f_expl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = self.name
        
        return model
