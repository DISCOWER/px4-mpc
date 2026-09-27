import warnings
warnings.filterwarnings("ignore", message=".*AcadosSimSolver is created from an AcadosOcp.*")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Patch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.fixedwing_model import FixedWingModel
from controllers.fixedwing_mpc import FixedWingMPC
from px4_mpc.safety_filters import CBFSafetyFilter, CompositeCBFSafetyFilter

def quat_to_R(q):
    """Converts a quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (in radians) to a quaternion [qw, qx, qy, qz].
    Standard aerospace ZYX rotation sequence.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])

def quat_to_euler(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))
    
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.degrees(np.array([roll, pitch, yaw]))

def print_diagnostics(diag, case_label=""):
        step = diag.get('sim_step', -1)
        status = diag.get('status_code', -1)
        mode = diag.get('solver_mode', 'unknown')
        pen = diag.get('final_penalty', 0.0)
        
        print(f"\n[!] solver failed in case: {case_label}")
        print(f"    step: {step} | mode: {mode} | status: {status} | penalty: {pen:.4e}")
        
        x0 = np.array(diag.get('x0', []))
        if len(x0) == 8:
            print(f"    state x0: pos=[{x0[0]:.2f}, {x0[1]:.2f}, {x0[2]:.2f}] v={x0[3]:.2f}")
            
        u_nom = np.array(diag.get('U_nom', []))
        u_fail = np.array(diag.get('U_failed', []))
        
        # pull just the first step commands to see what it was attempting vs what failed
        if u_nom.ndim == 2 and u_fail.ndim == 2:
            print(f"    u_nom[0]:  [{u_nom[0,0]:+.3f}, {u_nom[1,0]:+.3f}, {u_nom[2,0]:+.3f}]")
            print(f"    u_fail[0]: [{u_fail[0,0]:+.3f}, {u_fail[1,0]:+.3f}, {u_fail[2,0]:+.3f}]")

class SimulationCase:
    def __init__(self, v0, rpy_deg, u_raw, dist_k=-1, dist_dv=0.0, solver_mode="custom"):
        self.v0 = v0
        self.rpy_deg = rpy_deg
        self.u_raw = np.array(u_raw)
        self.dist_k = dist_k
        self.dist_dv = dist_dv
        self.solver_mode = solver_mode
        self.cbf_filter=None
        # format legend entries/labels to have same widtths and include the solver type
        r = self.rpy_deg
        u = self.u_raw
        wind_str = "+ wind" if self.dist_dv != 0.0 else ""
        self.label = (f"[{self.solver_mode:6s}] rpy: [{r[0]:4.0f}, {r[1]:4.0f}, {r[2]:4.0f}], "
                      f"u: [{u[0]:5.2f}, {u[1]:5.2f}, {u[2]:5.2f}] {wind_str:<6s}")
        
        self.X = None
        self.U = None
        self.shield_log = None
        self.h_log = None
        self.cbf_log = None
        
        self.color = None
        self.linestyle = 'dashdot' if self.solver_mode == "custom" else '-'
        self.flown_line = None
        self.current_pt = None
        self.quivers = []
        self.vlines = []
        
    def run_sim(self, mpc, cbf_filter, max_steps):
        self.Ts = mpc.Ts
        self.cbf_filter = cbf_filter
        x0 = np.zeros(8)
        x0[2] = 50.0
        x0[3] = self.v0
        x0[4:8] = euler_to_quaternion(np.radians(self.rpy_deg[0]), np.radians(self.rpy_deg[1]), np.radians(self.rpy_deg[2]))
        
        x_curr = x0.copy()
        
        self.X = np.zeros((max_steps, mpc.nx))
        self.U = np.zeros((max_steps, mpc.nu))
        self.shield_log = np.zeros(max_steps, dtype=bool)
        self.h_log = np.zeros(max_steps)
        self.cbf_log = np.zeros(max_steps)
        
        print(f"running sim: {self.label}")
        for k in range(max_steps):
            if k == self.dist_k:
                x_curr[3] += self.dist_dv
            # _, _, _ = mpc.solve_null(x_curr, np.zeros((mpc.N+1, mpc.nx)), verbose=False)
            
            simU_flat = np.tile(self.u_raw, (mpc.N, 1))
            u_filter_horizon, shield_active,_, diagnostics= cbf_filter.filter(x_curr, simU_flat, k)
            u_act = u_filter_horizon[0, :]
            
            h_val, _, cbf_val = cbf_filter._get_dcbf_components(x_curr, u_act)
            
            mpc.integrator.set("x", x_curr)
            mpc.integrator.set("u", u_act)
            mpc.integrator.solve()
            x_curr = mpc.integrator.get("x")
            
            self.X[k, :] = x_curr.copy()
            self.U[k, :] = u_act.copy()
            self.shield_log[k] = shield_active
            self.h_log[k] = float(h_val)
            self.cbf_log[k] = float(cbf_val)
            if diagnostics is not None: 
                print("[!] SOLVER STATUS NEQ 0")
                print_diagnostics(diagnostics, self.label)

    def init_plots(self, ax_3d, axs_2d, color, t_array):
        self.color = color
        
        self.flown_line, = ax_3d.plot([], [], [], color=color, linewidth=1.0, 
                                      label=self.label, linestyle=self.linestyle)
        self.current_pt, = ax_3d.plot([], [], [], 'o', markersize=6, color=color)
        
        # initialize a dedicated floor surface object for this simulation case using its assigned color
        grid_size = 15.0
        x_g = np.linspace(-grid_size, grid_size, 5)
        y_g = np.linspace(-grid_size, grid_size, 5)
        xx_g, yy_g = np.meshgrid(x_g, y_g)
        zz_g = np.zeros_like(xx_g)
        
        self.floor_surface = ax_3d.plot_surface(xx_g, yy_g, zz_g, color=color, alpha=0.1, edgecolor=color, linewidth=0.2)
        
        ax_fxw, ax_fzw, ax_roll, ax_speed, ax_margin, ax_cbf = axs_2d
        
        ax_fxw.plot(t_array, self.U[:, 0], color=color, linestyle=self.linestyle)
        ax_fzw.plot(t_array, self.U[:, 1], color=color, linestyle=self.linestyle)
        ax_roll.plot(t_array, self.U[:, 2], color=color, linestyle=self.linestyle)
        ax_speed.plot(t_array, self.X[:, 3], color=color, linestyle=self.linestyle)
        ax_margin.plot(t_array, self.h_log, color=color, linestyle=self.linestyle)
        ax_cbf.plot(t_array, self.cbf_log, color=color, linestyle=self.linestyle)
        
        hatch_pattern = '///' if self.solver_mode == "custom" else "--"
        if np.any(self.shield_log):
            for ax in axs_2d:
                ax.fill_between(t_array, 0, 1, where=self.shield_log, color=color, 
                                alpha=0.075, hatch=hatch_pattern, transform=ax.get_xaxis_transform())
        
        self.vlines = [
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.8) for ax in axs_2d
        ]

    def update_frame(self, k, ax_3d, t_current):
        self.flown_line.set_data(self.X[:k+1, 0], self.X[:k+1, 1])
        self.flown_line.set_3d_properties(self.X[:k+1, 2])
        
        self.current_pt.set_data([self.X[k, 0]], [self.X[k, 1]])
        self.current_pt.set_3d_properties([self.X[k, 2]])
        
        for q in self.quivers:
            q.remove()
        self.quivers.clear()
        
        # remove the old surface collection if it exists
        if self.floor_surface is not None:
            self.floor_surface.remove()
            
        # extract current x, y position and build a surface grid centered beneath the plane
        px, py = self.X[k, 0], self.X[k, 1]
        floor_z = self.cbf_filter.zmin if hasattr(self.cbf_filter,'zmin') else 0.0
        grid_size = 10.0
        
        x_g = np.linspace(px - grid_size, px + grid_size, 5)
        y_g = np.linspace(py - grid_size, py + grid_size, 5)
        xx_g, yy_g = np.meshgrid(x_g, y_g)
        zz_g = np.full_like(xx_g, floor_z)
        
        # plot the updated surface using the case instance's unique color
        self.floor_surface = ax_3d.plot_surface(xx_g, yy_g, zz_g, color=self.color, alpha=0.1, edgecolor=self.color, linewidth=0.2)
        
        pos = self.X[k, 0:3]
        R = quat_to_R(self.X[k, 4:8])
        x_vec, y_vec, z_vec = R @ [1, 0, 0], R @ [0, 1, 0], R @ [0, 0, 1]
        
        scale = 8.0
        q_x = ax_3d.quiver(*pos, *(x_vec * scale), color='r', arrow_length_ratio=0.1)
        q_y = ax_3d.quiver(*pos, *(y_vec * scale), color='g', arrow_length_ratio=0.1)
        q_z = ax_3d.quiver(*pos, *(z_vec * scale), color='b', arrow_length_ratio=0.1)
        self.quivers.extend([q_x, q_y, q_z])
        
        for line in self.vlines:
            line.set_xdata([t_current, t_current])

def build_dashboard(cases, model, max_steps, Ts, N_horizon):
    t_array = np.arange(max_steps) * Ts
    Tf = N_horizon * Ts
    
    fig = plt.figure(figsize=(22, 11))
    fig.canvas.manager.set_window_title("filter testing in open-loop")
    gs = fig.add_gridspec(3, 4)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.1, wspace=0.3, hspace=0.3)
    
    ax_3d = fig.add_subplot(gs[:, :2], projection='3d')
    ax_fxw = fig.add_subplot(gs[0, 2])
    ax_fzw = fig.add_subplot(gs[1, 2], sharex=ax_fxw)
    ax_roll = fig.add_subplot(gs[2, 2], sharex=ax_fxw)
    ax_speed = fig.add_subplot(gs[0, 3], sharex=ax_fxw)
    ax_margin = fig.add_subplot(gs[1, 3], sharex=ax_fxw)
    ax_cbf = fig.add_subplot(gs[2, 3], sharex=ax_fxw)
    
    axes_2d = [ax_fxw, ax_fzw, ax_roll, ax_speed, ax_margin, ax_cbf]
    colors = ['#FF0505', "#00BDBD", "#5CB800", '#8205FF', '#FF9805', "#CA00A9"]
    
    for idx, case in enumerate(cases):
        color_idx = idx //3 # TODO change back to alternate every 2 or 3 if we do different solvers
        case.init_plots(ax_3d, axes_2d, colors[color_idx % len(colors)], t_array)
        
    all_x = np.concatenate([c.X[:, 0] for c in cases])
    all_y = np.concatenate([c.X[:, 1] for c in cases])
    all_z = np.concatenate([c.X[:, 2] for c in cases])
    mid_x = np.mean([all_x.min(), all_x.max()])
    mid_y = np.mean([all_y.min(), all_y.max()]) 
    mid_z = np.mean([all_z.min(), all_z.max()])
    max_range = np.max([all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()]) / 2.0
    
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # inject n, ts, and tf into the title
    ax_3d.set_title(f"3d flight paths & orientations\nN={N_horizon}, Ts={Ts}s, Tf={Tf:.2f}s")
    handles,labels = ax_3d.get_legend_handles_labels()
    
    custom_patch = Patch(facecolor='gray', alpha=0.3, hatch='///', label='[custom] filter active')
    acados_patch = Patch(facecolor='gray', alpha=0.3, hatch='--', label='[acados] filter active')

    handles.extend([custom_patch, acados_patch])

    ax_3d.legend(handles=handles, loc='upper left', bbox_to_anchor=(.6, 1.0))#, prop={'size': 'small'})

    ax_fxw.set_title('thrust cmd (f_xw)')
    ax_fxw.axhline(model.max_fxw, color='k', linestyle='-', alpha=0.25, label='_nolegend_')
    ax_fxw.axhline(model.min_fxw, color='k', linestyle='-', alpha=0.25, label='_nolegend_')
    ax_fxw.grid(True)
    
    ax_fzw.set_title('lift cmd (f_zw)')
    ax_fzw.axhline(model.max_fzw, color='k', linestyle='-', alpha=0.25, label='_nolegend_')
    ax_fzw.axhline(model.min_fzw, color='k', linestyle='-', alpha=0.25, label='_nolegend_')
    ax_fzw.grid(True)
    
    ax_roll.set_title('roll rate cmd')
    ax_roll.axhline(model.max_roll_rate, color='k', linestyle='-', alpha=0.25, label='_nolegend_')
    ax_roll.axhline(-model.max_roll_rate, color='k', linestyle='-', alpha=0.25, label='_nolegend_')
    ax_roll.set_xlabel('time (s)')
    ax_roll.grid(True)

    vmin = cases[0].X[0,3] if not hasattr(cases[0], 'v_min') else 10.0
    vmin = 10.0
    ax_speed.set_title('airspeed (shaded = filter active)')
    ax_speed.axhline(vmin, color='r', linestyle='--', alpha=0.5, label='_nolegend_')
    ax_speed.grid(True)
    
    ax_margin.set_title('safety margin h(x)')
    ax_margin.axhline(0, color='r', linestyle='--', alpha=0.5, label='_nolegend_')
    ax_margin.grid(True)
    
    ax_cbf.set_title('cbf condition value')
    ax_cbf.axhline(0, color='r', linestyle='-', alpha=0.5, label='_nolegend_')
    ax_cbf.set_xlabel('time (s)')
    ax_cbf.grid(True)

    ax_slider = plt.axes([0.15, 0.02, 0.7, 0.03])
    time_slider = Slider(ax=ax_slider, label='time step', valmin=0, valmax=max_steps-1, valinit=0, valstep=1, color='gray')
    
    def update(val):
        k = int(time_slider.val)
        t_current = t_array[k]
        for case in cases:
            case.update_frame(k, ax_3d, t_current)
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    update(0)
    
    ax_play = plt.axes([0.05, 0.02, 0.05, 0.03])
    btn_play = Button(ax_play, 'play', color='lightgreen')
    state = {'playing': False}
    
    def step_forward():
        if state['playing']:
            next_val = (time_slider.val + 1) % max_steps
            time_slider.set_val(next_val)
            
    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(step_forward)

    def toggle(event):
        state['playing'] = not state['playing']
        btn_play.label.set_text('pause' if state['playing'] else 'play')
        btn_play.color = 'salmon' if state['playing'] else 'lightgreen'
        timer.start() if state['playing'] else timer.stop()
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle)
    
    fig._slider = time_slider
    fig._btn = btn_play
    fig._timer = timer
    
    plt.show()

if __name__ == "__main__":
    max_steps = 500
    Ts = 0.05
    N_horizon = 40
    K_repair = 30
    v_min = 10.0
    z_min = 25.0
    model = FixedWingModel()
    
    print("compiling acados solvers...")
    mpc_solver = FixedWingMPC(model, N=N_horizon, Ts=Ts, cbf_filter=None, x0_init=np.zeros(8))
    cbf_custom = CBFSafetyFilter(model, N_horizon, K_repair, Ts, filter_mode="HOCBF", solver_mode="custom", 
                                 vmin=v_min, gamma1=1.0, gamma2=1.0, beta=8.5, max_iters=8)
    cbf_acados = CBFSafetyFilter(model, N_horizon, K_repair, Ts, filter_mode="HOCBF", solver_mode="acados", 
                                 vmin=v_min, gamma1=1.0, gamma2=1.0, beta=8.5, max_iters=8)
    cbf_comp = CompositeCBFSafetyFilter(model, N_horizon, K_repair, Ts, solver_mode="acados",
                                        vmin=v_min, zmin=z_min, gamma=90.0, p0 = -6.0, kappa=5.0,alpha=.5) 
                                        # vmin=v_min, zmin=z_min, gamma=50.0, p0 = -2.0, kappa=5.0,alpha=.4) 
                                        # vmin=v_min, zmin=z_min, gamma=15.0, p0 = -1.5, kappa=15.0,alpha=2.0) # default
                                        # vmin=v_min, zmin=z_min, gamma=70.0, p0 = -6.0, kappa=80.0) # initial aggressive
    
    print("compilation complete. running cases...")
    
    cases = []
    base_u = [4.0, 9.81, .5]
    
    # roll pitch yaw angles in degrees
    rpy_params = [
        [0, 45, 80],
        [0, 45, 80],
        [0, 45, 80],
        [0, 25, 80],
        [0, 25, 80],
        [0, 25, 80],
        [0, -25, 80],
        [0, -25, 80],
        [0, -25, 80],
        [0, -45, 80],
        [0, -45, 80],
        [0, -45, 80],
        
    ]
    
    u_params = [
        [4.0, 9.81, 0.25],
        [4.0, 9.81, 0.0],
        [4.0, 9.81, -0.25],
        [4.0, 9.81, 0.25],
        [4.0, 9.81, 0.0],
        [4.0, 9.81, -0.25],
        [4.0, 9.81, 0.25],
        [4.0, 9.81, 0.0],
        [4.0, 9.81, -0.25],
        [4.0, 9.81, 0.25],
        [4.0, 9.81, 0.0],
        [4.0, 9.81, -0.25],
    ]

    # parallel array for wind parameters [dist_k, dist_dv]
    wind_params = [
        [ 0, 0.0], 
        [ 0, 0.0], 
        [ 0, 0.0], 
        [ 0, 0.0], 
        [ 0, 0.0], 
        [ 0, 0.0], 
        [ 0, 0.0], 
        [ 0, 0.0], 
        [ 0, 0.0], 
        [ 0, 0.0], 
        [ 0, 0.0], 
        [ 0, 0.0], 
               
    ]
    
    for rpy, u, wind in zip(rpy_params, u_params, wind_params):
        dist_k, dist_dv = wind
        # for solver in ["custom", "acados"]:
        solver = "acados"
        c = SimulationCase(v0=15.0, rpy_deg=rpy, u_raw=u, dist_k=dist_k, dist_dv=dist_dv, solver_mode=solver)
        cases.append(c)
    
    for c in cases:
        # active_filter = cbf_custom if c.solver_mode == "custom" else cbf_comp
        active_filter = cbf_comp
        c.run_sim(mpc_solver, active_filter, max_steps)
        
    build_dashboard(cases, model, max_steps, Ts, N_horizon)
