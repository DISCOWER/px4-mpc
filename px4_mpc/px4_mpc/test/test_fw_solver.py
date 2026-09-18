import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
# from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import models and controllers using absolute ros 2 package paths
from models.fixedwing_model import FixedWingModel
from controllers.fixedwing_mpc import FixedWingMPC
from px4_mpc.safety_filters import CBFSafetyFilter

def quat_to_R(q):
    """converts a quaternion [qw, qx, qy, qz] to a rotation matrix"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (in radians) to a quaternion [qw, qx, qy, qz].
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

def parametrized_ref_path(s):
    """generates a 3d reference point or array of points based on the parameter s"""
    radius = 50.0
    heading_deg = 0.0
    incl_deg = 30.0
    center = np.array([0.0, 0.0, 100.0])
    z_pitch = 40.0  

    h = np.radians(heading_deg)
    inc = np.radians(incl_deg)
    
    cos_h, sin_h = np.cos(h), np.sin(h)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    
    r_z = np.array([[cos_h, -sin_h, 0.0],
                    [sin_h,  cos_h, 0.0],
                    [0.0,    0.0,   1.0]])
    r_y = np.array([[ cos_i, 0.0, sin_i],
                    [ 0.0,   1.0, 0.0  ],
                    [-sin_i, 0.0, cos_i]])
    r_mat = r_y @ r_z
    
    # ensure s is a numpy array for vectorized operations
    s = np.atleast_1d(s)
    
    # calculate base circle in local xy plane
    p_local = radius * np.array([np.cos(s), np.sin(s), np.zeros_like(s)])
    
    # calculate vertical translation to create the coil effect
    z_offset = np.array([np.zeros_like(s), np.zeros_like(s), (s / (2 * np.pi)) * z_pitch])
    
    # rotate and translate into world frame
    path = center + (r_mat @ p_local).T + z_offset.T
    
    return path

def parametrized_ref_path_ellipse(s):
    """generate a 3d elliptical path that climbs in z as s increases"""
    return np.array([
        50.0 * np.cos(s),          # x = a * cos(s)
        80.0 * np.sin(s),          # y = b * sin(s)
        15.0 +40.0 * np.sin(s)     # z increases linearly with s
    ]).T 
    
def parametrized_ref_path_fig8(s):
    """generate figure 8 path that has somes peaks that would induce a stall"""
    peak_height = 15.0
    freq= 1.5
    return np.array([
        20.0*s,
        30.0*np.sin(freq/2*s),
        25.0 + peak_height *np.sin(freq * s)
    ]).T
    
def parametrized_ref_path_stall(s):
    """generate a path that has a steep climb to induce a stall"""
    return np.array([
        20.0*s,
        20.0*s,
        # 20.0*np.cos(s),
        # 20.0*np.sin(s),
        25.0+ 30.0*s
    ]).T

def get_paced_reference(current_state, path, last_idx, nx, N, dt, target_v):
    """paces out a horizon of n+1 points without loop wrap around"""
    current_pos = current_state[0:3]
    actual_u = current_state[3]
    pacing_v = max(actual_u, target_v) 
    
    search_window = 400
    end_idx = min(last_idx + search_window, len(path))
    window_points = path[last_idx:end_idx]
    
    dists = np.linalg.norm(window_points - current_pos, axis=1)
    local_c_idx = np.argmin(dists)
    c_idx = last_idx + local_c_idx
    
    step_dist = pacing_v * dt
    ref_horizon = np.zeros((N + 1, nx))
    
    curr_path_idx = c_idx
    accumulated_dist = 0.0
    
    for i in range(N + 1):
        # assign the current point
        ref_horizon[i, 0:3] = path[curr_path_idx]
        ref_horizon[i, 3] = target_v if actual_u < target_v else actual_u 
        ref_horizon[i, 4] = 1.0       
        
        target_accum = accumulated_dist + step_dist
        
        # increment index to maintain the required velocity pacing
        while accumulated_dist < target_accum:
            next_idx = curr_path_idx + 1
            # exit if lookahead asks for a point that doesn't exist; reached end
            if next_idx >= len(path):
                return None, c_idx, True
                
            dist = np.linalg.norm(path[next_idx] - path[curr_path_idx])
            accumulated_dist += dist
            curr_path_idx = next_idx
            
    return ref_horizon, c_idx, False

def get_raw_command(yes, default):
    """return raw cmd if yes, otherwise default cmd"""
    return np.array([6.0, 9.8, 0.0]) if yes else default

def run_closed_loop_mpc(verbose = False):
    """main simulation loop handling the mpc solver and acados plant integrator"""
    model = FixedWingModel()
    max_sim_steps = 4000 
    N_horizon = 40
    Ts = 0.05
    target_velocity = 15.0
    use_filter = True
    use_raw_cmd = False
    # pack all the filter parameters that are unused elsewhere;
    # plots should be pulling them from the filter instance anyway (ie cbf_obj.gamma1)
    filter_params_hocbf = {
        "filter_mode": "THIRD_ORDER",
        "solver_mode": "acados",
        "repair_horizon": 30,
        "vmin": 10.0,
        "gamma1": 1.0,
        "gamma2": 1.0,
        "beta": 8.5,
        "max_iters": 4,
        "CT_or_DT": "DT",
        "gamma3": 1.0 # only used for experimental 3rd order cbf
    }
    
    # initialize slightly off the path in z-up frame
    x0 = np.zeros(8)
    density = 5000
    s_array = np.linspace(0, 8*np.pi, int(density))
    
    global_path = parametrized_ref_path_stall(s_array)
    x0[0:3] = global_path[0] 
    x0[3] = target_velocity
    x0[4:8] = euler_to_quaternion(0.0, -np.pi/3, np.pi/4) 
    
    # unpack the dictionary using **
    cbf_filter = CBFSafetyFilter(model, N=N_horizon, Ts=Ts, **filter_params_hocbf) if use_filter else None
    mpc = FixedWingMPC(model, N=N_horizon, Ts=Ts, cbf_filter=cbf_filter, x0_init=x0, trackingAttitude=True)
    
    # preallocate logging arrays
    X_hist = np.zeros((max_sim_steps, mpc.nx))
    U_hist = np.zeros((max_sim_steps, mpc.nu))
    X_pred_hist = np.zeros((max_sim_steps, mpc.N + 1, mpc.nx))
    Ref_hist = np.zeros((max_sim_steps, mpc.N + 1, mpc.nx))
    shield_hist = np.zeros(max_sim_steps, dtype=bool)
    
    perf_logs = {
        "ms_mpc": np.zeros(max_sim_steps),
        "ms_filter": np.zeros(max_sim_steps),
        "ms_format": np.zeros(max_sim_steps),
        "ms_init_eval": np.zeros(max_sim_steps),
        # "ms_map": np.zeros(max_sim_steps),
        "ms_loop": np.zeros(max_sim_steps),
        "iters": np.zeros(max_sim_steps, dtype=int)
    }    
    x_curr = x0.copy()
    # for warm-start of rollout
    hover_u = np.array([0.0, model.gravity, 0.0])
    u_safe_guess = np.tile(hover_u, (mpc.N, 1))
    x_safe_guess = np.tile(x0, (mpc.N + 1, 1))
    # u_safe_guess = np.zeros((mpc.N,mpc.nu))
    # x_safe_guess = np.zeros((mpc.N + 1, mpc.nx)) # includes terminal step
    last_closest_idx = 0 
    fail_idx = None
    
    print(f"generated path with {len(global_path)} waypoints")
    print("starting closed-loop paced mpc simulation")
    import gc # garbage collector causes a 20 ms spike
    gc.disable()
    k = 0 
    while k < max_sim_steps:
        ref_slice, closest_idx, path_exhausted = get_paced_reference(
            x_curr, global_path, last_closest_idx, 
            mpc.nx, mpc.N, mpc.Ts, target_velocity
        )
        last_closest_idx = closest_idx
        
        if path_exhausted:
            print(f"mission complete: reached the end of the path buffer at step {k}")
            break
        
        # get reference time
        t_start = time.perf_counter()
        
        # solve mpc and store predicted path
        simU, simX, solver_status = mpc.solve(x_curr, ref_slice,x_safe_guess, u_safe_guess)
        X_pred_hist[k, :, :] = simX
        t_end = time.perf_counter()
        perf_logs["ms_mpc"][k] = (t_end - t_start) * 1000.0 # in ms

        if solver_status != 0:
            print(f"Optimizer mathematically crashed at step {k}. Halting.")
            fail_idx = k
            break
        
        t_filter_call = time.perf_counter()
        
        if cbf_filter is not None and use_filter:
            u_filter_horizon, shield_active,_,_ = cbf_filter.filter(x_curr, get_raw_command(use_raw_cmd,simU),k)
            u_filter = u_filter_horizon[0, :]
            shield_hist[k] = shield_active
            if shield_active and verbose:
                print(f"CBF safety filter activated at step {k}")
            
            u_safe_guess[:-1, :] = u_filter_horizon[1:, :]
            u_safe_guess[ -1, :] = u_filter_horizon[-1, :]
            
            filter_log = cbf_filter.last_perf_breakdown
            perf_logs["ms_format"][k] = filter_log["ms_format"]
            perf_logs["ms_init_eval"][k] = filter_log["ms_init_eval"]
            # perf_logs["ms_map"][k]  = filter_log["ms_map"]
            perf_logs["ms_loop"][k] = filter_log["ms_loop"]
            perf_logs["iters"][k]   = filter_log["iters"]
        else:
            u_filter = simU[0, :]
            shield_hist[k] = False
            
            u_safe_guess[:-1, :] = simU[1:, :]
            u_safe_guess[ -1, :] = simU[-1, :]
        
        x_safe_guess[:-1, :] = simX[1:, :]
        x_safe_guess[ -1, :] = simX[-1, :]
        t_filter_end = time.perf_counter()
        
        perf_logs["ms_filter"][k] = (t_filter_end - t_filter_call) * 1000.0 # in ms
        
        # plant simulation doesn't count toward time
        
        # simulate plant forward
        mpc.integrator.set("x", x_curr)
        mpc.integrator.set("u", u_filter)
        integrator_status = mpc.integrator.solve()
        
        if integrator_status != 0:
            print(f"Integrator failed at step {k}")
            fail_idx = k
            break
            
        x_curr = mpc.integrator.get("x")
        
        X_hist[k, :] = x_curr
        U_hist[k, :] = u_filter
        # store reference slice for plotting
        Ref_hist[k, :, :] = ref_slice

        k += 1

    print("simulation execution finished")
    gc.enable()
    # slice history arrays to match the actual timeline of the simulation
    X_hist = X_hist[:k, :]
    U_hist = U_hist[:k, :]
    X_pred_hist = X_pred_hist[:k, :, :]
    shield_hist = shield_hist[:k]
    for key in perf_logs:
        perf_logs[key]= perf_logs[key][:k]
    
    plot_compute_times(mpc, cbf_filter, k, perf_logs, shield_hist)
    plot_mpc_results(k, X_hist, U_hist, X_pred_hist,Ref_hist, 
                     global_path, mpc, model, cbf_filter, fail_idx, shield_hist)

def setup_control_subplots(fig, gs, col_idx, t, U, model_obj, shield_log=None):
    """builds the 3 control subplots in the specified gridspec column and returns their vertical lines"""
    ax_fxw = fig.add_subplot(gs[0, col_idx])
    ax_fzw = fig.add_subplot(gs[1, col_idx], sharex=ax_fxw)
    ax_roll = fig.add_subplot(gs[2, col_idx], sharex=ax_fxw)
    
    ax_fxw.plot(t, U[:, 0], 'k')
    ax_fxw.axhline(y=model_obj.max_fxw, color='r', linestyle='-', linewidth=1.0, alpha=0.7, label='hardware bound')
    ax_fxw.axhline(y=model_obj.min_fxw, color='r', linestyle='-', linewidth=1.0, alpha=0.7)
    ax_fxw.set_ylabel('f_xw (m/s^2)')
    ax_fxw.set_title('INDI targets')
    ax_fxw.grid(True)
    ax_fxw.legend(loc='upper right', fontsize='small')
    
    ax_fzw.plot(t, U[:, 1], 'b')
    ax_fzw.axhline(y=model_obj.max_fzw, color='r', linestyle='-', linewidth=1.0, alpha=0.7)
    ax_fzw.axhline(y=model_obj.min_fzw, color='r', linestyle='-', linewidth=1.0, alpha=0.7)
    ax_fzw.set_ylabel('f_zw (m/s^2)')
    ax_fzw.grid(True)
    
    ax_roll.plot(t, U[:, 2], 'g')
    ax_roll.axhline(y=model_obj.max_roll_rate, color='r', linestyle='-', linewidth=1.0, alpha=0.7)
    ax_roll.axhline(y=-model_obj.max_roll_rate, color='r', linestyle='-', linewidth=1.0, alpha=0.7)
    ax_roll.set_ylabel('roll rate (rad/s)')
    ax_roll.set_xlabel('time (s)')
    ax_roll.grid(True)
    
    if shield_log is not None and np.any(shield_log):
        # use transform=ax.get_xaxis_transform() and y-values 0 to 1 
        # to shade the entire vertical height of the graph regardless of data scale!
        ax_fxw.fill_between(t, 0, 1, where=shield_log, color='red',edgecolor='none', alpha=0.15, transform=ax_fxw.get_xaxis_transform(), label='Filter Active')
        ax_fzw.fill_between(t, 0, 1, where=shield_log, color='red',edgecolor='none', alpha=0.15, transform=ax_fzw.get_xaxis_transform())
        ax_roll.fill_between(t, 0, 1, where=shield_log, color='red',edgecolor='none', alpha=0.15, transform=ax_roll.get_xaxis_transform())
        ax_fxw.legend(loc='upper right', fontsize='small')

    vline_fxw = ax_fxw.axvline(x=t[0], color='magenta', linestyle='--', linewidth=1.5)
    vline_fzw = ax_fzw.axvline(x=t[0], color='magenta', linestyle='--', linewidth=1.5)
    vline_roll = ax_roll.axvline(x=t[0], color='magenta', linestyle='--', linewidth=1.5)
    
    return [vline_fxw, vline_fzw, vline_roll]

def setup_safety_subplots(fig, gs, col_idx, t, X, U, cbf_obj,shield_log=None):
    """builds the 3 safety metrics subplots in the specified gridspec column and returns their vertical lines"""
    N = len(t)
    h_log = np.zeros(N)
    h_dot_log = np.zeros(N)
    cbf_val_log = np.zeros(N)
    
    for k in range(N):
        h, h_dot, cbf_val = cbf_obj.evaluate_cbf(X[k], U[k])
        h_log[k] = h
        h_dot_log[k] = h_dot
        cbf_val_log[k] = cbf_val
        
    ax_speed = fig.add_subplot(gs[0, col_idx])
    ax_margin = fig.add_subplot(gs[1, col_idx], sharex=ax_speed)
    ax_cbf = fig.add_subplot(gs[2, col_idx], sharex=ax_speed)
    
    ax_speed.plot(t, X[:, 3], 'b-', label='actual airspeed')
    ax_speed.axhline(y=cbf_obj.vmin, color='r', linestyle='--', label='vmin (stall limit)')
    ax_speed.set_ylabel('airspeed (m/s)')
    ax_speed.set_title(f"airspeed CBF stats ($N_{{repair}}={cbf_obj.K}, \\gamma_1 = {cbf_obj.gamma1}, \\gamma_2 = {cbf_obj.gamma2})$"
                       f"\n Filter params: $\\beta={cbf_obj.beta}, N_{{steps\\;max}}={cbf_obj.max_iters}$")
    ax_speed.grid(True)
    ax_speed.legend(loc='upper right', fontsize='small')
    
    ax_margin.plot(t, h_log, 'g-', label='h(x) = v_a - vmin')
    ax_margin.axhline(y=0.0, color='r', linestyle='-', alpha=0.5)
    ax_margin.set_ylabel('safety margin h(x)')
    ax_margin.grid(True)
    ax_margin.legend(loc='upper right', fontsize='small')
    
    ax_cbf.plot(t, cbf_val_log, 'k-', label='cbf condition')
    ax_cbf.axhline(y=0.0, color='r', linestyle='-', alpha=0.8, label='violation (< 0)')
    ax_cbf.set_ylabel('h_dot + gamma*h')
    ax_cbf.set_xlabel('time (s)')
    ax_cbf.grid(True)
    ax_cbf.legend(loc='upper right', fontsize='small')
    
    if shield_log is not None and np.any(shield_log):
        ax_speed.fill_between(t, 0, 1, where=shield_log, color='red',
                              edgecolor='red', alpha=0.15, transform=ax_speed.get_xaxis_transform(), label='Filter Active')
        ax_margin.fill_between(t, 0, 1, where=shield_log, color='red',
                               edgecolor='red', alpha=0.15, transform=ax_margin.get_xaxis_transform())
        ax_cbf.fill_between(t, 0, 1, where=shield_log, color='red',
                            edgecolor='red', alpha=0.15, transform=ax_cbf.get_xaxis_transform())
        ax_speed.legend(loc='upper right', fontsize='small')
    
    vline_speed = ax_speed.axvline(x=t[0], color='magenta', linestyle='--', linewidth=1.5)
    vline_margin = ax_margin.axvline(x=t[0], color='magenta', linestyle='--', linewidth=1.5)
    vline_cbf = ax_cbf.axvline(x=t[0], color='magenta', linestyle='--', linewidth=1.5)
    
    return [vline_speed, vline_margin, vline_cbf]

def add_playback_controls(fig, time_slider, max_val, interval_ms=80):
    """adds a play and pause button that drives a matplotlib slider via a background timer"""
    ax_play = plt.axes([0.01, 0.011, 0.03, 0.02]) # size: [left, bottom, width, height] in figure coordinates
    btn_play = Button(ax_play, 'play', color='lightgreen', hovercolor='palegreen')
    state = {'playing': False}

    def update_step():
        if state['playing']:
            next_val = time_slider.val + 1
            if next_val >= max_val:
                next_val = 0 
            time_slider.set_val(next_val)

    timer = fig.canvas.new_timer(interval=interval_ms)
    timer.add_callback(update_step)

    def toggle(event):
        if state['playing']:
            state['playing'] = False
            btn_play.label.set_text('play')
            btn_play.color = 'lightgreen'
            timer.stop()
        else:
            state['playing'] = True
            btn_play.label.set_text('pause')
            btn_play.color = 'salmon'
            timer.start()
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle)
    return btn_play, timer

def plot_mpc_results(N, X, U, X_pred, ref_slice, 
                     ref_path, mpc, model_obj, cbf_filter, fail_idx=None, shield_log=None):
    """plots an interactive 3d spatial path alongside control and safety subplots"""
    t = np.arange(N) * mpc.Ts
    if fail_idx is not None:
        end_idx = fail_idx
    else:
        valid_idx = np.where(X[:, 4] != 0)[0]
        if len(valid_idx) == 0: return
        end_idx = valid_idx[-1] + 1
        
    t = t[:end_idx]
    X = X[:end_idx]
    U = U[:end_idx]
    X_pred = X_pred[:end_idx]
    px, py, pz = X[:, 0], X[:, 1], X[:, 2] 
    
    fig = plt.figure(figsize=(20, 9))
    fig.canvas.manager.set_window_title("mpc dashboard: tracking, controls, and safety")
    
    # 3x4 gridspec: 2 cols for 3d, 1 col for controls, 1 col for safety
    gs = fig.add_gridspec(3, 3 + (1 if cbf_filter is not None else 0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.12, wspace=0.25, hspace=0.3)
    
    # setup 3d subplot in columns 0 and 1
    ax_3d = fig.add_subplot(gs[:, :2], projection='3d')
    ax_3d.plot(ref_path[:, 0], ref_path[:, 1], ref_path[:, 2], 'g--', label='global reference', alpha=0.3)
    ax_3d.scatter(px[0], py[0], pz[0], color='yellow', s=150, marker='*', label="start", zorder=5)
    
    if fail_idx is not None and fail_idx > 0:
        fail_pos = X[-1, 0:3] 
        ax_3d.scatter(fail_pos[0], fail_pos[1], fail_pos[2], color='red', s=200, marker='X', zorder=10, label="solver fail")

    flown_line, = ax_3d.plot([], [], [], 'k-', linewidth=1, label='flown trajectory')
    pred_line, = ax_3d.plot([], [], [], color='darkorange', linestyle='-', linewidth=2.5, label='prediction horizon', zorder=4)
    current_pt, = ax_3d.plot([], [], [], 'yo', markersize=8)
    x0_pt, = ax_3d.plot([], [], [], linestyle='None', marker='o', color='k', markersize=3, label='rollout y start', zorder=6)
    rollout_end_pt, = ax_3d.plot([], [], [], linestyle='None', marker='D', 
                                 markeredgecolor='k', markerfacecolor='none', markersize=8, label='rollout y end', zorder=6)
    dynamic_quivers = []

    # invisible lines for legend formatting
    ax_3d.plot([], [], [], color='r', label='wind x (fwd)')
    ax_3d.plot([], [], [], color='g', label='wind y (left)')
    ax_3d.plot([], [], [], color='b', label='wind z (up)')
    ax_3d.plot([], [], [], color='c', label='net accel', linewidth=2)
    
    max_range = np.array([px.max()-px.min(), py.max()-py.min(), pz.max()-pz.min()]).max() / 2.0
    if max_range <1.0:  max_range = 10.0
    mid_x, mid_y, mid_z = (px.max()+px.min()) * 0.5, (py.max()+py.min()) * 0.5, (pz.max()+pz.min()) * 0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    ax_3d.set_title(f"Interactive MPC trajectory (z-up), N={mpc.N}, Ts={mpc.Ts}, Tf={mpc.Tf} s")
    ax_3d.legend(loc='upper left', fontsize='small')

    # build 2d subplots using modular functions
    vlines_ctrl = setup_control_subplots(fig, gs, 2, t, U, model_obj,shield_log)
    vlines_safe = []
    if cbf_filter is not None:
        print("CBF safety filter is active, setting up safety subplots.")
        vlines_safe = setup_safety_subplots(fig, gs, 3, t, X, U, cbf_filter,shield_log)
    else:
        print("No CBF safety filter provided, skipping safety subplots.")
    all_vlines = vlines_ctrl + vlines_safe

    # slider and playback ui setup
    ax_slider = plt.axes([0.08, 0.005, 0.8, 0.04]) # size: [left, bottom, width, height] in figure coordinates
    time_slider = Slider(ax=ax_slider, label='time', valmin=0, valmax=len(t)-1, valinit=0, valstep=1, color='magenta')

    def update(val):
        k = int(time_slider.val)
        # dont just constantly clear and redraw the entire 3d plot, 
        # just update the data of the lines and quivers
        flown_line.set_data(px[:k+1], py[:k+1])
        flown_line.set_3d_properties(pz[:k+1])
        
        curr_prediction = X_pred[k]
        pred_line.set_data(curr_prediction[:, 0], curr_prediction[:, 1])
        pred_line.set_3d_properties(curr_prediction[:, 2])
        
        current_pt.set_data([px[k]], [py[k]])
        current_pt.set_3d_properties([pz[k]])
        
        # rollout horizon
        curr_ref=ref_slice[k]
        x0_pt.set_data([curr_ref[0, 0]], [curr_ref[0, 1]])
        x0_pt.set_3d_properties([curr_ref[0, 2]])
        # update empty diamond (last point in the TARGET REFERENCE horizon)
        rollout_end_pt.set_data([curr_ref[-1, 0]], [curr_ref[-1, 1]])
        rollout_end_pt.set_3d_properties([curr_ref[-1, 2]])
        
        for q in dynamic_quivers:
            q.remove()
        dynamic_quivers.clear()
        
        pos = np.array([px[k], py[k], pz[k]])
        R = quat_to_R(X[k, 4:8])
        x_vec, y_vec, z_vec = R @ [1, 0, 0], R @ [0, 1, 0], R @ [0, 0, 1]
        
        scale_triad = 10.0
        q_x = ax_3d.quiver(*pos, *(x_vec * scale_triad), color='r', arrow_length_ratio=0.1)
        q_y = ax_3d.quiver(*pos, *(y_vec * scale_triad), color='g', arrow_length_ratio=0.1)
        q_z = ax_3d.quiver(*pos, *(z_vec * scale_triad), color='b', arrow_length_ratio=0.1)
        dynamic_quivers.extend([q_x, q_y, q_z])
        
        g_inertial = np.array([0.0, 0.0, -9.81])
        f_wind = np.array([U[k, 0], 0.0, U[k, 1]]) 
        accel_inertial = g_inertial + (R @ f_wind)
        
        scale_accel = 0.5 
        if np.linalg.norm(accel_inertial) > 0.1: 
            q_a = ax_3d.quiver(*pos, *(accel_inertial * scale_accel), color='c', linewidth=2, arrow_length_ratio=0.2)
            dynamic_quivers.append(q_a)
            
        # update all vertical timelines simultaneously
        current_time = t[k]
        for line in all_vlines:
            line.set_xdata([current_time, current_time])
            
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    update(0)
    
    play_btn, play_timer = add_playback_controls(fig, time_slider, len(t) - 1, interval_ms=20)
    
    # attach widgets to the figure to prevent garbage collection
    fig._slider = time_slider 
    fig._play_btn = play_btn
    fig._play_timer = play_timer
    
    plt.show()

def plot_compute_times(mpc_obj, cbf_obj, N, perf_logs, shield_hist):
    """Plot granular real-time performance breakdown"""
    dt = mpc_obj.Ts
    t = np.arange(N) * dt
    # AXIS 1: loop times
    fig, ax1 = plt.subplots(figsize=(20, 6))
    fig.canvas.manager.set_window_title("Performance Breakdown")
    
    ms_mpc = perf_logs["ms_mpc"]
    ax1.bar(t, ms_mpc, width=dt, color='#008dff', alpha=0.85, label='MPC (Acados)')
    
    # layer on top the in-filter logs and customize visuals
    filter_layers = [
        ("ms_format",   '#c701ff',  'Filter: Input Formatting'),
        ("ms_init_eval",'#f9e858',  'Filter: Initial Safety Eval'),
        # ("ms_map",      '#ffa6f2',  'Filter: Initial Map Eval'),
        ("ms_loop",     '#ff73b6',  'Filter: Gradient Descent Loop')
    ]
    
    bottom_curr = ms_mpc.copy()
    # iterate dynamically
    for key, color, label_text in filter_layers:
        if key in perf_logs:
            layer_data = perf_logs[key]
            ax1.bar(t, layer_data, bottom=bottom_curr, width=dt, color=color, alpha=0.9, label=label_text)
            bottom_curr += layer_data
            
    # if total filter time differs from internal sum
    if "ms_filter" in perf_logs:
        ms_internal_sum = bottom_curr - ms_mpc
        ms_wrapper_overhead = np.maximum(0.0, perf_logs["ms_filter"] - ms_internal_sum)
        if np.any(ms_wrapper_overhead > 1e-4):
            ax1.bar(t, ms_wrapper_overhead, bottom=bottom_curr, width=dt, 
                   color="#00b45a", alpha=0.7, label='Wrapper/Python Overhead')
            bottom_curr += ms_wrapper_overhead

    # highlight active safety filter triggers
    if shield_hist is not None and np.any(shield_hist):
        active_indices = np.where(shield_hist)[0]
        if len(active_indices) > 0:
            blocks = np.split(active_indices, np.where(np.diff(active_indices) > 1)[0] + 1)
            for block in blocks:
                # plot as a span centered over current timestep: +/- dt/2
                t_start_span = t[block[0]] - (dt / 2.0)
                t_end_span = t[block[-1]] + (dt / 2.0)
                
                ax1.axvspan(t_start_span, t_end_span, color='gray', alpha=0.25, linewidth=0.0, 
                           label='Shield Active' if block is blocks[0] else "")
    
    ax1.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label="50 Hz Target (20ms)")
    ax1.set_xlim(t[0], t[-1])
    ax1.set_xlabel('Simulation Time (s)')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title(f"Detailed Breakdown of Compute Overhead\n$N_{{horizon}}= {mpc_obj.N}, N_{{repair}}={cbf_obj.K}, T_s={dt}\\quad"
                 f"\\gamma_1 = {cbf_obj.gamma1}, \\gamma_2 = {cbf_obj.gamma2}\\quad "
                 f"N_{{steps\\;max}}={cbf_obj.max_iters}, \\beta={cbf_obj.beta}$")
    ax1.grid(True, axis='y', linestyle='-', alpha=0.3)
    # AXIS 2: safety filter iteration hisotry
    ax2 = ax1.twinx()
    iters_data = perf_logs["iters"]
    ax2.axhline(y=cbf_obj.max_iters, color='k', linestyle='--', linewidth=1, alpha=1, label="Max num steps (gradient)")
    
    ax2.plot(t, iters_data, 'k-', linewidth=1.0, alpha=0.5, label='num steps by filter')
    ax2.set_ylabel(r"Safety Filter Gradient Steps ($n_{iters}$)")
    ax2.tick_params(axis='y', labelcolor='black')
    
    # set bounds for the right axis so it doesn't distort view
    max_possible_iters = cbf_obj.max_iters if hasattr(cbf_obj, 'max_iters') else 100
    ax2.set_ylim(0, max(max_possible_iters, np.max(iters_data) if len(iters_data) > 0 else 10)*1.1)
    ax2.grid(False) # Turn off secondary grid lines to avoid cluttering the primary grid

    # combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize='10',bbox_to_anchor=(1.23,1))
    plt.subplots_adjust(right=0.82, top=0.90, bottom=0.1, left=0.05)
if __name__ == "__main__":
    run_closed_loop_mpc()
