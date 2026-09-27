import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.fixedwing_red_model import FixedWingReducedModel

# ==========================================
# COORDINATE FRAME & ARCHITECTURE NOTE
# ==========================================
# 1. Coordinate Frame: 
#    The state vector strictly uses the NED (North-East-Down) frame.
#    However, the 3D plots flip the Z-axis (pz = -X[:, 2]) so that altitude 
#    visually points "up" (similar to an ENU visualization).
#
# 2. Control Architecture (Cascaded iNDI):
#    Unlike traditional models that command raw actuators (throttle, elevator, aileron),
#    this reduced-order model commands specific aerodynamic forces and rates. 
#    We assume a low-level incremental Nonlinear Dynamic Inversion (iNDI) 
#    controller handles the high-frequency actuator mixing to achieve these targets.
#    - U[0]: f_xw (Drag-augmented forward specific force)
#    - U[1]: f_zw (Lift specific force)
#    - U[2]: roll_rate (Body roll rate, p)
# ==========================================

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def quat_to_R(q):
    """Converts a quaternion [qw, qx, qy, qz] to a rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

def setup_simulator():
    """Compiles the CasADi f_expl function for numerical integration."""
    model_obj = FixedWingReducedModel()
    model = model_obj.get_acados_model()
    f_dyn = cs.Function('f_dyn', [model.x, model.u], [model.f_expl_expr])
    # --- CHANGED: Return the model object to access control bounds for plotting ---
    return f_dyn, model_obj

def rk4_step(f_dyn, x, u, dt):
    """4th-Order Runge-Kutta Integrator with Quaternion Normalization."""
    k1 = np.array(f_dyn(x, u)).flatten()
    k2 = np.array(f_dyn(x + 0.5 * dt * k1, u)).flatten()
    k3 = np.array(f_dyn(x + 0.5 * dt * k2, u)).flatten()
    k4 = np.array(f_dyn(x + dt * k3, u)).flatten()
    
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Normalize quaternion (Indices 4:8 for the reduced 8-state model)
    q = x_next[4:8]
    x_next[4:8] = q / np.linalg.norm(q)
    return x_next, k1

# ==========================================
# 2. CORE SIMULATION ENGINE
# ==========================================
def simulate_scenario(f_dyn, x0, U, dt):
    """Runs the forward propagation for a given control sequence."""
    N = len(U)
    # 8 States: [px, py, pz, speed, qw, qx, qy, qz]
    X = np.zeros((N, 8))
    X_dot = np.zeros((N, 8))
    X[0, :] = x0
    
    for k in range(N - 1):
        X[k+1, :], X_dot[k, :] = rk4_step(f_dyn, X[k, :], U[k, :], dt)
        
    _, X_dot[-1, :] = rk4_step(f_dyn, X[-1, :], U[-1, :], dt)
    
    t = np.arange(0, N * dt, dt)
    return t, X, X_dot

# ==========================================
# 3. SCENARIO GENERATORS
# ==========================================
def get_initial_state():
    x0 = np.zeros(8)
    x0[0:3] = [0, 0, 100]  # Altitude: 100m ENU (Z-Up)
    x0[3]   = 15.0          # Speed: 15 m/s
    x0[4:8] = [1, 0, 0, 0]  # Level flight quaternion (qw=1)
    return x0

def generate_controls(scenario_name, N, dt):
    """
    Generates the U matrix [f_xw, f_zw, roll_rate] acting as targets for the iNDI.
    """
    U = np.zeros((N, 3))
    
    # Base Trim State (Crucial for reduced model to not immediately fall out of sky)
    # f_xw = 0.0 (Drag perfectly matched by thrust)
    # f_zw = 9.81 (Lift perfectly matches gravity to maintain altitude)
    # roll_rate = 0.0
    U[:, 0] = 0.0
    U[:, 1] = 9.81
    U[:, 2] = 0.0
    
    # Timing indices for the phases
    t_1  = int(1.0 / dt)
    t_5  = int(5.0 / dt)
    t_9  = int(9.0 / dt)
    t_13 = int(13.0 / dt)
    
    if scenario_name == "Trim Flight":
        pass # Maintains base trim
        
    elif scenario_name == "Forward Force (f_xw) Doublet":
        # 1~5 sec: Net positive forward force (Accelerate)
        U[t_1:t_5, 0] = 2.0 
        # 5~9 sec: Trim
        # 9~13 sec: Net negative forward force (Decelerate, drag dominates)
        U[t_9:t_13, 0] = -2.0
        
    elif scenario_name == "Lift (f_zw) Doublet":
        # 1~5 sec: Excess lift (Pitch up / Climb)
        U[t_1:t_5, 1] = 9.81 + 2.0 
        # 5~9 sec: Trim
        # 9~13 sec: Deficit lift (Pitch down / Descend)
        U[t_9:t_13, 1] = 9.81 - 2.0
        
    elif scenario_name == "Roll Rate Doublet":
        # 1~5 sec: Positive roll rate (Roll right)
        U[t_1:t_5, 2] = 0.2
        # 5~9 sec: Trim
        # 9~13 sec: Negative roll rate (Roll left)
        U[t_9:t_13, 2] = -0.2
        
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")
        
    return U

# ==========================================
# 4. PLOTTING
# ==========================================
def plot_scenario(t, X, X_dot, U, title, model_obj):
    """Generates the 3D trajectory and control plots for a specific scenario."""
    
    fig = plt.figure(figsize=(16, 8))
    fig.canvas.manager.set_window_title(title)
    
    # 3D Trajectory Subplot
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Extract positions (No flipping needed, we are natively Z-Up!)
    px, py, pz = X[:, 0], X[:, 1], X[:, 2] 
    
    ax_3d.plot(px, py, pz, color='k', linewidth=1.5, label='Flight Path')
    
    # Detect step changes in the control array to plot markers
    dU = np.diff(U, axis=0)
    change_indices = np.where(np.any(dU != 0, axis=1))[0] + 1
    
    ax_3d.scatter(px[0], py[0], pz[0], color='yellow', s=150, marker='*', label="Start")
    for i, idx in enumerate(change_indices):
        label = 'Control Target Change' if i == 0 else "" 
        ax_3d.scatter(px[idx], py[idx], pz[idx], color='magenta', s=150, marker='*', zorder=5, label=label)

    # Plot orientation triads and acceleration vectors
    step = max(1, len(t) // 40) 
    for k in range(0, len(t), step):
        pos = np.array([px[k], py[k], pz[k]])
        R = quat_to_R(X[k, 4:8])
        
        # Triad vectors
        x_vec, y_vec, z_vec = R @ [1, 0, 0], R @ [0, 1, 0], R @ [0, 0, 1]
        
        scale_triad = 4.0
        ax_3d.quiver(*pos, *(x_vec * scale_triad), color='r', arrow_length_ratio=0.1)
        ax_3d.quiver(*pos, *(y_vec * scale_triad), color='g', arrow_length_ratio=0.1)
        ax_3d.quiver(*pos, *(z_vec * scale_triad), color='b', arrow_length_ratio=0.1)
        
        # Cyan Acceleration Vector 
        g_inertial = np.array([0.0, 0.0, -9.81])
        # f_wind = [f_xw, f_yw, f_zw]. The paper assumes zero side-slip, so f_yw = 0.
        f_wind = np.array([U[k, 0], 0.0, U[k, 1]]) 
        
        # Total inertial acceleration
        accel_inertial = g_inertial + (R @ f_wind)
        
        scale_accel = 0.5 # Scaled to not visually overwhelm the orientation triads
        if np.linalg.norm(accel_inertial) > 0.1: # Avoid plotting tiny noise dots
            ax_3d.quiver(*pos, *(accel_inertial * scale_accel), color='c', linewidth=2, arrow_length_ratio=0.2)

    # Invisible lines for legend
    ax_3d.plot([], [], [], color='r', label='Wind Frame X (Velocity)')
    ax_3d.plot([], [], [], color='g', label='Wind Frame Y')
    ax_3d.plot([], [], [], color='b', label='Wind Frame Z (Lift)')
    ax_3d.plot([], [], [], color='c', label='Net Accel', linewidth=2) # Acceleration Legend

    # Auto-scale axes to be equal
    max_range = np.array([px.max()-px.min(), py.max()-py.min(), pz.max()-pz.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (px.max()+px.min()) * 0.5, (py.max()+py.min()) * 0.5, (pz.max()+pz.min()) * 0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    ax_3d.set_xlabel('North (m)')
    ax_3d.set_ylabel('East (m)')
    ax_3d.set_zlabel('Altitude (Up) (m)')
    ax_3d.set_title(f'3D Trajectory: {title}')
    ax_3d.legend(loc='upper left', fontsize='small')

    # 2D Control Subplots (iNDI targets)
    ax_fxw = fig.add_subplot(3, 2, 2)
    ax_fzw = fig.add_subplot(3, 2, 4, sharex=ax_fxw)
    ax_roll = fig.add_subplot(3, 2, 6, sharex=ax_fxw)
    
    # --- Plot f_xw + Bounds ---
    ax_fxw.plot(t, U[:, 0], 'k')
    ax_fxw.axhline(y=model_obj.max_fxw, color='r', linestyle='-', linewidth=1.0, alpha=0.8)
    ax_fxw.axhline(y=model_obj.min_fxw, color='r', linestyle='-', linewidth=1.0, alpha=0.8)
    ax_fxw.set_ylabel('f_xw (m/s^2)')
    ax_fxw.grid(True)
    ax_fxw.set_title('iNDI Control Targets')
    
    # --- Plot f_zw + Bounds ---
    ax_fzw.plot(t, U[:, 1], 'b')
    ax_fzw.axhline(y=model_obj.max_fzw, color='r', linestyle='-', linewidth=1.0, alpha=0.8)
    ax_fzw.axhline(y=model_obj.min_fzw, color='r', linestyle='-', linewidth=1.0, alpha=0.8)
    ax_fzw.set_ylabel('f_zw (m/s^2)')
    ax_fzw.grid(True)
    
    # --- Plot Roll Rate + Bounds ---
    ax_roll.plot(t, U[:, 2], 'g')
    ax_roll.axhline(y=model_obj.max_roll_rate, color='r', linestyle='-', linewidth=1.0, alpha=0.8)
    ax_roll.axhline(y=-model_obj.max_roll_rate, color='r', linestyle='-', linewidth=1.0, alpha=0.8)
    ax_roll.set_ylabel('Roll Rate (rad/s)')
    ax_roll.set_xlabel('Time (s)')
    ax_roll.grid(True)
    
    # Add vertical lines to 2D plots for control changes
    for idx in change_indices:
        t_change = t[idx]
        ax_fxw.axvline(t_change, color='magenta', linestyle='--', alpha=0.5)
        ax_fzw.axvline(t_change, color='magenta', linestyle='--', alpha=0.5)
        ax_roll.axvline(t_change, color='magenta', linestyle='--', alpha=0.5)
    
    plt.tight_layout()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure setup_simulator returns both the function and the model instance
    f_dyn, model_obj = setup_simulator()
    
    dt = 0.05
    T = 20.0
    N = int(T / dt)
    x0 = get_initial_state()
    
    scenarios = [
        "Trim Flight",
        "Forward Force (f_xw) Doublet",
        "Lift (f_zw) Doublet",
        "Roll Rate Doublet"
    ]
    
    for name in scenarios:
        print(f"Simulating: {name}...")
        U = generate_controls(name, N, dt)
        # --- CHANGED: Passed model_obj into plot_scenario to get bounds ---
        t, X, X_dot = simulate_scenario(f_dyn, x0, U, dt)
        plot_scenario(t, X, X_dot, U, name, model_obj)
        
    print("Simulations complete. Displaying plots...")
    plt.show()