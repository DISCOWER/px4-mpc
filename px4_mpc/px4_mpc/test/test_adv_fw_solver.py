import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.fixedwing_adv_model import AdvancedFixedWingModel
from controllers.fixedwing_adv_mpc import AdvancedFixedWingMPC

def quat_to_R(q):
    """Converts a quaternion [qw, qx, qy, qz] to a rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

def generate_global_path(length_meters=6000, resolution=0.05):
    """
    Generates a highly dense, purely spatial 3D path. 
    Points are spaced by 'resolution' meters.
    """
    s = np.arange(0, length_meters, resolution)
    path = np.zeros((len(s), 3))
    
    # Keep the same S-curve geometry geometry 
    # (scaled so it stretches properly over the distance)
    path[:, 0] = s
    path[:, 1] = 20 * np.sin(s * (1.0/ 15.0)) 
    path[:, 2] = -100.0
    
    return path

def get_paced_reference(current_state, path, last_idx, nx, nu, N, dt, target_v):
    """
    Finds the closest point on the path, then paces out a horizon of N+1 points 
    spaced exactly by the distance the aircraft should travel in dt at pacing_v.
    """
    current_pos = current_state[0:3]
    actual_u = current_state[3]
    # Bound the current velocity so the horizon doesn't collapse to 0 if we stall
    pacing_v = max(actual_u, 10.0) 
    
    # 1. Find the Anchor (Closest Point)
    search_window = 200  # Only search the next 200 points
    end_idx = min(last_idx + search_window, len(path))
    window_points = path[last_idx:end_idx]
    
    dists = np.linalg.norm(window_points - current_pos, axis=1)
    local_c_idx = np.argmin(dists)
    c_idx = last_idx + local_c_idx
    
    # 2. Pace out the Horizon using ACTUAL current velocity
    step_dist = pacing_v * dt
    ref_horizon = np.zeros((N + 1, nx + nu))
    
    curr_path_idx = c_idx
    accumulated_dist = 0.0
    
    for i in range(N + 1):
        # Populate the target state for stage i
        ref_horizon[i, 0:3] = path[curr_path_idx]
        ref_horizon[i, 3] = target_v  # We still target the ideal 15 m/s
        ref_horizon[i, 6] = 1.0       # Quaternions (qw = 1)
        ref_horizon[i, 13] = 0.5      # Target Thrust (hover/trim)
        
        # Advance along the dense path array by 'step_dist' for the next stage
        target_accum = accumulated_dist + step_dist
        while accumulated_dist < target_accum and curr_path_idx < len(path) - 1:
            p1 = path[curr_path_idx]
            p2 = path[curr_path_idx + 1]
            dist = np.linalg.norm(p2 - p1)
            accumulated_dist += dist
            curr_path_idx += 1
            
    return ref_horizon, c_idx

def run_test():
    model = AdvancedFixedWingModel()
    mpc = AdvancedFixedWingMPC(model, N=40, Ts=0.05)
    solver = mpc.solver
    integrator = mpc.integrator
    
    total_steps = 300
    target_velocity = 15.0
    
    # 1. Generate the dense global geometry
    global_path = generate_global_path()
    
    x_curr = np.zeros(mpc.nx)
    x_curr[0:3] = [0, 0, -100] 
    x_curr[3] = target_velocity           
    x_curr[6] = 1.0            
    
    u_trim = np.array([0.5, 0.0, 0.0])
    for i in range(mpc.N + 1):
        solver.set(i, "x", x_curr)
    for i in range(mpc.N):
        solver.set(i, "u", u_trim)

    X_history = np.zeros((total_steps, mpc.nx))
    U_history = np.zeros((total_steps, mpc.nu))
    
    fail_idx = None
    last_closest_idx = 0 

    print("Starting closed-loop simulation...")
    for k in range(total_steps):
        
        # --- THE SPATIAL PACING MAGIC ---
        ref_slice, closest_idx = get_paced_reference(
            x_curr, global_path, last_closest_idx, 
            mpc.nx, mpc.nu, mpc.N, mpc.Ts, target_velocity
        )
        last_closest_idx = closest_idx
        # --------------------------------
        
        for i in range(mpc.N):
            solver.set(i, "yref", ref_slice[i])
            
        # Truncate terminal reference to match state dimension
        terminal_state_ref = ref_slice[mpc.N, 0:mpc.nx] 
        solver.set(mpc.N, "yref", terminal_state_ref) 
        
        solver.set(0, "lbx", x_curr)
        solver.set(0, "ubx", x_curr)
        
        status = solver.solve()
        if status != 0:
            print(f"\n>>> SOLVER FAILED AT STEP {k} with status {status} <<<")
            np.set_printoptions(suppress=True, precision=4)
            print(f"Current State (x_curr):\n{x_curr}")
            print(f"Target Reference (yref stage 0):\n{ref_slice[0]}")
            fail_idx = k
            break
        
        u_opt = solver.get(0, "u")
        X_history[k, :] = x_curr
        U_history[k, :] = u_opt
        
        integrator.set("x", x_curr)
        integrator.set("u", u_opt)
        integrator_status = integrator.solve()
        if integrator_status != 0:
            print(f"Integrator failed at step {k}")
            fail_idx = k
            break
            
        x_curr = integrator.get("x")
        
        q_norm = np.linalg.norm(x_curr[6:10])
        x_curr[6:10] = x_curr[6:10] / q_norm
        
    print("Simulation Complete.")
    plot_results(total_steps, X_history, U_history, global_path, mpc.Ts, fail_idx)

def plot_results(N, X, U, ref_path, dt, fail_idx=None):
    t = np.arange(N) * dt
    
    if fail_idx is not None:
        end_idx = fail_idx
    else:
        valid_idx = np.where(X[:, 6] != 0)[0]
        if len(valid_idx) == 0: return
        end_idx = valid_idx[-1] + 1
        
    t = t[:end_idx]
    X = X[:end_idx]
    U = U[:end_idx]
    
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    px, py, pz = X[:, 0], X[:, 1], -X[:, 2]
    
    # Only plot the reference path up to where the plane actually flew
    dists = np.linalg.norm(ref_path - X[-1, 0:3], axis=1)
    final_ref_idx = np.argmin(dists) + 50 
    ref_x = ref_path[:final_ref_idx, 0]
    ref_y = ref_path[:final_ref_idx, 1]
    ref_z = -ref_path[:final_ref_idx, 2]

    ax1.plot(ref_x, ref_y, ref_z, 'g--', label='Reference', alpha=0.7)
    ax1.plot(px, py, pz, 'y-', label='Actual MPC', linewidth=2)
    
    step = max(1, len(t) // 30)
    for k in range(0, len(t), step):
        pos = np.array([px[k], py[k], pz[k]])
        R = quat_to_R(X[k, 6:10])
        x_vec, y_vec, z_vec = R @ [1, 0, 0], R @ [0, 1, 0], R @ [0, 0, 1]
        x_vec[2], y_vec[2], z_vec[2] = -x_vec[2], -y_vec[2], -z_vec[2]
        
        scale_triad = 3.0
        ax1.quiver(*pos, *(x_vec * scale_triad), color='r', arrow_length_ratio=0.1)
        ax1.quiver(*pos, *(y_vec * scale_triad), color='g', arrow_length_ratio=0.1)
        ax1.quiver(*pos, *(z_vec * scale_triad), color='b', arrow_length_ratio=0.1)

    if fail_idx is not None and fail_idx > 0:
        fail_pos = X[-1, 0:3] 
        ax1.scatter(fail_pos[0], fail_pos[1], -fail_pos[2], color='red', s=200, marker='X', label='Failure Point', zorder=10)

    ax1.plot([], [], [], color='r', label='Body X (Fwd)')
    ax1.plot([], [], [], color='g', label='Body Y (Right)')
    ax1.plot([], [], [], color='b', label='Body Z (Down)')

    max_range = np.array([px.max()-px.min(), py.max()-py.min(), pz.max()-pz.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (px.max()+px.min()) * 0.5, (py.max()+py.min()) * 0.5, (pz.max()+pz.min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    ax1.set_xlabel('North (m)')
    ax1.set_ylabel('East (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('MPC Spatial Path Tracking')
    ax1.legend()
    
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t, U[:, 0], 'k')
    ax2.set_ylabel('Thrust')
    ax2.grid(True)
    
    ax3 = fig.add_subplot(3, 2, 4, sharex=ax2)
    ax3.plot(t, U[:, 1], 'b')
    ax3.set_ylabel('Aileron (rad)')
    ax3.grid(True)
    
    ax4 = fig.add_subplot(3, 2, 6, sharex=ax2)
    ax4.plot(t, U[:, 2], 'r')
    ax4.set_ylabel('Elevator (rad)')
    ax4.set_xlabel('Time (s)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()