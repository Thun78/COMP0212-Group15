import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Physical constants
MU_E = 3.986004418e14          # Earth GM
MU_S = 1.32712440018e20        # Sun GM
MU_M = 4.9048695e12            # Moon GM

R_EARTH = 6.371e6              # Earth radius (m)
R0_AU = 1.496e11               # 1 AU (m)

# Fixed directions of sun and moon with realistic magnitudes
R_SUN = np.array([1.496e11, 0.0])       # ~1 AU along +x
R_MOON = np.array([384400e3, 0.0])      # ~384,400 km along +x

# Non-grav parameters
C_D = 2.2
RHO_0 = 1.0e-12
SIGMA_RHO = 0.10

C_R = 1.5
P_SUN = 4.56e-6
SIGMA_P = 0.05

# Area, mass model
MASS_INIT = 100.0
A_M_INIT = 0.01
AREA_CONSTANT = A_M_INIT * MASS_INIT  # A fixed

# Jump parameters
LAMBDA = 4.0e-5
SIGMA_J = 10.0
P_M_MIN = 0.01
P_M_MAX = 0.40
MASS_LOSS_ON_JUMP = False

# Simulation parameters
DT = 10.0
TIME_MAX = 100000.0
STEPS = int(TIME_MAX / DT)
N_RUNS = 100
N_PLOTS_RANDOM = 3

# Initial velocity observation error
V0_SIGMA = 150.0

# Deterministic accelerations
def agrav_third_body_2d(r_vec: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(r_vec)
    if r == 0.0:
        return np.zeros(2)

    # Earth central gravity
    a_E = -MU_E * r_vec / (r**3)

    # Sun third-body perturbation
    rs_minus_r = R_SUN - r_vec
    a_S = MU_S * (rs_minus_r / (np.linalg.norm(rs_minus_r)**3) - R_SUN / (np.linalg.norm(R_SUN)**3))

    # Moon third-body perturbation
    rm_minus_r = R_MOON - r_vec
    a_M = MU_M * (rm_minus_r / (np.linalg.norm(rm_minus_r)**3) - R_MOON / (np.linalg.norm(R_MOON)**3))

    return a_E + a_S + a_M

def adrag_rho0_2d(v_vec: np.ndarray, m_current: float) -> np.ndarray:
    v = np.linalg.norm(v_vec)
    if v == 0.0 or m_current <= 0.0:
        return np.zeros(2)
    A_over_m = AREA_CONSTANT / m_current
    return -0.5 * C_D * A_over_m * RHO_0 * v * v_vec

def a_srp0_2d(r_vec: np.ndarray, m_current: float) -> np.ndarray:
    if m_current <= 0.0:
        return np.zeros(2)

    A_over_m = AREA_CONSTANT / m_current

    rel = R_SUN - r_vec
    d = np.linalg.norm(rel)
    if d == 0.0:
        return np.zeros(2)

    direction = rel / d
    scale = (R0_AU / d) ** 2
    return C_R * P_SUN * A_over_m * scale * direction

# Drift RHS for RK4: dr = v, dv = a_det
def drift_rhs(state: np.ndarray, m_current: float) -> np.ndarray:
    x, y, vx, vy = state
    r_vec = np.array([x, y], dtype=float)
    v_vec = np.array([vx, vy], dtype=float)

    r = np.linalg.norm(r_vec)
    if r <= R_EARTH or m_current <= 0.0:
        return np.zeros(4)

    a_det = agrav_third_body_2d(r_vec) + adrag_rho0_2d(v_vec, m_current) + a_srp0_2d(r_vec, m_current)
    return np.array([vx, vy, a_det[0], a_det[1]])

def rk4_step(state: np.ndarray, dt: float, m_current: float) -> np.ndarray:
    k1 = drift_rhs(state, m_current)
    k2 = drift_rhs(state + 0.5 * dt * k1, m_current)
    k3 = drift_rhs(state + 0.5 * dt * k2, m_current)
    k4 = drift_rhs(state + dt * k3, m_current)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Monte Carlo simulation
def run_monte_carlo_simulation(initial_state_nominal, run_id):
    print(f"\n--- Running object {run_id} simulation (N={N_RUNS}) ---")

    impact_count = 0
    jump_count = 0
    trajectories = []
    random_process_data = []

    x0, y0, vx0_nom, vy0_nom = initial_state_nominal

    for run in range(N_RUNS):
        is_data_run = (run < N_PLOTS_RANDOM)
        if is_data_run:
            run_data = {
                "run_id": run + 1,
                "time": [],
                "dv_diff_x": [],
                "dv_diff_y": [],
                "jump_time": []
            }

        # Initial observation error sampling
        vx0 = np.random.normal(vx0_nom, V0_SIGMA)
        vy0 = np.random.normal(vy0_nom, V0_SIGMA)

        state = np.array([x0, y0, vx0, vy0], dtype=float)
        m = MASS_INIT

        traj = [state.copy()]
        is_impact = False

        for step in range(STEPS):
            t = step * DT
            r = np.linalg.norm(state[0:2])

            # termination
            if r <= R_EARTH or m <= 0.0:
                is_impact = (r <= R_EARTH)
                break

            # 1) Deterministic RK4 step
            next_state = rk4_step(state, DT, m)

            # 2) Diffusion (Euler–Maruyama)
            r_for_noise = next_state[0:2]
            v_for_noise = next_state[2:4]

            a_drag0 = adrag_rho0_2d(v_for_noise, m)
            a_srp0 = a_srp0_2d(r_for_noise, m)

            xi = np.random.normal(0.0, 1.0)  # scalar
            dV_diff = (SIGMA_RHO * a_drag0 + SIGMA_P * a_srp0) * xi * np.sqrt(DT)

            next_state[2:4] += dV_diff

            if is_data_run:
                run_data["time"].append(t)
                run_data["dv_diff_x"].append(dV_diff[0])
                run_data["dv_diff_y"].append(dV_diff[1])

            # 3) Poisson jump
            if np.random.rand() < LAMBDA * DT:
                jump_count += 1

                Jk = np.random.normal(0.0, SIGMA_J, size=2)
                next_state[2:4] += Jk

                if MASS_LOSS_ON_JUMP:
                    shed_ratio = np.random.uniform(P_M_MIN, P_M_MAX)
                    m -= m * shed_ratio

                if is_data_run:
                    run_data["jump_time"].append(t)

            # update
            state = next_state
            traj.append(state.copy())

            # escape check (same heuristic as your 3D code)
            r_vec = state[0:2]
            v_vec = state[2:4]
            if np.linalg.norm(r_vec) > 8.0e7 and np.dot(r_vec, v_vec) > 0:
                break

        if is_impact:
            impact_count += 1

        trajectories.append({"data": np.array(traj), "impact": is_impact})
        if is_data_run:
            random_process_data.append(run_data)

    return (
        {
            "ID": run_id,
            "N_Runs": N_RUNS,
            "Impact_Count": impact_count,
            "Probability (%)": 100.0 * impact_count / N_RUNS,
            "Total_Jump_Events": jump_count,
        },
        trajectories,
        random_process_data,
    )

# Random initial conditions
def generate_random_initial_conditions_LEO_2d(h_min=160e3, h_max=2000e3, ecc_max=0.05):
    h = np.random.uniform(h_min, h_max)
    r = R_EARTH + h

    theta = np.random.uniform(0.0, 2*np.pi)
    x0 = r * np.cos(theta)
    y0 = r * np.sin(theta)

    r_hat = np.array([np.cos(theta), np.sin(theta)])

    # random tangential direction
    t_hat = np.array([-r_hat[1], r_hat[0]]) if np.random.rand() < 0.5 else np.array([r_hat[1], -r_hat[0]])

    v_circ = np.sqrt(MU_E / r)
    v_tan = v_circ * np.random.uniform(0.98, 1.02)
    v_rad = np.random.uniform(-ecc_max, ecc_max) * v_circ

    v_vec = v_tan * t_hat + v_rad * r_hat
    return [x0, y0, v_vec[0], v_vec[1]]

# Plotting
def plot_object_analysis_2d(obj_id, trajectories, impact_prob, random_process_data):
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0])
    earth = plt.Circle((0, 0), 1.0, color="blue", alpha=0.5)
    ax1.add_patch(earth)

    impact_count = 0
    start_point_plotted = False

    for tr in trajectories:
        data = tr["data"]
        x = data[:, 0] / R_EARTH
        y = data[:, 1] / R_EARTH

        color = "red" if tr["impact"] else "green"
        ax1.plot(x, y, color=color, alpha=0.3)

        if tr["impact"]:
            impact_count += 1

        if not start_point_plotted:
            ax1.plot(x[0], y[0], "ko")
            start_point_plotted = True

    ax1.set_title(
        f"Object {obj_id} Trajectories (2D)\n"
        f"N={len(trajectories)}, Impacts={impact_count}, P={impact_prob:.1f}%"
    )
    ax1.set_aspect("equal")
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1])
    colors = plt.cm.plasma(np.linspace(0, 1, len(random_process_data)))

    for i, run_data in enumerate(random_process_data):
        t_hr = np.array(run_data["time"]) / 3600.0
        ax2.plot(t_hr, run_data["dv_diff_x"], "--", color=colors[i], alpha=0.7)
        ax2.plot(t_hr, run_data["dv_diff_y"], "-", color=colors[i], alpha=0.7)
        for tj in np.array(run_data["jump_time"]) / 3600.0:
            ax2.axvline(tj, color="red", linestyle=":", alpha=0.5)

    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("ΔV_diff per step (m/s)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

# Main
N_OBJECTS = 5
all_results = {}
conditions = {}

for i in range(1, N_OBJECTS + 1):
    obj_id = f"O{i}"
    conditions[obj_id] = generate_random_initial_conditions_LEO_2d()

for obj_id, init_state in conditions.items():
    result, trajs, rp = run_monte_carlo_simulation(init_state, obj_id)
    all_results[obj_id] = result
    plot_object_analysis_2d(obj_id, trajs, result["Probability (%)"], rp)

df = pd.DataFrame(all_results).T
print("\n=======================================")
print("Monte Carlo Impact Statistics (2D, Section 2-aligned)")
print("=======================================")
print(df)
print("=======================================")
