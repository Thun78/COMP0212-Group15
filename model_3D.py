import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D 

# Physical constants
MU_E = 3.986004418e14          # Sun GM 
MU_S = 1.32712440018e20        # Sun GM 
MU_M = 4.9048695e12            # Moon GM 

R_EARTH = 6.371e6              # Earth radius

# Fixed directions of sun and moon with realistic magnitudes
R_SUN = np.array([1.496e11, 0.0, 0.0])        # ~1 AU along +X (m)
R_MOON = np.array([384400e3, 0.0, 0.0])       # ~384,400 km along +X (m)

# Non-grav parameters
C_D = 2.2
RHO_0 = 1.0e-12                 # nominal density rho0 (kg/m^3)
SIGMA_RHO = 0.10                # relative density fluctuation intensity

C_R = 1.5
P_SUN = 4.56e-6                 # solar radiation pressure at 1 AU (N/m^2)
R0_AU = 1.496e11                # 1 AU (m)
SIGMA_P = 0.05                  # SRP fluctuation intensity sigma_P 

# Area / mass model
MASS_INIT = 100.0
A_M_INIT = 0.01                 # (A/m) at t0
AREA_CONSTANT = A_M_INIT * MASS_INIT  # A is kept constant, m changes -> A/m varies

# Jump parameters
LAMBDA = 4.0e-5                 # Poisson 
SIGMA_J = 10.0                  # velocity jump std dev
P_M_MIN = 0.01                  # fraction of mass lost per event
P_M_MAX = 0.40

# Simulation parameters
DT = 10.0
TIME_MAX = 10000.0
STEPS = int(TIME_MAX / DT)
N_RUNS = 100
N_PLOTS_RANDOM = 3

# initial velocity observation error
V0_SIGMA = 150.0

def agrav_third_body(r_vec):
    r = np.linalg.norm(r_vec)

    # Earth central gravity
    a_E = -MU_E * r_vec / (r**3)

    # Sun third-body perturbation
    rs_minus_r = R_SUN - r_vec
    a_S = MU_S * (rs_minus_r / (np.linalg.norm(rs_minus_r)**3) - R_SUN / (np.linalg.norm(R_SUN)**3))

    # Moon third-body perturbation
    rm_minus_r = R_MOON - r_vec
    a_M = MU_M * (rm_minus_r / (np.linalg.norm(rm_minus_r)**3) - R_MOON / (np.linalg.norm(R_MOON)**3))

    return a_E + a_S + a_M

# Drag acceleration at nominal density rho0
def adrag_rho0(v_vec, m_current):
    v = np.linalg.norm(v_vec)
    if v == 0.0 or m_current <= 0.0:
        return np.zeros(3)
    A_over_m = AREA_CONSTANT / m_current
    return -0.5 * C_D * A_over_m * RHO_0 * v * v_vec

# Deterministic SRP acceleration a_SRP0
def a_srp0(r_vec, m_current):
    if m_current <= 0.0:
        return np.zeros(3)
    A_over_m = AREA_CONSTANT / m_current

    rel = R_SUN - r_vec
    d = np.linalg.norm(rel)
    if d == 0.0:
        return np.zeros(3)

    direction = rel / d
    scale = (R0_AU / d) ** 2
    return C_R * P_SUN * A_over_m * scale * direction

# Deterministic drift function for RK4:
def drift_rhs(state, m_current):
    x, y, z, vx, vy, vz = state
    r_vec = np.array([x, y, z], dtype=float)
    v_vec = np.array([vx, vy, vz], dtype=float)

    r = np.linalg.norm(r_vec)
    if r <= R_EARTH or m_current <= 0.0:
        return np.zeros(6)

    a = agrav_third_body(r_vec) + adrag_rho0(v_vec, m_current) + a_srp0(r_vec, m_current)
    return np.array([vx, vy, vz, a[0], a[1], a[2]])

def rk4_step(state, dt, m_current):
    k1 = drift_rhs(state, m_current)
    k2 = drift_rhs(state + dt/2 * k1, m_current)
    k3 = drift_rhs(state + dt/2 * k2, m_current)
    k4 = drift_rhs(state + dt * k3, m_current)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# LEO initial conditions 
def generate_random_initial_conditions_LEO(
    h_min=160e3, h_max=2000e3,
    ecc_max=0.05
):
    # 1) Random altitude in LEO
    h = np.random.uniform(h_min, h_max)
    r = R_EARTH + h

    # 2) Uniform position on a sphere
    u = np.random.uniform(-1.0, 1.0)
    phi = np.random.uniform(0.0, 2*np.pi)
    sin_theta = np.sqrt(1.0 - u**2)

    x0 = r * sin_theta * np.cos(phi)
    y0 = r * sin_theta * np.sin(phi)
    z0 = r * u

    r_vec = np.array([x0, y0, z0], dtype=float)
    r_hat = r_vec / np.linalg.norm(r_vec)

    # 3) Unbiased tangential direction (projected random vector)
    rand_vec = np.random.normal(0.0, 1.0, size=3)
    t_hat = rand_vec - np.dot(rand_vec, r_hat) * r_hat
    t_hat /= np.linalg.norm(t_hat)

    # 4) Near-circular orbital speed
    v_circ = np.sqrt(MU_E / r)

    v_tan = v_circ * np.random.uniform(0.98, 1.02)   # ±2%
    v_rad = np.random.uniform(-ecc_max, ecc_max) * v_circ

    v_vec = v_tan * t_hat + v_rad * r_hat

    return [x0, y0, z0, v_vec[0], v_vec[1], v_vec[2]]


# Monte Carlo simulation
def run_monte_carlo_simulation(initial_state_nominal, run_id):
    print(f"\n--- Running object {run_id} simulation (N={N_RUNS}) ---")

    impact_count = 0
    jump_count = 0
    trajectories = []
    random_process_data = []

    x0, y0, z0, vx0_nom, vy0_nom, vz0_nom = initial_state_nominal

    for run in range(N_RUNS):
        is_data_run = (run < N_PLOTS_RANDOM)
        if is_data_run:
            run_data = {
                "run_id": run + 1,
                "time": [],
                "dv_diff_x": [],
                "dv_diff_y": [],
                "dv_diff_z": [],
                "jump_time": []
            }

        # Initial observation error sampling
        vx0 = np.random.normal(vx0_nom, V0_SIGMA)
        vy0 = np.random.normal(vy0_nom, V0_SIGMA)
        vz0 = np.random.normal(vz0_nom, V0_SIGMA)

        state = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=float)
        m = MASS_INIT

        traj = [state.copy()]
        is_impact = False

        for step in range(STEPS):
            t = step * DT
            r_vec = state[0:3]
            v_vec = state[3:6]
            r = np.linalg.norm(r_vec)

            # termination
            if r <= R_EARTH or m <= 0.0:
                is_impact = (r <= R_EARTH)
                break

            # 1) Deterministic RK4 step:
            next_state = rk4_step(state, DT, m)

            # 2) Diffusion (Euler–Maruyama):
            a_drag0 = adrag_rho0(v_vec, m)
            a_srp0_now = a_srp0(r_vec, m)

            dW_rho = np.random.normal(0.0, 1.0, size=3) * np.sqrt(DT)
            dW_P = np.random.normal(0.0, 1.0, size=3) * np.sqrt(DT)

            dV_diff = SIGMA_RHO * a_drag0 * dW_rho + SIGMA_P * a_srp0_now * dW_P

            next_state[3:6] += dV_diff

            if is_data_run:
                run_data["time"].append(t)
                run_data["dv_diff_x"].append(dV_diff[0])
                run_data["dv_diff_y"].append(dV_diff[1])
                run_data["dv_diff_z"].append(dV_diff[2])

            # 3) Poisson jump:
            if np.random.rand() < LAMBDA * DT:
                jump_count += 1

                # velocity jump
                Jk = np.random.normal(0.0, SIGMA_J, size=3)
                next_state[3:6] += Jk

                # mass loss
                shed_ratio = np.random.uniform(P_M_MIN, P_M_MAX)
                m -= m * shed_ratio

                if is_data_run:
                    run_data["jump_time"].append(t)

            # update
            state = next_state
            traj.append(state.copy())

            # escape check
            r_vec = state[0:3]
            v_vec = state[3:6]
            r = np.linalg.norm(r_vec)
            if r > 8.0e7 and np.dot(r_vec, v_vec) > 0:
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

# Plotting
def plot_object_analysis(obj_id, trajectories, impact_prob):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Earth wireframe (unit sphere)
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    xe = np.outer(np.cos(u), np.sin(v))
    ye = np.outer(np.sin(u), np.sin(v))
    ze = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xe, ye, ze, linewidth=0.5, alpha=0.4)

    impact_count = 0
    start_point_plotted = False

    # Extract initial position (from first trajectory)
    x0, y0, z0 = trajectories[0]["data"][0, 0:3]

    for tr in trajectories:
        data = tr["data"]

        x = data[:, 0] / R_EARTH
        y = data[:, 1] / R_EARTH
        z = data[:, 2] / R_EARTH

        color = "red" if tr["impact"] else "green"
        ax.plot(x, y, z, color=color, alpha=0.35, linewidth=1)

        if tr["impact"]:
            impact_count += 1

        # Plot initial point only once
        if not start_point_plotted and len(x) > 0:
            ax.scatter(x[0], y[0], z[0], marker="o")
            start_point_plotted = True

    N_runs = len(trajectories)

    ax.set_title(
        f"3D Trajectory Simulation\n"
        f"N={N_runs}, Impacts={impact_count}, P={impact_prob:.1f}%\n"
        f"Initial Position (m): "
        f"x₀={x0:.2e}, y₀={y0:.2e}, z₀={z0:.2e}",
        fontsize=14
    )

    ax.set_xlabel("X ($R_E$)")
    ax.set_ylabel("Y ($R_E$)")
    ax.set_zlabel("Z ($R_E$)")

    max_R_init = np.max([
        np.linalg.norm(tr["data"][0, 0:3]) for tr in trajectories
    ])
    max_RE = max_R_init / R_EARTH + 1.0

    ax.set_xlim(-max_RE, max_RE)
    ax.set_ylim(-max_RE, max_RE)
    ax.set_zlim(-max_RE, max_RE)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# Main
N_OBJECTS = 5
all_results = []
conditions = {}

for i in range(1, N_OBJECTS + 1):
    obj_id = f"Obj{i}"
    conditions[obj_id] = generate_random_initial_conditions_LEO()

for obj_id, init_state in conditions.items():
    result, trajs, rp = run_monte_carlo_simulation(init_state, obj_id)
    all_results.append(result)
    plot_object_analysis(obj_id, trajs, result["Probability (%)"])

results_df = pd.DataFrame(all_results).set_index("ID")

print("\n=======================================")
print("Jump-Diffusion Debris Monte Carlo Results")
print("=======================================")
init_df = pd.DataFrame(conditions).T
init_df.columns = ["X0 (m)", "Y0 (m)", "Z0 (m)", "Vx0 (m/s)", "Vy0 (m/s)", "Vz0 (m/s)"]
print("\n--- Initial nominal conditions ---")
print(init_df.to_string(float_format="%.2f"))

print("\n--- Monte Carlo stats ---")
print(results_df[["N_Runs", "Impact_Count", "Probability (%)", "Total_Jump_Events"]].to_string(float_format="%.2f"))
print("=======================================")

plt.figure(figsize=(12, 7))
results_df["Probability (%)"].plot(kind="bar")
plt.title(f"Impact Probability Comparison (N={N_RUNS} per object)")
plt.ylabel("Impact Probability (%)")
plt.xlabel("Object ID")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
