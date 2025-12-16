import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import argparse
import os

# Physical Constants
MU = 3.986e14
R_EARTH = 6.371e6
R0_AU_SQ = (1.496e11)**2

# Simulation Parameters
DT = 10.0
TIME_MAX = 60000
STEPS = int(TIME_MAX / DT)
N_RUNS = 100
N_PLOTS_RANDOM = 3

V0_SIGMA = 150.0

# Disturbance Parameters
C_D = 2.2
RHO_NOMINAL = 1.0e-12
SIGMA_RHO = 0.10
C_R = 1.5
P_SUN = 4.56e-6
SIGMA_P = 0.05
R_SUN_VEC = np.array([0.0, -1.0])


# Fragment Shedding Parameters
LAMBDA_SHEDDING = 4.0e-5
MASS_INIT = 100.0
A_M_INIT = 0.01
AREA_CONSTANT = A_M_INIT * MASS_INIT

SEP_VEL_SIGMA = 10
P_M_MIN = 0.01
P_M_MAX = 0.40


# Equations of Motion
def equations_of_motion(state, m_current):
    x, y, vx, vy = state
    r_vec = np.array([x, y])
    v_vec = np.array([vx, vy])
    r = np.linalg.norm(r_vec)

    if r <= R_EARTH:
        return np.zeros(4)

    A_M_current = AREA_CONSTANT / m_current

    a_grav = -MU * r_vec / r**3

    v_mag = np.linalg.norm(v_vec)
    a_drag = -0.5 * C_D * A_M_current * RHO_NOMINAL * v_mag * v_vec

    a_SRP = C_R * P_SUN * A_M_current * (1 / R0_AU_SQ) * R_SUN_VEC

    a_total = a_grav + a_drag + a_SRP

    return np.array([vx, vy, a_total[0], a_total[1]])

def rk4_step(state, dt, m):
    k1 = equations_of_motion(state, m)
    k2 = equations_of_motion(state + 0.5 * dt * k1, m)
    k3 = equations_of_motion(state + 0.5 * dt * k2, m)
    k4 = equations_of_motion(state + dt * k3, m)
    return state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)


# Monte Carlo Simulation (with impact recording)
def run_monte_carlo_simulation(initial_state, run_id):
    impact_count = 0
    shedding_count = 0
    impact_points = []

    x0, y0, vx0, vy0, inc0, raan0 = initial_state

    for run in range(N_RUNS):
        vx = np.random.normal(vx0, V0_SIGMA)
        vy = np.random.normal(vy0, V0_SIGMA)

        state = np.array([x0, y0, vx, vy])
        m = MASS_INIT

        for step in range(STEPS):
            x, y, vx, vy = state
            r = np.linalg.norm([x, y])

            if r <= R_EARTH:
                impact_count += 1
                t_hit = step * DT
                impact_points.append((np.array([x, y]), inc0, raan0, t_hit))
                break

            state = rk4_step(state, DT, m)

            # SDE perturbation term (drag-induced diffusion)
            dW = np.random.normal(0, 1, size=2) * np.sqrt(DT)
            a_stoch = SIGMA_RHO * np.array([vx, vy]) * dW
            state[2:4] += a_stoch

            # Poisson jump process: instantaneous velocity change only (no explicit fragmentation tracking)
            if np.random.rand() < LAMBDA_SHEDDING * DT:
                shedding_count += 1
                dm = m * np.random.uniform(P_M_MIN, P_M_MAX)
                m -= dm
                state[2:4] += np.random.normal(0, SEP_VEL_SIGMA, size=2)

    impact_probability = 100 * impact_count / N_RUNS

    result_data = {
        'ID': run_id,
        'N_Runs': N_RUNS,
        'Impact_Count': impact_count,
        'Probability (%)': impact_probability,
        'Total_Shedding_Events': shedding_count
    }

    return result_data, impact_points


# Initial Condition Generation (GLOBAL + EQUATOR-BIASED)
def generate_random_initial_conditions(
    h_min=160e3, h_max=2000e3,   # altitude range (m)
    ecc_max=0.05, equator_bias=0.6,
    inc_equator_deg=40.0                # radial velocity fraction of v_circ
):
    # 1) random altitude
    h = np.random.uniform(h_min, h_max)
    r = R_EARTH + h

    # 2) random position angle (uniform on circle)
    theta = np.random.uniform(0.0, 2.0 * np.pi)

    x0 = r * np.cos(theta)
    y0 = r * np.sin(theta)

    r_hat = np.array([np.cos(theta), np.sin(theta)])

    # 3) unbiased tangential direction (CW or CCW)
    if np.random.rand() < 0.5:
        t_hat = np.array([-r_hat[1],  r_hat[0]])  # +90°
    else:
        t_hat = np.array([ r_hat[1], -r_hat[0]])  # -90°

    # 4) near-circular speed + small radial component
    v_circ = np.sqrt(MU / r)

    v_tan = v_circ * np.random.uniform(1.0 - 0.02, 1.0 + 0.02)
    v_rad = np.random.uniform(-ecc_max, ecc_max) * v_circ

    v_vec = v_tan * t_hat + v_rad * r_hat

    # Inclination-weighted sampling: equatorial band vs global distribution
    if np.random.rand() < equator_bias:
        inc = np.random.uniform(-inc_equator_deg, inc_equator_deg) * np.pi / 180.0
    else:
        inc = np.random.uniform(-90.0, 90.0) * np.pi / 180.0

    raan = np.random.uniform(0, 2 * np.pi)

    return [x0, y0, v_vec[0], v_vec[1], inc, raan]


# Helper: convert impact points to latitude-longitude

def impact_points_to_latlon(impact_points):
    omega_earth = 7.2921150e-5  # rad/s
    lats, lons = [], []
    for item in impact_points:
        pos2d, inc, raan, t_hit = item
        x2, y2 = pos2d[0], pos2d[1]

        r_orb = np.array([x2, y2, 0.0])

        # Rotate by inclination (x-axis) and then by RAAN (z-axis)
        cos_i = np.cos(inc); sin_i = np.sin(inc)
        Rx = np.array([[1, 0, 0],
                       [0, cos_i, -sin_i],
                       [0, sin_i,  cos_i]])
        cos_r = np.cos(raan); sin_r = np.sin(raan)
        Rz = np.array([[ cos_r, -sin_r, 0],
                       [ sin_r,  cos_r, 0],
                       [     0,      0, 1]])

        r_eci = Rz.dot(Rx.dot(r_orb))

        # Earth rotation from ECI to ECEF
        theta_e = omega_earth * t_hit
        cos_te = np.cos(-theta_e); sin_te = np.sin(-theta_e)
        Rte = np.array([[ cos_te, -sin_te, 0],
                        [ sin_te,  cos_te, 0],
                        [     0,       0, 1]])
        r_ecef = Rte.dot(r_eci)

        x_e, y_e, z_e = r_ecef
        rnorm = np.linalg.norm(r_ecef)
        if rnorm == 0:
            continue
        lat = np.degrees(np.arcsin(z_e / rnorm))
        lon = np.degrees(np.arctan2(y_e, x_e))
        lats.append(lat); lons.append(lon)
    return np.array(lats), np.array(lons)

# Plot 1: Longitude-Latitude Scatter (No Base Map)
def plot_impact_points_only(impact_points, save=False, outfn='impact_points_lonlat.png'):
    if len(impact_points) == 0:
        print(" No impact data")
        return
    lats, lons = impact_points_to_latlon(impact_points)

    plt.figure(figsize=(12, 6))
    plt.scatter(lons, lats, c='tab:red', s=18, edgecolors='k', linewidths=0.3)
    plt.xlim(-180, 180); plt.ylim(-90, 90)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title("Impact Locations (Lon–Lat Scatter, No Base Map)")
    plt.grid(alpha=0.35)
    if save:
        plt.savefig(outfn, dpi=150, bbox_inches='tight')
        print(f"Saved point scatter to {outfn}")
    else:
        plt.show()

# Plot 2: Heatmap Overlaid on Real Earth Map
def plot_impact_heatmap_map(impact_points, bins=(72, 36), save=False, outfn='heatmap_map.png'):
    if len(impact_points) == 0:
        print("No impact data")
        return

    lats, lons = impact_points_to_latlon(impact_points)

    # 2D histogram: x = longitude, y = latitude
    heatmap, xedges, yedges = np.histogram2d(
        lons, lats, bins=bins, range=[[-180, 180], [-90, 90]]
    )

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.stock_img()
    ax.coastlines(linewidth=0.7)

    lon_centers = (xedges[:-1] + xedges[1:]) / 2
    lat_centers = (yedges[:-1] + yedges[1:]) / 2

    mesh = ax.pcolormesh(
        lon_centers, lat_centers, heatmap.T,
        cmap='hot', alpha=0.6, shading="auto",
        transform=ccrs.PlateCarree()
    )
    plt.colorbar(mesh, ax=ax, label='Impact Count')
    plt.title("Earth Impact Heatmap (Projected on Real-World Map)")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    if save:
        plt.savefig(outfn, dpi=150, bbox_inches='tight')
        print(f"Saved map heatmap to {outfn}")
    else:
        plt.show()

# =========================
# Main Program
# =========================
if __name__ == '__main__':
    N_ASTEROIDS = 5
    all_results = []
    all_impact_points = []

    for i in range(1, N_ASTEROIDS + 1):
        ast_id = f"A{i}"
        initial_state = generate_random_initial_conditions(
            equator_bias=0.7,     # Equatorial band weighting (higher = denser near equator)
            inc_equator_deg=30.0  # Half-width of equatorial band (30°)
        )
        result, impact_points = run_monte_carlo_simulation(initial_state, ast_id)
        all_results.append(result)
        all_impact_points.extend(impact_points)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true',
                        help='Save figures to files instead of displaying them')
    parser.add_argument('--out_points', type=str,
                        default='impact_points_lonlat.png',
                        help='Output filename for lon-lat scatter (no map)')
    parser.add_argument('--out_map', type=str,
                        default='heatmap_map.png',
                        help='Output filename for heatmap projected on map')
    parser.add_argument('--bins_lon', type=int, default=72,
                        help='Number of longitude bins for the heatmap')
    parser.add_argument('--bins_lat', type=int, default=36,
                        help='Number of latitude bins for the heatmap')
    args = parser.parse_args()

    # Output statistics
    df = pd.DataFrame(all_results).set_index("ID")
    print("\n Impact Probability Statistics:")
    print(df)

    # Plot 1: Scatter plot (no map)
    plot_impact_points_only(all_impact_points, save=args.save, outfn=args.out_points)

    # Plot 2: Heatmap projected on Earth map
    plot_impact_heatmap_map(
        all_impact_points,
        bins=(args.bins_lon, args.bins_lat),
        save=args.save,
        outfn=args.out_map
    )
