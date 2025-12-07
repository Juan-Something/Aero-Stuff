"""
Four-TLE end-to-end script (updated):
- Parse TLE-like scalar fields (epoch, i, Ω, e, ω, M, n)
- Compute and print COEs in your format, return a payload dict
- Build poliastro Orbit objects at the given epochs
- Plot 2D (ECI X–Y) and 3D overlays of all orbits
- Animate objects moving along their orbits (2D & 3D) using scatter markers

Requires: numpy, matplotlib, astropy, poliastro
Optional (to save animations): ffmpeg installed on system PATH
"""

from math import pi, sin, cos, atan2, sqrt
from datetime import datetime, timedelta, timezone
import json
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.time import Time, TimeDelta
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from matplotlib import animation

# -------------------------
# Constants
# -------------------------
MU_EARTH = 398600.4418      # km^3/s^2
SEC_PER_DAY = 86400.0
R_EARTH = 6378.1363         # km

# -------------------------
# Utilities
# -------------------------
def solve_kepler(M, e, tol=1e-12, max_iter=50):
    """Solve M = E - e sinE for E (radians) via Newton."""
    M = (M + pi) % (2*pi) - pi
    E = M if e < 0.8 else pi
    for _ in range(max_iter):
        f = E - e*sin(E) - M
        fp = 1 - e*cos(E)
        dE = -f/fp
        E += dE
        if abs(dE) < tol:
            break
    return E

def tle_epoch_to_astropy_time(epoch_field):
    """
    Convert TLE epoch (YYDDD.DDDDDDD) -> astropy Time (UTC).
    Example: 25316.14507882 -> 2025-11-12T03:28:54.810048Z
    """
    yy = int(epoch_field // 1000)
    doy = epoch_field - yy*1000
    year = (2000 + yy) if yy < 57 else (1900 + yy)
    day_int = int(doy)
    frac_day = doy - day_int
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_int-1,
                                                               seconds=frac_day*SEC_PER_DAY)
    return Time(dt)

# -------------------------
# COE computation + pretty print (returns the payload and orbit)
# -------------------------
def tle_fields_to_coe_pretty(name, epoch_field, inc_deg, raan_deg, e, argp_deg, M_deg, n_rev_per_day, mu=MU_EARTH):
    # mean motion (rev/day) -> rad/s
    n_rad_s = n_rev_per_day * 2*pi / SEC_PER_DAY
    # semi-major axis
    a_km = (mu / (n_rad_s**2))**(1/3)
    # Kepler solve for ν
    M_rad = M_deg * pi/180.0
    E = solve_kepler(M_rad, e)
    sin_v = sqrt(1 - e*e) * sin(E) / (1 - e*cos(E))
    cos_v = (cos(E) - e)       / (1 - e*cos(E))
    v = atan2(sin_v, cos_v)
    nu_deg = (v * 180.0/pi) % 360.0
    # specific angular momentum
    hMag = sqrt(mu * a_km * (1 - e*e))

    # Print in requested layout
    print(f"\n== {name} ==")
    print("\n--- Classical Orbital Elements ---")
    print(f"Specific Angular Momentum (h): {hMag:12.4f} km²/s")
    print(f"Semi-Major Axis (a):           {a_km:12.4f} km")
    print(f"Eccentricity (e):              {e:12.6f}")
    print(f"True Anomaly (ν):              {nu_deg:12.4f}°")
    print(f"Inclination (i):               {inc_deg:12.4f}°")
    print(f"RAAN (Ω):                      {raan_deg:12.4f}°")
    print(f"Argument of Periapsis (ω):     {argp_deg:12.4f}°")
    print("----------------------------------")

    payload = {
        "a_km": a_km,
        "e": e,
        "i_deg": inc_deg,
        "raan_deg": raan_deg,
        "argp_deg": argp_deg,
        "M_deg": M_deg,
        "true_anomaly_deg": nu_deg,
        "n_rev_per_day": n_rev_per_day
    }

    print("\n--- Returned Payload ---")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("------------------------\n")

    # Build poliastro orbit at epoch
    t_epoch = tle_epoch_to_astropy_time(epoch_field)
    orb = Orbit.from_classical(
        Earth,
        payload["a_km"] * u.km,
        payload["e"] * u.one,
        payload["i_deg"] * u.deg,
        payload["raan_deg"] * u.deg,
        payload["argp_deg"] * u.deg,
        payload["true_anomaly_deg"] * u.deg,
        epoch=t_epoch,
    )
    return payload, orb

# -------------------------
# Static plotting helpers (2D & 3D)
# -------------------------
def _R3(th):
    return np.array([[ np.cos(th), -np.sin(th), 0.0],
                     [ np.sin(th),  np.cos(th), 0.0],
                     [ 0.0,         0.0,        1.0]])
def _R1(th):
    return np.array([[1.0, 0.0,        0.0       ],
                     [0.0, np.cos(th), -np.sin(th)],
                     [0.0, np.sin(th),  np.cos(th)]])

def ellipse_xy_from_payload(payload, samples=720):
    """Full Keplerian ellipse sampled 0..360 deg in ECI, returned X,Y arrays (km)."""
    a = payload["a_km"]; e = payload["e"]
    i = np.deg2rad(payload["i_deg"]); RAAN = np.deg2rad(payload["raan_deg"]); w = np.deg2rad(payload["argp_deg"])
    p = a * (1 - e**2)
    Q = _R3(RAAN) @ _R1(i) @ _R3(w)
    nus = np.linspace(0.0, 2*pi, samples, endpoint=True)
    xs, ys = [], []
    for nu in nus:
        r_mag = p / (1 + e*np.cos(nu))
        r_pqw = np.array([r_mag*np.cos(nu), r_mag*np.sin(nu), 0.0])
        r_eci = Q @ r_pqw
        xs.append(r_eci[0]); ys.append(r_eci[1])
    return np.array(xs), np.array(ys)

def _ellipse_xyz_from_payload(payload, samples=720):
    """Full Keplerian ellipse sampled 0..360 deg in ECI, returned X,Y,Z arrays (km)."""
    a = payload["a_km"]; e = payload["e"]
    i = np.deg2rad(payload["i_deg"]); RAAN = np.deg2rad(payload["raan_deg"]); w = np.deg2rad(payload["argp_deg"])
    p = a * (1 - e**2)
    Q = _R3(RAAN) @ _R1(i) @ _R3(w)
    nus = np.linspace(0.0, 2*pi, samples, endpoint=True)
    r_xyz = np.zeros((3, samples))
    for k, nu in enumerate(nus):
        r_mag = p / (1 + e*np.cos(nu))
        r_pqw = np.array([r_mag*np.cos(nu), r_mag*np.sin(nu), 0.0])
        r_xyz[:, k] = Q @ r_pqw
    return r_xyz[0], r_xyz[1], r_xyz[2]

def plot_eci_xy_overlaid(orbits_payloads, samples=720, title="ECI X–Y view of orbits"):
    plt.figure()
    theta = np.linspace(0, 2*pi, 360)
    plt.plot(R_EARTH*np.cos(theta), R_EARTH*np.sin(theta), label="Earth (equatorial)")
    for name, payload in orbits_payloads.items():
        xs, ys = ellipse_xy_from_payload(payload, samples=samples)
        plt.plot(xs, ys, label=name)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X (km)"); plt.ylabel("Y (km)")
    plt.title(title)
    plt.grid(True); plt.legend()
    plt.show()

def plot_eci_3d_overlaid(orbits_payloads, samples=720, title="Orbits in ECI (3D)", elev=25, azim=35):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Earth wireframe
    u = np.linspace(0, 2*pi, 120); v = np.linspace(0, pi, 60)
    xe = R_EARTH * np.outer(np.cos(u), np.sin(v))
    ye = R_EARTH * np.outer(np.sin(u), np.sin(v))
    ze = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xe, ye, ze, linewidth=0.3, alpha=0.6, label="Earth")

    lim_extent = R_EARTH
    for name, payload in orbits_payloads.items():
        x, y, z = _ellipse_xyz_from_payload(payload, samples=samples)
        ax.plot(x, y, z, label=name)
        lim_extent = max(lim_extent, np.max(np.abs(np.concatenate([x, y, z]))))

    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_zlabel("Z (km)")
    ax.set_title(title); ax.grid(True); ax.legend(loc="upper right")

    lim = lim_extent * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    try: ax.set_box_aspect([1, 1, 1])
    except Exception: pass
    ax.view_init(elev=elev, azim=azim)
    plt.show()

# -------------------------
# Animation helpers (2D & 3D) using poliastro propagation (scatter-based)
# -------------------------
def _sample_full_ellipse_xyz(payload, samples=720):
    return _ellipse_xyz_from_payload(payload, samples=samples)

def _precompute_positions(orbits_dict, hours=6, step_minutes=10):
    """
    Propagate each Orbit, store xyz over time.
    Returns: times (list[datetime]), pos (dict name -> (x,y,z)), max_extent
    """
    n_frames = int((hours*60)//step_minutes) + 1
    pos = {}
    times = []
    max_extent = R_EARTH

    # time stamps from the first orbit's epoch
    first_epoch = next(iter(orbits_dict.values())).epoch
    for k in range(n_frames):
        tof = TimeDelta(k*step_minutes*60, format="sec")
        times.append((first_epoch + tof).to_datetime(timezone=timezone.utc))

    for name, orb in orbits_dict.items():
        xs, ys, zs = [], [], []
        for k in range(n_frames):
            tof = TimeDelta(k*step_minutes*60, format="sec")
            ok = orb.propagate(tof)
            r = ok.r.to(u.km).value
            xs.append(r[0]); ys.append(r[1]); zs.append(r[2])
        xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
        pos[name] = (xs, ys, zs)
        max_extent = max(max_extent, np.max(np.abs(np.concatenate([xs, ys, zs]))))
    return times, pos, max_extent

def animate_eci_3d(all_payloads, all_orbits, hours=6, step_minutes=10, samples_ellipse=360, elev=25, azim=35):
    """
    3D animation: static wireframe Earth + static orbit ellipses + moving markers (scatter).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Earth wireframe
    u = np.linspace(0, 2*pi, 120); v = np.linspace(0, pi, 60)
    xe = R_EARTH*np.outer(np.cos(u), np.sin(v))
    ye = R_EARTH*np.outer(np.sin(u), np.sin(v))
    ze = R_EARTH*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xe, ye, ze, linewidth=0.3, alpha=0.6, label="Earth")

    # Static ellipses
    for name, payload in all_payloads.items():
        ex, ey, ez = _sample_full_ellipse_xyz(payload, samples=samples_ellipse)
        ax.plot(ex, ey, ez, label=name)

    # Precompute propagated positions
    times, pos, max_extent = _precompute_positions(all_orbits, hours=hours, step_minutes=step_minutes)

    # Moving markers (scatter)
    scatters = {}
    for name in all_orbits.keys():
        xs, ys, zs = pos[name]
        h = ax.scatter([xs[0]], [ys[0]], [zs[0]], s=20)  # Path3DCollection
        scatters[name] = (h, xs, ys, zs)

    time_txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    lim = max_extent * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    try: ax.set_box_aspect([1,1,1])
    except Exception: pass
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_zlabel("Z (km)")
    ax.set_title("Objects traveling along orbits (ECI 3D)")
    ax.legend(loc="upper right"); ax.grid(True)
    ax.view_init(elev=elev, azim=azim)

    def _update(frame):
        for name, (h, xs, ys, zs) in scatters.items():
            # update 3D scatter: use protected _offsets3d as matplotlib lacks public setter
            h._offsets3d = ([xs[frame]], [ys[frame]], [zs[frame]])
        time_txt.set_text(times[frame].strftime("%Y-%m-%d %H:%M:%S UTC"))
        return [h for h,_x,_y,_z in scatters.values()] + [time_txt]

    ani = animation.FuncAnimation(fig, _update, frames=len(times), interval=200, blit=False, repeat=True)
    plt.show()
    return ani

def animate_eci_xy_2d(all_payloads, all_orbits, hours=6, step_minutes=10, samples_ellipse=360):
    """
    2D animation (X–Y plane): static Earth + static ellipses + moving markers (scatter).
    """
    fig, ax = plt.subplots()
    th = np.linspace(0, 2*pi, 360)
    ax.plot(R_EARTH*np.cos(th), R_EARTH*np.sin(th), label="Earth (equatorial)")

    max_extent = R_EARTH
    for name, payload in all_payloads.items():
        xs, ys, _ = _sample_full_ellipse_xyz(payload, samples=samples_ellipse)
        ax.plot(xs, ys, label=name)
        max_extent = max(max_extent, np.max(np.abs(np.concatenate([xs, ys]))))

    times, pos, _ = _precompute_positions(all_orbits, hours=hours, step_minutes=step_minutes)

    # Moving markers (scatter)
    scatters = {}
    for name in all_orbits.keys():
        xs, ys, zs = pos[name]
        h = ax.scatter([xs[0]], [ys[0]], s=20)  # PathCollection
        scatters[name] = (h, xs, ys)

    ax.set_aspect("equal", adjustable="box")
    lim = max_extent * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
    ax.set_title("Objects traveling along orbits (ECI X–Y)")
    ax.legend(); ax.grid(True)
    time_txt = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def _update(frame):
        for name, (h, xs, ys) in scatters.items():
            # set_offsets expects shape (N, 2)
            h.set_offsets(np.c_[[xs[frame]], [ys[frame]]])
        time_txt.set_text(times[frame].strftime("%Y-%m-%d %H:%M:%S UTC"))
        return [h for h,_x,_y in scatters.values()] + [time_txt]

    ani = animation.FuncAnimation(fig, _update, frames=len(times), interval=200, blit=False, repeat=True)
    plt.show()
    return ani

# -------------------------
# Inputs: four objects (parsed TLE-like fields)
# -------------------------
objects = [
    {
        "name": "INTELSAT 33E DEB (61998)",
        "epoch": 25316.14507882,
        "inc": 0.9979,
        "raan": 85.7856,
        "ecc": 0.0009167,
        "argp": 252.6559,
        "M": 21.5383,
        "n": 1.00437626,
    },
    {
        "name": "FREGAT R/B (39192)",
        "epoch": 25316.66227286,
        "inc": 0.0773,
        "raan": 183.5260,
        "ecc": 0.0008031,
        "argp": 266.3417,
        "M": 270.2279,
        "n": 5.20983481235749,
    },
    {
        "name": "OBJECT 52939U 22072E",
        "epoch": 25316.14146735,
        "inc": 9.9523,
        "raan": 276.8703,
        "ecc": 0.0026188,
        "argp": 221.3777,
        "M": 138.4452,
        "n": 15.20682587186371,
    },
    {
        "name": "OBJECT 33393U 08048A",
        "epoch": 25316.56274537,
        "inc": 9.3468,
        "raan": 320.0996,
        "ecc": 0.0009937,
        "argp": 135.6367,
        "M": 224.4630,
        "n": 14.952780489390428,
    },
]

# -------------------------
# Main: compute payloads & orbits, plot, and animate
# -------------------------
if __name__ == "__main__":
    all_payloads = {}
    all_orbits = {}

    for obj in objects:
        payload, orb = tle_fields_to_coe_pretty(
            name=obj["name"],
            epoch_field=obj["epoch"],
            inc_deg=obj["inc"],
            raan_deg=obj["raan"],
            e=obj["ecc"],
            argp_deg=obj["argp"],
            M_deg=obj["M"],
            n_rev_per_day=obj["n"],
        )
        all_payloads[obj["name"]] = payload
        all_orbits[obj["name"]] = orb

    # 2D static overlay
    plot_eci_xy_overlaid(all_payloads, samples=720, title="Four orbits (ECI X–Y projection)")

    # 3D static overlay (adjust view if desired)
    plot_eci_3d_overlaid(all_payloads, samples=720, title="Four orbits in ECI (3D)", elev=30, azim=40)

    # Animated propagation
    ani3d = animate_eci_3d(all_payloads, all_orbits, hours=720, step_minutes=15, elev=35, azim=40)
    # ani2d = animate_eci_xy_2d(all_payloads, all_orbits, hours=12, step_minutes=15)

    # To save animations instead of just showing:
    # ani3d.save("orbits_3d.mp4", fps=10)   # requires ffmpeg
    # ani2d.save("orbits_xy.mp4", fps=10)   # requires ffmpeg
