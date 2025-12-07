"""
Minimal TLE → COEs → propagation (no plotting)

- Convert TLE scalar fields (epoch, i, Ω, e, ω, M, n) to classical orbital elements
- Pretty-print COEs in the requested format
- Build poliastro Orbit objects at epoch
- Propagate and print a compact table (UTC, Δt, |r|, |v|, altitude)
- Also prints UTC epoch and Julian Date

Requires: numpy, astropy, poliastro
"""

from math import pi, sin, cos, atan2, sqrt
from datetime import datetime, timedelta, timezone
import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

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

def tle_epoch_to_time(epoch_field):
    """
    Convert TLE epoch YYDDD.DDDDDDD -> (datetime UTC, astropy Time, JD).
    Example: 25316.14507882 -> 2025-11-12T03:28:54.810048Z
    """
    yy = int(epoch_field // 1000)
    doy = epoch_field - yy*1000
    year = (2000 + yy) if yy < 57 else (1900 + yy)
    day_int = int(doy)
    frac_day = doy - day_int
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_int-1,
                                                               seconds=frac_day*SEC_PER_DAY)
    t = Time(dt)
    return dt, t, float(t.jd)

# -------------------------
# TLE fields -> COEs + pretty print
# -------------------------
def tle_fields_to_coe_pretty(name, epoch_field, inc_deg, raan_deg, e, argp_deg, M_deg, n_rev_per_day, mu=MU_EARTH):
    """
    Returns (payload dict, poliastro Orbit at epoch).
    Prints COEs in your requested layout plus epoch and JD.
    """
    # Epoch
    dt_epoch, t_epoch, jd = tle_epoch_to_time(epoch_field)
    print(f"\n== {name} ==")
    print(f"Epoch (UTC): {dt_epoch.isoformat()}   |   Julian Date: {jd:.8f}")

    # Mean motion rev/day -> rad/s
    n_rad_s = n_rev_per_day * 2*pi / SEC_PER_DAY
    # Semi-major axis from Kepler's third law
    a_km = (mu / (n_rad_s**2))**(1/3)

    # Mean anomaly -> true anomaly
    M_rad = M_deg * pi/180.0
    E = solve_kepler(M_rad, e)
    sin_v = sqrt(1 - e*e) * sin(E) / (1 - e*cos(E))
    cos_v = (cos(E) - e)       / (1 - e*cos(E))
    v = atan2(sin_v, cos_v)
    nu_deg = (v * 180.0/pi) % 360.0

    # Specific angular momentum
    hMag = sqrt(mu * a_km * (1 - e*e))

    # Pretty print
    print("\n--- Classical Orbital Elements ---")
    print(f"Specific Angular Momentum (h): {hMag:12.4f} km²/s")
    print(f"Semi-Major Axis (a):           {a_km:12.4f} km")
    print(f"Eccentricity (e):              {e:12.6f}")
    print(f"True Anomaly (ν):              {nu_deg:12.4f}°")
    print(f"Inclination (i):               {inc_deg:12.4f}°")
    print(f"RAAN (Ω):                      {raan_deg:12.4f}°")
    print(f"Argument of Periapsis (ω):     {argp_deg:12.4f}°")
    print("----------------------------------\n")

    payload = {
        "a_km": a_km,
        "e": e,
        "i_deg": inc_deg,
        "raan_deg": raan_deg,
        "argp_deg": argp_deg,
        "M_deg": M_deg,
        "true_anomaly_deg": nu_deg,
        "n_rev_per_day": n_rev_per_day,
        "epoch_jd": jd,
        "epoch_iso": dt_epoch.isoformat()
    }

    # Build poliastro orbit (using ν at epoch)
    orb = Orbit.from_classical(
        Earth,
        a_km * u.km,
        e * u.one,
        inc_deg * u.deg,
        raan_deg * u.deg,
        argp_deg * u.deg,
        nu_deg * u.deg,
        epoch=t_epoch,
    )
    return payload, orb

# -------------------------
# Propagation table
# -------------------------
def propagate_and_print(orbit, hours=24, step_minutes=60):
    """
    Propagate the orbit forward for 'hours' with 'step_minutes' resolution.
    Prints: UTC, Δt (min), |r| (km), |v| (km/s), altitude (km).
    """
    from numpy.linalg import norm

    steps = int((hours * 60) // step_minutes) + 1
    print(f"--- Propagation: {hours} hours, step {step_minutes} min ---")
    print(f"{'UTC Time':25s} {'Δt (min)':>9s} {'|r| (km)':>12s} {'|v| (km/s)':>12s} {'Altitude (km)':>14s}")
    print("-" * 76)

    for k in range(steps):
        dt_min = k * step_minutes
        tof = TimeDelta(dt_min * 60, format="sec")
        orb_k = orbit.propagate(tof)
        r = orb_k.r.to(u.km).value
        v = orb_k.v.to(u.km/u.s).value
        r_norm = norm(r)
        v_norm = norm(v)
        alt = r_norm - R_EARTH
        tstamp = (orbit.epoch + tof).to_datetime(timezone=timezone.utc)
        print(f"{tstamp.strftime('%Y-%m-%d %H:%M:%S'):25s} {dt_min:9.0f} {r_norm:12.3f} {v_norm:12.6f} {alt:14.3f}")
    print("-" * 76 + "\n")

# -------------------------
# Example usage: four objects (fill with your own as needed)
# -------------------------
if __name__ == "__main__":
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

    # Compute and propagate each object
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
        # Example: 12 hours, 30-minute step table
        propagate_and_print(orb, hours=12, step_minutes=30)
