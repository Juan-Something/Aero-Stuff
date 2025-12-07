# -*- coding: utf-8 -*-
"""
Mission plan (no plotting), RAAN-aware, with Kepler propagation (two-body, no perturbations)

CHANGE: The hop between the two LEO objects uses a "plane-match at high apogee"
bi-elliptic step followed by a COPLANAR Lambert rendezvous. Other legs remain
Hohmann-with-plane-rotation Burn #1 + Kepler phasing + dwell 5 periods.

Sequence:
  INTELSAT 33E DEB (start, on-orbit dwell) →
  FREGAT R/B →
  OBJECT 33393U 08048A →
  OBJECT 52939U 22072E

Requires: numpy, astropy
"""

from math import pi, sqrt, sin, cos
from datetime import datetime, timedelta, timezone
import numpy as np
from astropy.time import Time, TimeDelta

# -------------------------
# Constants
# -------------------------
MU = 398600.4418          # km^3/s^2
SEC_PER_DAY = 86400.0

# -------------------------
# Angle & Kepler helpers
# -------------------------
def wrap2pi(x): return np.mod(x, 2*np.pi)
def wrap_pi(x): return (x + np.pi) % (2*np.pi) - np.pi

def solve_kepler_E_from_M(M, e, tol=1e-12, max_iter=60):
    M = (M + np.pi) % (2*np.pi) - np.pi
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f  = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        dE = -f/fp
        E += dE
        if abs(dE) < tol: break
    return E

def M_to_nu(M, e):
    E = solve_kepler_E_from_M(M, e)
    sv = np.sqrt(1-e**2)*np.sin(E)/(1-e*np.cos(E))
    cv = (np.cos(E)-e)/(1-e*np.cos(E))
    return np.arctan2(sv, cv)

# -------------------------
# Geometry / rotations
# -------------------------
def plane_angle_deg(i1_deg, Omega1_deg, i2_deg, Omega2_deg):
    i1, i2 = np.deg2rad([i1_deg, i2_deg])
    dO = np.deg2rad((Omega2_deg - Omega1_deg + 180) % 360 - 180)
    cth = np.cos(i1)*np.cos(i2) + np.sin(i1)*np.sin(i2)*np.cos(dO)
    return float(np.rad2deg(np.arccos(np.clip(cth, -1.0, 1.0))))

def R3(th):
    return np.array([[ np.cos(th), -np.sin(th), 0.0],
                     [ np.sin(th),  np.cos(th), 0.0],
                     [ 0.0,         0.0,        1.0]])

def R1(th):
    return np.array([[1.0, 0.0,        0.0       ],
                     [0.0, np.cos(th), -np.sin(th)],
                     [0.0, np.sin(th),  np.cos(th)]])

def Q_eci_from_pqw(raan_deg, inc_deg, argp_deg):
    Ω = np.deg2rad(raan_deg); i = np.deg2rad(inc_deg); ω = np.deg2rad(argp_deg)
    return R3(Ω) @ R1(i) @ R3(ω)

# -------------------------
# Burns / periods
# -------------------------
def v_circ(r): return sqrt(MU / r)
def period_from_a(a_km): return 2*np.pi*np.sqrt((a_km**3)/MU)

def dv_combined_burn(v_before, v_after, plane_angle_deg_value):
    di = np.deg2rad(plane_angle_deg_value)
    return sqrt(v_before*v_before + v_after*v_after - 2*v_before*v_after*cos(di))

def dv_leg_with_plane_rotation_at_burn1(r1, r2, plane_angle_deg_value):
    at = 0.5*(r1+r2)
    v1 = v_circ(r1)
    vt1 = sqrt(MU*(2/r1 - 1/at))
    dv1 = dv_combined_burn(v1, vt1, plane_angle_deg_value)
    v2 = v_circ(r2)
    vt2 = sqrt(MU*(2/r2 - 1/at))
    dv2 = abs(v2 - vt2)
    tof_half = np.pi * np.sqrt(at**3 / MU)
    return dv1, dv2, dv1+dv2, tof_half

# -------------------------
# TLE-like target with Kepler propagation
# -------------------------
def tle_epoch_to_Time(epoch_field):
    yy = int(epoch_field // 1000)
    doy = epoch_field - yy*1000
    year = (2000 + yy) if yy < 57 else (1900 + yy)
    day_int = int(doy)
    frac_day = doy - day_int
    dt = datetime(year,1,1,tzinfo=timezone.utc) + timedelta(days=day_int-1,
                                                            seconds=frac_day*SEC_PER_DAY)
    return Time(dt)

class TargetKepler:
    def __init__(self, name, epoch_yydoy, inc_deg, raan_deg, e, argp_deg, M_deg, n_rev_per_day):
        self.name = name
        self.epoch = tle_epoch_to_Time(epoch_yydoy)
        self.i_deg = float(inc_deg)
        self.raan_deg = float(raan_deg)
        self.e = float(e)
        self.argp_deg = float(argp_deg)
        self.M0_deg = float(M_deg)
        n_rad_s = n_rev_per_day * 2*np.pi / SEC_PER_DAY
        self.a_km = (MU / (n_rad_s**2))**(1/3)
        self.n_rad_s = np.sqrt(MU / self.a_km**3)

    def M_at(self, t: Time) -> float:
        dt = (t - self.epoch).to_value("sec")
        return wrap2pi(np.deg2rad(self.M0_deg) + self.n_rad_s * dt)

    def nu_at(self, t: Time) -> float:
        return M_to_nu(self.M_at(t), self.e)

    def u_at(self, t: Time) -> float:
        return wrap2pi(np.deg2rad(self.argp_deg) + self.nu_at(t))

    def r_eci(self, t: Time) -> np.ndarray:
        ν = self.nu_at(t)
        p = self.a_km * (1 - self.e**2)
        rmag = p / (1 + self.e*np.cos(ν))
        r_pqw = np.array([rmag*np.cos(ν), rmag*np.sin(ν), 0.0])
        Q = Q_eci_from_pqw(self.raan_deg, self.i_deg, self.argp_deg)
        return Q @ r_pqw

    def v_eci(self, t: Time) -> np.ndarray:
        ν = self.nu_at(t)
        a, e = self.a_km, self.e
        p = a*(1 - e**2)
        h = np.sqrt(MU*p)
        v_pqw = np.array([-np.sin(ν), e + np.cos(ν), 0.0]) * (MU/h)
        Q = Q_eci_from_pqw(self.raan_deg, self.i_deg, self.argp_deg)
        return Q @ v_pqw

    def period(self) -> float:
        return 2*np.pi / self.n_rad_s

# -------------------------
# Chaser circular state in given plane at u
# -------------------------
def chaser_circular_rv(a_km, i_deg, raan_deg, u_rad, mu=MU):
    rmag = a_km
    Q = Q_eci_from_pqw(raan_deg, i_deg, 0.0)
    r_pqw = np.array([rmag*np.cos(u_rad), rmag*np.sin(u_rad), 0.0])
    t_pqw = np.array([-np.sin(u_rad), np.cos(u_rad), 0.0])
    r_eci = Q @ r_pqw
    v_eci = v_circ(rmag) * (Q @ t_pqw)
    return r_eci, v_eci

# -------------------------
# Universal-variable Lambert (0-rev)
# -------------------------
def stumpff_C(z):
    if z > 0:
        sz = np.sqrt(z); return (1 - np.cos(sz)) / z
    if z < 0:
        sz = np.sqrt(-z); return (1 - np.cosh(sz)) / z
    return 0.5

def stumpff_S(z):
    if z > 0:
        sz = np.sqrt(z); return (sz - np.sin(sz)) / (sz**3)
    if z < 0:
        sz = np.sqrt(-z); return (np.sinh(sz) - sz) / (sz**3)
    return 1.0/6.0

def lambert_universal(r1_vec, r2_vec, tof, mu=MU, prograde=True, max_iter=60, tol=1e-9):
    r1 = np.linalg.norm(r1_vec); r2 = np.linalg.norm(r2_vec)
    cos_dth = np.clip(np.dot(r1_vec, r2_vec)/(r1*r2), -1.0, 1.0)
    dth = np.arccos(cos_dth)
    if prograde:
        if np.cross(r1_vec, r2_vec)[2] < 0:
            dth = 2*np.pi - dth
    else:
        dth = 2*np.pi - dth
    if abs(dth) < 1e-12: raise ValueError("Lambert: Δθ≈0.")
    A = np.sin(dth) * np.sqrt(r1*r2/(1 - np.cos(dth)))
    if abs(A) < 1e-12: raise ValueError("Lambert: A≈0.")
    z = 0.0
    for _ in range(max_iter):
        C = stumpff_C(z); S = stumpff_S(z)
        y = r1 + r2 + A*(z*S - 1.0)/np.sqrt(C)
        if y < 0.0: z += 0.1; continue
        chi = np.sqrt(y/C)
        tof_z = (chi**3 * S + A*np.sqrt(y)) / np.sqrt(mu)
        if abs(tof_z - tof) < tol: break
        # Simple secant step if derivative unstable
        z += 0.1 if (tof_z < tof) else -0.1
    C = stumpff_C(z); S = stumpff_S(z)
    y = r1 + r2 + A*(z*S - 1.0)/np.sqrt(C)
    if y < 0: raise RuntimeError("Lambert failed: y<0.")
    f = 1.0 - y/r1
    g = A*np.sqrt(y/mu)
    gdot = 1.0 - y/r2
    v1 = (r2_vec - f*r1_vec)/g
    v2 = (gdot*r2_vec - r1_vec)/g
    return v1, v2

# -------------------------
# Kepler phasing rendezvous at target circle (non-Lambert legs)
# -------------------------
def plan_phasing_to_rendezvous_kepler(r_circ, t_start: Time, tracker: TargetKepler,
                                      max_orbits_search=24, drift_span_pct=0.05):
    u_ref = tracker.u_at(t_start)
    v_c = v_circ(r_circ)
    best = None
    for N in range(1, max_orbits_search+1):
        for s in np.linspace(-drift_span_pct, drift_span_pct, 121):
            a_d = r_circ * (1.0 + s)
            if a_d <= 0: continue
            n_d = np.sqrt(MU / a_d**3)
            T_d = 2*np.pi / n_d
            t_d = N*T_d
            t_int = t_start + TimeDelta(t_d, format="sec")
            du = abs(wrap_pi(tracker.u_at(t_int) - u_ref))
            at = 0.5*(r_circ + a_d)
            vt1 = sqrt(MU*(2/r_circ - 1/at))
            dv_out = abs(vt1 - v_c)
            v_back_entry = sqrt(MU*(2/a_d - 1/at))
            dv_back = abs(v_c - v_back_entry)
            dv_tot = dv_out + dv_back
            score = (du, dv_tot)
            if (best is None) or (score < best["score"]):
                best = dict(N_drift=N, a_drift=a_d, dv_out=dv_out, dv_back=dv_back,
                            dv_total=dv_tot, t_intercept=t_int, score=score)
    if best is None:
        raise RuntimeError("Phasing search failed.")
    best.pop("score", None)
    return best

# -------------------------
# Standard leg: transfer (plane-rotation Hohmann) → phasing → dwell
# -------------------------
def leg_transfer_then_rendezvous_then_dwell(t_now: Time,
                                            r_start, i_start_deg, Omega_start_deg,
                                            tracker: TargetKepler,
                                            leg_label: str):
    r_tgt = tracker.a_km
    i_tgt = tracker.i_deg
    Omega_tgt = tracker.raan_deg

    theta_deg = plane_angle_deg(i_start_deg, Omega_start_deg, i_tgt, Omega_tgt)

    if abs(r_start - r_tgt) < 1e-6 and theta_deg < 1e-9:
        t_arrive = t_now
        dv_leg = dv1 = dv2 = 0.0
    else:
        dv1, dv2, dv_leg, tof_half = dv_leg_with_plane_rotation_at_burn1(r_start, r_tgt, theta_deg)
        t_arrive = t_now + TimeDelta(tof_half, format="sec")

    Omega_after = Omega_tgt
    i_after = i_tgt

    ph = plan_phasing_to_rendezvous_kepler(r_tgt, t_arrive, tracker, max_orbits_search=24, drift_span_pct=0.05)
    dv_phase = ph["dv_total"]
    t_int = ph["t_intercept"]

    T = period_from_a(r_tgt)
    t_after_dwell = t_int + TimeDelta(5*T, format="sec")

    def fmt_time(t): return t.to_datetime(timezone=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n=== {leg_label} ===")
    print(f"Start time                     : {fmt_time(t_now)}")
    print(f"Start (r,i,Ω)                  : ({r_start:,.1f} km, {i_start_deg:.3f}°, {Omega_start_deg:.3f}°)")
    print(f"Target (r,i,Ω,ω,e)             : ({r_tgt:,.1f} km, {i_tgt:.3f}°, {Omega_tgt:.3f}°, {tracker.argp_deg:.3f}°, {tracker.e:.5f})")
    if dv_leg > 0:
        print(f"Plane-rotation angle θ         : {theta_deg:.3f}°  (Δi + ΔΩ)")
        print(f"Transfer Burn #1 (incl+RAAN)   : {dv1:7.3f} km/s")
        print(f"Transfer Burn #2 (circularize) : {dv2:7.3f} km/s")
        print(f"Δv transfer subtotal           : {dv_leg:7.3f} km/s")
        print(f"Arrival (TOF ~ half xfer)      : {fmt_time(t_arrive)}")
    else:
        print("No radius/plane transfer; proceeding to phasing rendezvous.")
    print("Phasing rendezvous (Kepler):")
    print(f"  Drift orbits (N)             : {ph['N_drift']}")
    print(f"  Drift semi-major axis        : {ph['a_drift']:,.1f} km")
    print(f"  Phasing burns (out/back)     : {ph['dv_out']:.3f} + {ph['dv_back']:.3f} = {dv_phase:.3f} km/s")
    print(f"  Intercept time               : {fmt_time(t_int)}")
    print(f"Dwell 5 periods                : {timedelta(seconds=5*T)}  → until {fmt_time(t_after_dwell)}")

    dv_total = dv_leg + dv_phase
    return t_after_dwell, i_after, Omega_after, dv_total

# -------------------------
# NEW: Plane-match at high apogee → coplanar Lambert → dwell (use for LEO↔LEO leg)
# -------------------------
def leg_plane_match_bielliptic_then_lambert(
    t_now: Time,
    a_start, i_start_deg, Omega_start_deg,   # current circular
    tgt: TargetKepler,                       # arrival target
    leg_label: str,
    r_ap_scale=4.0,                          # apogee multiplier (3–6 is a good scan range)
    tof_scan_scales=(0.4, 1.6, 31),          # TOF scan around a short baseline
    prograde=True
):
    # Radii
    r1 = a_start
    r2 = tgt.a_km
    r_ap = r_ap_scale * max(r1, r2)

    # Plane angle to match (Δi + ΔΩ)
    theta_deg = plane_angle_deg(i_start_deg, Omega_start_deg, tgt.i_deg, tgt.raan_deg)
    theta = np.deg2rad(theta_deg)

    # Raise to apogee: ellipse E1 (rp=r1, ra=r_ap), burn at perigee
    a1 = 0.5 * (r1 + r_ap)
    v_c1 = v_circ(r1)
    v_peri_E1 = np.sqrt(MU * (2/r1 - 1/a1))
    dv_raise = abs(v_peri_E1 - v_c1)
    tof_leg1 = np.pi * np.sqrt(a1**3 / MU)
    t_ap = t_now + TimeDelta(tof_leg1, format="sec")

    # Speed at apogee on E1
    v_ap_E1 = np.sqrt(MU * (2/r_ap - 1/a1))

    # Switch at apogee to ellipse E2 (rp=r2, ra=r_ap) AND rotate plane to target plane
    a2 = 0.5 * (r2 + r_ap)
    v_ap_E2 = np.sqrt(MU * (2/r_ap - 1/a2))
    dv_plane_combo = np.sqrt(v_ap_E1**2 + v_ap_E2**2 - 2*v_ap_E1*v_ap_E2*np.cos(theta))
    tof_leg2 = np.pi * np.sqrt(a2**3 / MU)
    t_perigee_r2 = t_ap + TimeDelta(tof_leg2, format="sec")  # now in target plane, at r2

    # Now coplanar with target at r2: short Lambert to exact rendezvous
    u_dep = tgt.u_at(t_perigee_r2)  # align departure point with target’s u
    r1_vec, v1_circ = chaser_circular_rv(r2, tgt.i_deg, tgt.raan_deg, u_dep)

    # Baseline TOF (short)
    tof0 = 0.5 * np.pi * np.sqrt(r2**3 / MU)
    lo, hi, Ns = tof_scan_scales
    best = None
    for s in np.linspace(lo, hi, Ns):
        tof = max(60.0, s * tof0)
        t_arr = t_perigee_r2 + TimeDelta(tof, format="sec")
        r2_vec = tgt.r_eci(t_arr)
        v2_tar = tgt.v_eci(t_arr)
        try:
            v1_tr, v2_tr = lambert_universal(r1_vec, r2_vec, tof, mu=MU, prograde=prograde)
        except Exception:
            continue
        dv_dep = np.linalg.norm(v1_tr - v1_circ)
        dv_arr = np.linalg.norm(v2_tar - v2_tr)
        dv_tot = dv_dep + dv_arr
        if (best is None) or (dv_tot < best[0]):
            best = (dv_tot, tof, t_arr)

    if best is None:
        raise RuntimeError("Lambert scan (coplanar) failed.")
    dv_lam, tof_best, t_arrival = best

    # Dwell 5 periods on the target
    T = tgt.period()
    t_after_dwell = t_arrival + TimeDelta(5*T, format="sec")

    # After the leg we carry the target’s plane
    i_after = tgt.i_deg
    Omega_after = tgt.raan_deg

    def fmt(t): return t.to_datetime(timezone=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n=== {leg_label} (Plane-match @ apogee → coplanar Lambert) ===")
    print(f"Start time                     : {fmt(t_now)}")
    print(f"Start circular (r,i,Ω)         : ({r1:,.1f} km, {i_start_deg:.3f}°, {Omega_start_deg:.3f}°)")
    print(f"Target (r,i,Ω,ω,e)             : ({r2:,.1f} km, {tgt.i_deg:.3f}°, {tgt.raan_deg:.3f}°, {tgt.argp_deg:.3f}°, {tgt.e:.5f})")
    print(f"Plane angle to match           : {theta_deg:.3f}°")
    print(f"Chosen apogee                  : r_ap = {r_ap:,.1f} km  (scale={r_ap_scale:.2f})")
    print(f"Raise-to-apogee dv             : {dv_raise:7.3f} km/s  (to apogee at {fmt(t_ap)})")
    print(f"Apogee combined dv (plane+ra)  : {dv_plane_combo:7.3f} km/s")
    print(f"Arrive perigee r2              : {fmt(t_perigee_r2)}")
    print(f"Lambert (coplanar) dv          : {dv_lam:7.3f} km/s  (TOF={timedelta(seconds=tof_best)})")
    print(f"Δv leg subtotal                : {dv_raise + dv_plane_combo + dv_lam:7.3f} km/s")
    print(f"Arrival time                   : {fmt(t_arrival)}")
    print(f"Dwell 5 periods                : {timedelta(seconds=5*T)}  → until {fmt(t_after_dwell)}")

    dv_total = dv_raise + dv_plane_combo + dv_lam
    return t_after_dwell, i_after, Omega_after, dv_total

# -------------------------
# Inputs (four objects)
# -------------------------
objects = [
    {"name": "INTELSAT 33E DEB (61998)", "epoch": 25316.14507882, "inc": 0.9979, "raan": 85.7856,
     "ecc": 0.0009167, "argp": 252.6559, "M": 21.5383, "n": 1.00437626},
    {"name": "FREGAT R/B (39192)", "epoch": 25316.66227286, "inc": 0.0773, "raan": 183.5260,
     "ecc": 0.0008031, "argp": 266.3417, "M": 270.2279, "n": 5.20983481235749},
    {"name": "OBJECT 33393U 08048A", "epoch": 25316.56274537, "inc": 9.3468, "raan": 320.0996,
     "ecc": 0.0009937, "argp": 135.6367, "M": 224.4630, "n": 14.952780489390428},
    {"name": "OBJECT 52939U 22072E", "epoch": 25316.14146735, "inc": 9.9523, "raan": 276.8703,
     "ecc": 0.0026188, "argp": 221.3777, "M": 138.4452, "n": 15.20682587186371},
]

# Build trackers; sort by descending a and ensure INTELSAT first
trackers = [TargetKepler(o["name"], o["epoch"], o["inc"], o["raan"], o["ecc"],
                         o["argp"], o["M"], o["n"]) for o in objects]
trackers.sort(key=lambda t: t.a_km, reverse=True)
for i, t in enumerate(trackers):
    if "INTELSAT 33E DEB" in t.name:
        if i != 0:
            trackers.insert(0, trackers.pop(i))
        break

LEO_A = "OBJECT 33393U 08048A"
LEO_B = "OBJECT 52939U 22072E"

# -------------------------
# Run sequence
# -------------------------
if __name__ == "__main__":
    start_time = max(t.epoch for t in trackers)
    start = trackers[0]

    print("=== MISSION START (RAAN-aware; Kepler; LEO↔LEO plane-match+Lambert) ===")
    print(f"Common start epoch             : {start_time.to_datetime(timezone=timezone.utc)} UTC")
    print(f"Start object                   : {start.name}")
    print(f"Start orbit (a,i,Ω,ω,e)        : ({start.a_km:,.1f} km, {start.i_deg:.3f}°, "
          f"{start.raan_deg:.3f}°, {start.argp_deg:.3f}°, {start.e:.5f})")

    total_dv = 0.0
    cur_t = start_time
    cur_r = start.a_km
    cur_i = start.i_deg
    cur_Omega = start.raan_deg

    # Leg 0: rendezvous at INTELSAT (phasing-only placeholder) + dwell
    cur_t, cur_i, cur_Omega, dv0 = leg_transfer_then_rendezvous_then_dwell(
        cur_t, cur_r, cur_i, cur_Omega, start, f"Rendezvous at {start.name} (starting orbit)"
    )
    total_dv += dv0
    cur_r = start.a_km

    # Remaining legs; use plane-match+Lambert for the LEO↔LEO hop
    for k in range(1, len(trackers)):
        prev = trackers[k-1]
        tgt  = trackers[k]
        leg_label = f"{prev.name} → {tgt.name}"

        if (prev.name == LEO_A and tgt.name == LEO_B) or (prev.name == LEO_B and tgt.name == LEO_A):
            cur_t, cur_i, cur_Omega, dv_leg = leg_plane_match_bielliptic_then_lambert(
                cur_t, cur_r, cur_i, cur_Omega, tgt, leg_label,
                r_ap_scale=4.0,                   # tune 3–6 as needed
                tof_scan_scales=(0.4, 1.6, 41),   # scan TOF to cut Δv
                prograde=True
            )
        else:
            cur_t, cur_i, cur_Omega, dv_leg = leg_transfer_then_rendezvous_then_dwell(
                cur_t, cur_r, cur_i, cur_Omega, tgt, leg_label
            )
        total_dv += dv_leg
        cur_r = tgt.a_km

    print("\n=== MISSION SUMMARY ===")
    print(f"Total Δv (all legs; no margins): {total_dv:.3f} km/s")
    print(f"Completion time (UTC)          : {cur_t.to_datetime(timezone=timezone.utc)}")
