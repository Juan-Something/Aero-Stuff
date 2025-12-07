# -*- coding: utf-8 -*-
# Mission plan with geometry-accurate plotting of burns and trajectories.
# Produces:
#   ./plots/*_impulses.png              (Δv vs UTC time, one figure per leg)
#   ./plots/geometry/*_geometry.png     (3D ECI geometry-accurate plots per leg)
#
# Notes:
# - Uses astropy.Time/TimeDelta if available; otherwise falls back to datetime/timedelta.
# - Hohmann+plane legs: Burn 1 rotates velocity into target plane at constant position;
#   transfer ellipse is drawn in the target plane; Burn 2 at the opposite point on target circle.
# - Phasing burns are at the arrival point on the target circle (depart/return at same location).
# - Lambert leg: exact transfer conic drawn by propagating the (r0, v_tr) solution.
# - Leg 3 fix: draw the two coast ellipses (E1 and E2) so Burn 2 lies on-trajectory.

from math import pi, sqrt, sin, cos
from datetime import datetime, timedelta, timezone
import os
import re
import numpy as np

# --- Try astropy time; fall back to datetime if unavailable ---
USE_ASTROPY = True
try:
    from astropy.time import Time, TimeDelta
except Exception:
    USE_ASTROPY = False

# -------------------------
# Constants
# -------------------------
MU = 398600.4418      # km^3/s^2
SEC_PER_DAY = 86400.0

# -------------------------
# Tiny helpers
# -------------------------
def v_circ(r): return sqrt(MU / r)
def period_from_a(a): return 2*pi*sqrt(a**3/MU)

def wrap2pi(x): return np.mod(x, 2*np.pi)
def wrap_pi(x): return (x + np.pi) % (2*np.pi) - np.pi

def solve_E_from_M(M, e, tol=1e-12, it=60):
    M = (M + np.pi) % (2*np.pi) - np.pi
    E = M if e < 0.8 else np.pi
    for _ in range(it):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        dE = -f/fp
        E += dE
        if abs(dE) < tol: break
    return E

def M_to_nu(M, e):
    E = solve_E_from_M(M, e)
    sv = sqrt(1-e**2)*sin(E)/(1-e*cos(E))
    cv = (cos(E)-e)/(1-e*cos(E))
    return np.arctan2(sv, cv)

def tle_epoch_to_Time(epoch_field):
    # Returns astropy.Time if available; otherwise a timezone-aware datetime
    yy = int(epoch_field // 1000)
    doy = epoch_field - yy*1000
    year = (2000 + yy) if yy < 57 else (1900 + yy)
    day_int = int(doy)
    frac_day = doy - day_int
    dt = datetime(year,1,1,tzinfo=timezone.utc) + timedelta(days=day_int-1,
                                                            seconds=frac_day*SEC_PER_DAY)
    return Time(dt) if USE_ASTROPY else dt

def seconds_between(t2, t1):
    if USE_ASTROPY:
        return (t2 - t1).to_value("sec")
    return (t2 - t1).total_seconds()

def add_seconds(t, s):
    if USE_ASTROPY:
        return t + TimeDelta(s, format="sec")
    return t + timedelta(seconds=s)

def to_datetime_utc(t):
    if USE_ASTROPY:
        return t.to_datetime(timezone=timezone.utc)
    return t

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

# Target element accessors (no class)
def target_from_tle_fields(name, epoch, inc, raan, e, argp, Mdeg, n_rev_per_day):
    n_rad_s = n_rev_per_day * 2*np.pi / SEC_PER_DAY
    a = (MU / n_rad_s**2)**(1/3)
    return dict(name=name, epoch=tle_epoch_to_Time(epoch),
                a=a, e=float(e), i=float(inc), Omega=float(raan),
                omega=float(argp), M0=float(Mdeg), n=n_rad_s)

def M_at(tgt, t):  # mean anomaly at time t
    dt = seconds_between(t, tgt["epoch"])
    return wrap2pi(np.deg2rad(tgt["M0"]) + tgt["n"]*dt)

def nu_at(tgt, t): return M_to_nu(M_at(tgt, t), tgt["e"])
def u_at(tgt, t):  return wrap2pi(np.deg2rad(tgt["omega"]) + nu_at(tgt, t))

def r_eci(tgt, t):
    nu = nu_at(tgt, t); a, e = tgt["a"], tgt["e"]
    p = a*(1 - e**2); rmag = p/(1 + e*np.cos(nu))
    r_pqw = np.array([rmag*np.cos(nu), rmag*np.sin(nu), 0.0])
    Q = Q_eci_from_pqw(tgt["Omega"], tgt["i"], tgt["omega"])
    return Q @ r_pqw

def v_eci(tgt, t):
    nu = nu_at(tgt, t); a, e = tgt["a"], tgt["e"]
    p = a*(1 - e**2); h = sqrt(MU*p)
    v_pqw = np.array([-np.sin(nu), e + np.cos(nu), 0.0]) * (MU/h)
    Q = Q_eci_from_pqw(tgt["Omega"], tgt["i"], tgt["omega"])
    return Q @ v_pqw

def chaser_circ_rv(a_km, i_deg, Omega_deg, u_rad):
    Q = Q_eci_from_pqw(Omega_deg, i_deg, 0.0)
    r_pqw = np.array([a_km*np.cos(u_rad), a_km*np.sin(u_rad), 0.0])
    t_pqw = np.array([-np.sin(u_rad), np.cos(u_rad), 0.0])
    return Q @ r_pqw, v_circ(a_km)*(Q @ t_pqw)

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

# -------------------------
# Orbital element helpers and propagation
# -------------------------
def elements_from_rv(r, v, mu=MU):
    r = np.asarray(r); v = np.asarray(v)
    rmag = np.linalg.norm(r); vmag = np.linalg.norm(v)
    h = np.cross(r, v); hmag = np.linalg.norm(h)
    k = np.array([0.0, 0.0, 1.0])
    n = np.cross(k, h); nmag = np.linalg.norm(n)
    e_vec = (np.cross(v, h) / mu) - (r / rmag); e = np.linalg.norm(e_vec)
    inc = np.arccos(np.clip(h[2] / max(hmag,1e-15), -1.0, 1.0))
    Omega = 0.0 if nmag < 1e-12 else np.arctan2(n[1], n[0])
    if nmag < 1e-12 or e < 1e-12:
        omega = 0.0
    else:
        omega = np.arctan2(np.dot(np.cross(n, e_vec), h) / (max(nmag,1e-15) * max(hmag,1e-15)),
                           np.dot(n, e_vec) / (max(nmag,1e-15) * max(e,1e-15)))
    if e < 1e-12:
        if nmag > 0:
            nu = np.arctan2(np.dot(np.cross(n, r), h) / (nmag*hmag*rmag),
                            np.dot(n, r) / (nmag*rmag))
        else:
            nu = np.arctan2(r[1], r[0])
    else:
        nu = np.arctan2(np.dot(np.cross(e_vec, r), h) / (e * hmag),
                        np.dot(e_vec, r) / (e * rmag))
    a = 1.0 / (2.0 / rmag - vmag * vmag / mu)
    return float(a), float(e), float(inc), float(Omega), float(omega), float(wrap_pi(nu))

def E_from_nu(nu, e):
    return 2*np.arctan(np.tan(nu/2) * np.sqrt((1-e)/(1+e)))

def kepler_propagate_positions(a, e, inc, Omega, omega, nu0, tof, npts=600, mu=MU):
    n = np.sqrt(mu / a**3)
    E0 = E_from_nu(nu0, e)
    M0 = E0 - e*np.sin(E0)
    ts = np.linspace(0.0, tof, npts)
    Q = Q_eci_from_pqw(np.rad2deg(Omega), np.rad2deg(inc), np.rad2deg(omega))
    r_list = []
    for t in ts:
        M = M0 + n*t
        E = solve_E_from_M(M, e)
        rmag = a*(1 - e*np.cos(E))
        cosnu = (np.cos(E) - e) / (1 - e*np.cos(E))
        sinnu = (np.sqrt(1-e**2) * np.sin(E)) / (1 - e*np.cos(E))
        r_pqw = np.array([rmag * cosnu, rmag * sinnu, 0.0])
        r_list.append(Q @ r_pqw)
    return np.vstack(r_list)

def align_target_plane_argp_to_vector(i_tgt_deg, Omega_tgt_deg, rdir_eci):
    Q0 = R3(np.deg2rad(Omega_tgt_deg)) @ R1(np.deg2rad(i_tgt_deg))
    p_in_plane = Q0.T @ normalize(rdir_eci)
    omega = np.arctan2(p_in_plane[1], p_in_plane[0])
    return float(omega)

def circle_in_plane_points(a_radius, i_deg, Omega_deg, omega_deg, npts=720):
    Q = Q_eci_from_pqw(Omega_deg, i_deg, omega_deg)
    nus = np.linspace(0, 2*np.pi, npts)
    r_pqw = np.vstack([a_radius*np.cos(nus), a_radius*np.sin(nus), np.zeros_like(nus)]).T
    return (Q @ r_pqw.T).T

# -------------------------
# Burn & geometry instrumentation
# -------------------------
ALL_BURNS = []   # dict(leg, label, time, dv_kms)
GEOM_POINTS = [] # dict(leg, label, r (3,), t)
GEOM_CURVES = {} # dict(leg -> list of (N,3) arrays)

def record_burn(leg, label, t, dv):
    ALL_BURNS.append({"leg": leg, "label": label, "time": t, "dv_kms": float(dv)})

def record_geom_point(leg, label, r_vec, t):
    GEOM_POINTS.append({'leg': leg, 'label': label, 'r': np.asarray(r_vec), 't': t})

def add_curve(leg, curve_xyz):
    GEOM_CURVES.setdefault(leg, []).append(np.asarray(curve_xyz))

# -------------------------
# Universal-variable Lambert (0-rev)
# -------------------------
def C(z):
    if z > 0: sz=np.sqrt(z); return (1-np.cos(sz))/z
    if z < 0: sz=np.sqrt(-z); return (1-np.cosh(sz))/z
    return 0.5
def S(z):
    if z > 0: sz=np.sqrt(z); return (sz-np.sin(sz))/(sz**3)
    if z < 0: sz=np.sqrt(-z); return (np.sinh(sz)-sz)/(sz**3)
    return 1/6

def lambert_universal(r1, r2, tof, mu=MU, prograde=True, it=60, tol=1e-9):
    r1n = np.linalg.norm(r1); r2n = np.linalg.norm(r2)
    cd = np.clip(np.dot(r1,r2)/(r1n*r2n), -1.0, 1.0)
    dth = np.arccos(cd)
    if prograde and (np.cross(r1,r2)[2] < 0): dth = 2*np.pi - dth
    if not prograde: dth = 2*np.pi - dth
    if dth < 1e-12: raise ValueError("Δθ≈0")
    A = np.sin(dth)*np.sqrt(r1n*r2n/(1-np.cos(dth)))
    if abs(A) < 1e-12: raise ValueError("A≈0")
    z = 0.0
    for _ in range(it):
        Cz, Sz = C(z), S(z)
        y = r1n + r2n + A*(z*Sz - 1)/np.sqrt(Cz)
        if y < 0: z += 0.1; continue
        x = np.sqrt(y/Cz)
        tof_z = (x**3*Sz + A*np.sqrt(y))/np.sqrt(mu)
        if abs(tof_z - tof) < tol: break
        z += 0.1 if (tof_z < tof) else -0.1
    Cz, Sz = C(z), S(z)
    y = r1n + r2n + A*(z*Sz - 1)/np.sqrt(Cz)
    f = 1 - y/r1n; g = A*np.sqrt(y/mu); gdot = 1 - y/r2n
    v1 = (r2 - f*r1)/g
    v2 = (gdot*r2 - r1)/g
    return v1, v2

def dv_combined(vb, va, ang_deg):
    """Magnitude of Δv to rotate from speed vb to va with an instantaneous plane change of ang_deg."""
    di = np.deg2rad(ang_deg)
    return sqrt(vb*vb + va*va - 2*vb*va*cos(di))

# -------------------------
# Mission legs (with geometry capture)
# -------------------------
def leg_hohmann_plane_then_phasing(t_now, r_start, i_start, Omega_start, tgt, leg_label):
    r_tgt = tgt["a"]; i_tgt, Omega_tgt = tgt["i"], tgt["Omega"]
    theta = plane_angle_deg(i_start, Omega_start, i_tgt, Omega_tgt)

    # Reproducible point on start circle (align with target u for visuals)
    u_chaser = u_at(tgt, t_now)
    r_now, v_now = chaser_circ_rv(r_start, i_start, Omega_start, u_chaser)

    # Defaults
    dv_leg = 0.0
    t_arr = t_now
    r_arr_circ = normalize(r_now) * r_tgt  # if already co-orbital

    if not (abs(r_start - r_tgt) < 1e-6 and theta < 1e-9):
        # Burn 1 (plane+injection) at r_now
        at = 0.5*(r_start + r_tgt)
        v1 = v_circ(r_start)
        vt1 = sqrt(MU*(2/r_start - 1/at))
        dv1 = dv_combined(v1, vt1, theta)
        record_burn(leg_label, "Transfer+Plane Burn", t_now, dv1)
        record_geom_point(leg_label, "Burn 1", r_now, t_now)

        # Transfer ellipse in target plane, aligned with r_now
        omega_tgt_rad = align_target_plane_argp_to_vector(i_tgt, Omega_tgt, r_now)
        a_t = at
        e_t = (r_tgt - r_start)/(r_tgt + r_start)
        p_t = a_t*(1-e_t**2)
        nus = np.linspace(0, np.pi, 600)
        r_pqw = np.stack([p_t/(1+e_t*np.cos(nus))*np.cos(nus),
                          p_t/(1+e_t*np.cos(nus))*np.sin(nus),
                          np.zeros_like(nus)], axis=1)
        Q_t = Q_eci_from_pqw(Omega_tgt, i_tgt, np.rad2deg(omega_tgt_rad))
        transfer_xyz = (Q_t @ r_pqw.T).T
        add_curve(leg_label, circle_in_plane_points(r_start, i_start, Omega_start, np.rad2deg(u_chaser)))
        add_curve(leg_label, circle_in_plane_points(r_tgt, i_tgt, Omega_tgt, np.rad2deg(omega_tgt_rad)))
        add_curve(leg_label, transfer_xyz)

        # Burn 2 (circularize) opposite r_now on target circle
        r_arr_circ = -normalize(r_now) * r_tgt
        t_arr = add_seconds(t_now, pi*sqrt(a_t**3/MU))
        v2 = v_circ(r_tgt)
        vt2 = sqrt(MU*(2/r_tgt - 1/a_t))
        dv2 = abs(v2 - vt2)
        record_burn(leg_label, "Circularize at Target SMA", t_arr, dv2)
        record_geom_point(leg_label, "Burn 2", r_arr_circ, t_arr)
        dv_leg = dv1 + dv2

    # Phasing (drift near r_tgt)
    v_c = v_circ(r_tgt)
    u_ref = u_at(tgt, t_arr)
    best = None
    best_components = None
    for N in range(1, 25):
        for s in np.linspace(-0.05, 0.05, 41):
            a_d = r_tgt*(1+s); n_d = sqrt(MU/a_d**3)
            T_d = 2*pi/n_d; t_d = N*T_d
            t_int = add_seconds(t_arr, t_d)
            du = abs(wrap_pi(u_at(tgt, t_int) - u_ref))
            at_loc = 0.5*(r_tgt + a_d)
            vt1_loc = sqrt(MU*(2/r_tgt - 1/at_loc))
            dv_out = abs(vt1_loc - v_c)
            v_back_entry = sqrt(MU*(2/a_d - 1/at_loc))
            dv_back = abs(v_c - v_back_entry)
            dv_tot = dv_out + dv_back
            score = (du, dv_tot)
            if (best is None) or (score < best[0]):
                best = (score, dv_tot, t_int, N, s)
                best_components = (dv_out, dv_back)
    dv_phase = best[1]; t_int = best[2]
    dv_out, dv_back = best_components

    # Phasing burns at arrival point on the target circle
    record_burn(leg_label, "Phasing: depart drift orbit", t_arr, dv_out)
    record_burn(leg_label, "Phasing: return to circular", add_seconds(t_int, -1.0), dv_back)
    record_geom_point(leg_label, "Phasing depart/return location", r_arr_circ, t_arr)

    T = period_from_a(r_tgt)
    t_after = add_seconds(t_int, 5*T)
    return t_after, i_tgt, Omega_tgt, dv_leg + dv_phase

def leg_leo_to_leo_plane_apogee_then_lambert(t_now, r_start, i_start, Omega_start, tgt, leg_label):
    r1 = r_start; r2 = tgt["a"]
    r_ap = 4.0 * max(r1, r2)
    theta = plane_angle_deg(i_start, Omega_start, tgt["i"], tgt["Omega"])

    # Plane normals (unit angular-momentum directions)
    def hhat(inc_deg, Omega_deg):
        i = np.deg2rad(inc_deg); O = np.deg2rad(Omega_deg)
        return np.array([-np.sin(O)*np.sin(i), np.cos(O)*np.sin(i), np.cos(i)])
    h1 = hhat(i_start, Omega_start)
    h2 = hhat(tgt["i"], tgt["Omega"])

    # Line of nodes between the two planes
    node_dir = normalize(np.cross(h1, h2))
    if np.linalg.norm(node_dir) < 1e-12:
        # planes already coincident; pick any fixed direction in that plane
        node_dir = normalize(np.cross(h1, np.array([1.0,0.0,0.0])))
        if np.linalg.norm(node_dir) < 1e-12:
            node_dir = normalize(np.cross(h1, np.array([0.0,1.0,0.0])))

    # Burn 1: perigee raise to apogee (perigee on -node_dir, apogee on +node_dir)
    a1 = 0.5*(r1+r_ap)
    dv_raise = abs(sqrt(MU*(2/r1 - 1/a1)) - v_circ(r1))
    record_burn(leg_label, "Perigee raise to apogee", t_now, dv_raise)

    r_peri = -node_dir * r1                   # perigee of E1
    r_ap_xyz =  node_dir * r_ap               # apogee of E1 and E2 (node, shared by both planes)
    record_geom_point(leg_label, "Burn 1", r_peri, t_now)
    t_ap = add_seconds(t_now, pi*sqrt(a1**3/MU))

    # Burn 2: rotate velocity about r_ap to target plane (node), so position is unchanged
    a2 = 0.5*(r2+r_ap)
    v_ap_E1 = sqrt(MU*(2/r_ap - 1/a1))
    v_ap_E2 = sqrt(MU*(2/r_ap - 1/a2))
    dv_plane = sqrt(v_ap_E1**2 + v_ap_E2**2 - 2*v_ap_E1*v_ap_E2*cos(np.deg2rad(theta)))
    record_burn(leg_label, "Apogee plane alignment", t_ap, dv_plane)
    record_geom_point(leg_label, "Burn 2 (node)", r_ap_xyz, t_ap)

    # --- Draw coast ellipses that meet at Burn 2 (node) ---
    # Ellipse E1 in START plane: perigee along r_peri, apogee along r_ap_xyz
    omega1 = align_target_plane_argp_to_vector(i_start, Omega_start, r_peri)  # PQW +x -> r_peri
    e1 = (r_ap - r1) / (r_ap + r1); p1 = a1*(1-e1**2)
    nus1 = np.linspace(0.0, np.pi, 600)   # perigee->apogee
    r1_pqw = np.stack([p1/(1+e1*np.cos(nus1))*np.cos(nus1),
                       p1/(1+e1*np.cos(nus1))*np.sin(nus1),
                       np.zeros_like(nus1)], axis=1)
    Q1 = Q_eci_from_pqw(Omega_start, i_start, np.rad2deg(omega1))
    ellipse_E1 = (Q1 @ r1_pqw.T).T
    add_curve(leg_label, circle_in_plane_points(r1, i_start, Omega_start, np.rad2deg(omega1)))
    add_curve(leg_label, ellipse_E1)

    # Ellipse E2 in TARGET plane: apogee at r_ap_xyz (node), perigee at r2 along -node_dir
    omega2 = align_target_plane_argp_to_vector(tgt["i"], tgt["Omega"], -r_ap_xyz)  # PQW +x -> perigee dir
    e2 = (r_ap - r2) / (r_ap + r2); p2 = a2*(1-e2**2)
    nus2 = np.linspace(0.0, 2*np.pi, 800)
    r2_pqw = np.stack([p2/(1+e2*np.cos(nus2))*np.cos(nus2),
                       p2/(1+e2*np.cos(nus2))*np.sin(nus2),
                       np.zeros_like(nus2)], axis=1)
    Q2 = Q_eci_from_pqw(tgt["Omega"], tgt["i"], np.rad2deg(omega2))
    ellipse_E2 = (Q2 @ r2_pqw.T).T
    add_curve(leg_label, circle_in_plane_points(r2, tgt["i"], tgt["Omega"], np.rad2deg(omega2)))
    add_curve(leg_label, ellipse_E2)

    # Time to perigee at r2 on E2 (half period from apogee)
    t_peri_r2 = add_seconds(t_ap, pi*sqrt(a2**3/MU))

    # Lambert segment from circular r2 to exact target state
    udep = u_at(tgt, t_peri_r2)
    r0_vec, v1_circ = chaser_circ_rv(r2, tgt["i"], tgt["Omega"], udep)
    tof0 = 0.5*pi*sqrt(r2**3/MU)
    best = None; best_dep=None; best_arr=None; best_pair=None
    for s in np.linspace(0.4, 1.6, 41):
        tof = max(60.0, s*tof0)
        t_arr = add_seconds(t_peri_r2, tof)
        r2_vec = r_eci(tgt, t_arr); v2_tar = v_eci(tgt, t_arr)
        try:
            v1_tr, v2_tr = lambert_universal(r0_vec, r2_vec, tof, mu=MU, prograde=True)
        except Exception:
            continue
        dv_dep = np.linalg.norm(v1_tr - v1_circ); dv_arr = np.linalg.norm(v2_tar - v2_tr)
        dv_tot = dv_dep + dv_arr
        if (best is None) or (dv_tot < best[0]):
            best = (dv_tot, t_arr, tof); best_dep=dv_dep; best_arr=dv_arr; best_pair=(v1_tr, v2_tr, r2_vec)
    if best is None: raise RuntimeError("Lambert scan failed.")
    dv_lam, t_arrival, tof = best; v1_tr, v2_tr, r_arr_vec = best_pair

    record_burn(leg_label, "Lambert depart", t_peri_r2, best_dep)
    record_burn(leg_label, "Lambert arrival match", t_arrival, best_arr)
    record_geom_point(leg_label, "Lambert depart", r0_vec, t_peri_r2)
    record_geom_point(leg_label, "Lambert arrival", r_arr_vec, t_arrival)

    add_curve(leg_label, kepler_propagate_positions(*elements_from_rv(r0_vec, v1_tr, MU), tof, npts=800, mu=MU))

    T = period_from_a(r2)
    t_after = add_seconds(t_arrival, 5*T)
    return t_after, tgt["i"], tgt["Omega"], dv_raise + dv_plane + dv_lam


# -------------------------
# Inputs (four specific objects)
# -------------------------
objs = [
    dict(name="INTELSAT 33E DEB (61998)", epoch=25316.14507882, inc=0.9979, raan=85.7856,
         ecc=0.0009167, argp=252.6559, M=21.5383, n=1.00437626),
    dict(name="FREGAT R/B (39192)", epoch=25316.66227286, inc=0.0773, raan=183.5260,
         ecc=0.0008031, argp=266.3417, M=270.2279, n=5.20983481235749),
    dict(name="OBJECT 33393U 08048A", epoch=25316.56274537, inc=9.3468, raan=320.0996,
         ecc=0.0009937, argp=135.6367, M=224.4630, n=14.952780489390428),
    dict(name="OBJECT 52939U 22072E", epoch=25316.14146735, inc=9.9523, raan=276.8703,
         ecc=0.0026188, argp=221.3777, M=138.4452, n=15.20682587186371),
]

# Build targets (dicts) and order
T = [target_from_tle_fields(o["name"], o["epoch"], o["inc"], o["raan"], o["ecc"], o["argp"], o["M"], o["n"]) for o in objs]
start = next(t for t in T if "INTELSAT 33E" in t["name"])
freg  = next(t for t in T if "FREGAT" in t["name"])
leoA  = next(t for t in T if "33393U" in t["name"])
leoB  = next(t for t in T if "52939U" in t["name"])

# -------------------------
# Plotting primitives
# -------------------------
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_impulses_per_leg(outdir):
    os.makedirs(outdir, exist_ok=True)
    by_leg = {}
    for b in ALL_BURNS:
        by_leg.setdefault(b["leg"], []).append(b)
    for leg in by_leg:
        by_leg[leg].sort(key=lambda x: to_datetime_utc(x["time"]))

    for leg, items in by_leg.items():
        fig = plt.figure()
        ax = plt.gca()
        xs = [mdates.date2num(to_datetime_utc(b["time"])) for b in items]
        ys = [b["dv_kms"] for b in items]
        labels = [b["label"] for b in items]
        for x, y, lbl in zip(xs, ys, labels):
            ax.vlines(x, 0, y)
            ax.plot([x], [y], marker="o")
            ax.annotate(lbl, (x, y), xytext=(0, 6), textcoords="offset points",
                        rotation=90, ha="center", va="bottom")
        ax.set_title(f"{leg} — Impulse Plot (Δv vs time)")
        ax.set_ylabel("Δv (km/s)")
        ax.set_xlabel("UTC time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))
        plt.tight_layout()
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', leg)[:120]
        plt.savefig(os.path.join(outdir, f"{safe}_impulses.png"))
        plt.close(fig)

def plot_geometry_per_leg(outdir):
    os.makedirs(outdir, exist_ok=True)
    pts_by_leg = {}
    for bp in GEOM_POINTS:
        pts_by_leg.setdefault(bp['leg'], []).append(bp)

    for leg, curves in GEOM_CURVES.items():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for C in curves:
            ax.plot(C[:,0], C[:,1], C[:,2])
        for bp in pts_by_leg.get(leg, []):
            r = bp['r']; ax.scatter([r[0]],[r[1]],[r[2]])
            ax.text(r[0], r[1], r[2], bp['label'])
        ax.set_xlabel('ECI X (km)'); ax.set_ylabel('ECI Y (km)'); ax.set_zlabel('ECI Z (km)')
        ax.set_title(f"{leg} — geometry-accurate")
        # equal-ish aspect
        all_pts = []
        for C in curves: all_pts.append(C)
        if pts_by_leg.get(leg):
            all_pts.append(np.vstack([p['r'] for p in pts_by_leg[leg]]))
        all_pts = np.vstack(all_pts)
        mins = all_pts.min(0); maxs = all_pts.max(0)
        ctr = (mins+maxs)/2; span = (maxs-mins).max()/2
        ax.set_xlim(ctr[0]-span, ctr[0]+span)
        ax.set_ylim(ctr[1]-span, ctr[1]+span)
        ax.set_zlim(ctr[2]-span, ctr[2]+span)
        plt.tight_layout()
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', leg)[:120]
        plt.savefig(os.path.join(outdir, f"{safe}_geometry.png"))
        plt.close(fig)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    print("=== MISSION START ===")
    print("Start on INTELSAT, dwell then visit: FREGAT -> 33393U -> 52939U")

    # Common start at latest epoch
    t0 = max(t["epoch"] for t in T)
    cur_t = t0
    cur_r = start["a"]
    cur_i = start["i"]
    cur_O = start["Omega"]
    total_dv = 0.0

    # Leg 0: rendezvous/dwell on starting orbit (phasing only or trivial transfer)
    cur_t, cur_i, cur_O, dv = leg_hohmann_plane_then_phasing(
        cur_t, cur_r, cur_i, cur_O, start, "Leg 0: Rendezvous at INTELSAT (start)"
    )
    total_dv += dv
    cur_r = start["a"]

    # Leg 1: to FREGAT
    cur_t, cur_i, cur_O, dv = leg_hohmann_plane_then_phasing(
        cur_t, cur_r, cur_i, cur_O, freg, f"Leg 1: {start['name']} -> {freg['name']}"
    )
    total_dv += dv
    cur_r = freg["a"]

    # Leg 2: to 33393U
    cur_t, cur_i, cur_O, dv = leg_hohmann_plane_then_phasing(
        cur_t, cur_r, cur_i, cur_O, leoA, f"Leg 2: {freg['name']} -> {leoA['name']}"
    )
    total_dv += dv
    cur_r = leoA["a"]

    # Leg 3: LEO to LEO with apogee plane-match + coplanar Lambert
    cur_t, cur_i, cur_O, dv = leg_leo_to_leo_plane_apogee_then_lambert(
        cur_t, cur_r, cur_i, cur_O, leoB, f"Leg 3: {leoA['name']} -> {leoB['name']}"
    )
    total_dv += dv
    cur_r = leoB["a"]

    print("\n=== SUMMARY ===")
    print(f"Total DeltaV          : {total_dv:.3f} km/s")
    print(f"Completion (UTC)      : {to_datetime_utc(cur_t)}")

    # Plots
    out_imp = os.path.join("plots")
    out_geom = os.path.join("plots", "geometry")
    plot_impulses_per_leg(out_imp)
    plot_geometry_per_leg(out_geom)
    print(f"\nImpulse plots     -> {os.path.abspath(out_imp)}")
    print(f"Geometry plots    -> {os.path.abspath(out_geom)}")
