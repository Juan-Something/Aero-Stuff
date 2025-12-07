# -*- coding: utf-8 -*-
# Purpose-built mission plan (with 2D geometry plots), two-body Kepler, minimal scaffolding

from math import pi, sqrt, sin, cos
from datetime import datetime, timedelta, timezone
import numpy as np
from astropy.time import Time, TimeDelta

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
    yy = int(epoch_field // 1000)
    doy = epoch_field - yy*1000
    year = (2000 + yy) if yy < 57 else (1900 + yy)
    day_int = int(doy)
    frac_day = doy - day_int
    dt = datetime(year,1,1,tzinfo=timezone.utc) + timedelta(days=day_int-1,
                                                            seconds=frac_day*SEC_PER_DAY)
    return Time(dt)

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
    dt = (t - tgt["epoch"]).to_value("sec")
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

# Hohmann + plane-rotation (in Burn #1)
def dv_combined(vb, va, ang_deg):
    di = np.deg2rad(ang_deg)
    return sqrt(vb*vb + va*va - 2*vb*va*cos(di))

# -------------------------
# Universal-variable Lambert (0-rev), minimal
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

# -------------------------
# Mission legs (with corrected phasing metric)
# -------------------------
def leg_hohmann_plane_then_phasing(t_now, r_start, i_start, Omega_start, tgt, leg_label):
    r_tgt = tgt["a"]; i_tgt, Omega_tgt = tgt["i"], tgt["Omega"]
    theta = plane_angle_deg(i_start, Omega_start, i_tgt, Omega_tgt)
    if abs(r_start - r_tgt) < 1e-6 and theta < 1e-9:
        t_arr = t_now; dv1=dv2=dv_leg=0.0
    else:
        at = 0.5*(r_start + r_tgt)
        v1 = v_circ(r_start)
        vt1 = sqrt(MU*(2/r_start - 1/at))
        dv1 = dv_combined(v1, vt1, theta)
        v2 = v_circ(r_tgt)
        vt2 = sqrt(MU*(2/r_tgt - 1/at))
        dv2 = abs(v2 - vt2)
        dv_leg = dv1 + dv2
        t_arr = t_now + TimeDelta(pi*sqrt(at**3/MU), format="sec")

    # ---- corrected phasing: compare target vs drifter; forbid s≈0 ----
    v_c = v_circ(r_tgt)
    u_tar_at_arr = u_at(tgt, t_arr)
    # crude but consistent estimate: after a nontrivial transfer, assume opposite side
    u_ch_at_arr = wrap2pi(u_tar_at_arr + (np.pi if dv_leg > 0 else 0.0))
    du0 = wrap_pi(u_tar_at_arr - u_ch_at_arr)
    n_t = sqrt(MU / r_tgt**3)

    best = None
    for N in range(1, 25):
        for s in np.linspace(-0.05, 0.05, 41):
            if abs(s) < 1e-6:  # disallow trivial zero-burn phasing
                continue
            a_d = r_tgt*(1+s); n_d = sqrt(MU/a_d**3)
            T_d = 2*pi/n_d; t_d = N*T_d
            du_res = abs(wrap_pi(du0 - (n_d - n_t)*t_d))
            at = 0.5*(r_tgt + a_d)
            vt1 = sqrt(MU*(2/r_tgt - 1/at))
            dv_out = abs(vt1 - v_c)
            v_back_entry = sqrt(MU*(2/a_d - 1/at))
            dv_back = abs(v_c - v_back_entry)
            dv_tot = dv_out + dv_back
            score = (du_res, dv_tot)
            if (best is None) or (score < best[0]):
                best = (score, dv_tot, t_arr + TimeDelta(t_d, format="sec"))
    dv_phase = best[1]; t_int = best[2]

    T = period_from_a(r_tgt)
    t_after = t_int + TimeDelta(5*T, format="sec")
    # report
    def fmt(t): return t.to_datetime(timezone=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n=== {leg_label} ===")
    print(f"Start (r,i,Ω)                  : ({r_start:,.1f} km, {i_start:.3f}°, {Omega_start:.3f}°) @ {fmt(t_now)}")
    print(f"Target (r,i,Ω,ω,e)             : ({r_tgt:,.1f} km, {i_tgt:.3f}°, {Omega_tgt:.3f}°, {tgt['omega']:.3f}°, {tgt['e']:.5f})")
    if dv_leg > 0:
        print(f"Plane-rotation θ               : {theta:.3f}°")
        print(f"Transfer burns (dv1+dv2)       : {dv1:.3f} + {dv2:.3f} = {dv_leg:.3f} km/s")
        print(f"Arrival                        : {fmt(t_arr)}")
    print(f"Phasing dv (out+back)          : {dv_phase:.3f} km/s; intercept {fmt(t_int)}")
    print(f"Dwell 5 periods                : until {fmt(t_after)}")
    return t_after, i_tgt, Omega_tgt, dv_leg + dv_phase

# LEO↔LEO: apogee plane-match → coplanar Lambert → dwell
def leg_leo_to_leo_full3d_lambert(t_now, r_start, i_start, Omega_start, tgt, leg_label):
    """
    Full 3D Lambert from the current circular orbit (r_start, i_start, Omega_start)
    to the target's orbit (tgt) without any pre-plane-match. We scan departure
    argument u_dep and time-of-flight, solve Lambert in 3D, and choose the min-Δv pair.
    """
    r1 = r_start
    r2 = tgt["a"]

    # Baseline TOF scale (use mean radius for a mid-range Kepler time)
    r_mean = 0.5*(r1 + r2)
    tof_ref = 0.5*pi*sqrt(r_mean**3 / MU)

    best = None
    best_tuple = None

    # Joint scan: u_dep ∈ [0, 2π), TOF ∈ [0.4, 1.6] × tof_ref (≥60 s)
    for u_dep in np.linspace(0.0, 2*np.pi, 36, endpoint=False):
        r1_vec, v1_circ = chaser_circ_rv(r1, i_start, Omega_start, u_dep)
        for s in np.linspace(0.4, 1.6, 41):
            tof = max(60.0, s*tof_ref)
            t_arr = t_now + TimeDelta(tof, format="sec")
            r2_vec = r_eci(tgt, t_arr)
            v2_tar = v_eci(tgt, t_arr)
            try:
                # Full 3D: let the solver connect arbitrary planes
                v1_tr, v2_tr = lambert_universal(r1_vec, r2_vec, tof, mu=MU, prograde=True)
            except Exception:
                continue
            dv_dep = np.linalg.norm(v1_tr - v1_circ)
            dv_arr = np.linalg.norm(v2_tar - v2_tr)
            dv_tot = dv_dep + dv_arr
            if (best is None) or (dv_tot < best):
                best = dv_tot
                best_tuple = (dv_tot, dv_dep, dv_arr, t_arr, u_dep, tof)

    if best_tuple is None:
        raise RuntimeError("3D Lambert search failed to find a feasible transfer.")

    dv_tot, dv_dep, dv_arr, t_arr, u_dep_sel, tof_sel = best_tuple

    # Dwell after arrival
    T = period_from_a(r2)
    t_after = t_arr + TimeDelta(5*T, format="sec")

    # Report
    def fmt(t): return t.to_datetime(timezone=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n=== {leg_label} (full 3D Lambert) ===")
    print(f"Start circular (r,i,Ω)         : ({r1:,.1f} km, {i_start:.3f}°, {Omega_start:.3f}°) @ {fmt(t_now)}")
    print(f"Target (r,i,Ω,ω,e)             : ({r2:,.1f} km, {tgt['i']:.3f}°, {tgt['Omega']:.3f}°, {tgt['omega']:.3f}°, {tgt['e']:.5f})")
    print(f"Chosen u_dep                    : {np.rad2deg(u_dep_sel):.2f}°")
    print(f"TOF                             : {tof_sel/60.0:.2f} min")
    print(f"Δv_dep / Δv_arr                 : {dv_dep:.3f} + {dv_arr:.3f} = {dv_tot:.3f} km/s")
    print(f"Arrival                         : {fmt(t_arr)} | Dwell 5T until {fmt(t_after)}")

    # Orientation after this leg remains the target’s plane (since we rendezvous there)
    return t_after, tgt["i"], tgt["Omega"], dv_tot


# -------------------------
# PLOTTING ADD-ON
# -------------------------
import matplotlib.pyplot as plt

def _sample_circle(r, th0=0.0, th1=2*np.pi, num=400):
    th = np.linspace(th0, th1, num)
    return r*np.cos(th), r*np.sin(th)

def _sample_ellipse(a, e, th0, th1, num=400):
    th = np.linspace(th0, th1, num)
    r = a*(1-e**2)/(1+e*np.cos(th))
    return r*np.cos(th), r*np.sin(th)

def _hohmann_params(rp, ra):
    a = 0.5*(rp + ra)
    e = abs(ra - rp)/(ra + rp)
    return a, e

def _annot(ax, x, y, txt):
    ax.scatter([x],[y], zorder=6)
    ax.text(x, y, "  "+txt, va="bottom")

def plot_hohmann_plus_phasing(r1, r2, du0_deg=120.0,
                              title="Hohmann + plane-rotation with phasing overlay"):
    """Draw the Hohmann transfer r1→r2 and overlay the phasing sequence executed at r2."""
    fig, ax = plt.subplots(figsize=(7,6))

    # Base circles
    x,y = _sample_circle(r1); ax.plot(x,y, label="Start circular")
    x,y = _sample_circle(r2); ax.plot(x,y, linestyle="--", label="Target circular")

    # Hohmann geometry
    rp, ra = (min(r1,r2), max(r1,r2))
    a_t, e_t = _hohmann_params(rp, ra)
    # Put Burn 1 at +x for raising (r1<r2) and at −x for lowering (r1>r2)
    th1, th2 = (0.0, np.pi) if r1 < r2 else (np.pi, 0.0)
    xt, yt = _sample_ellipse(a_t, e_t, th1, th2)
    ax.plot(xt, yt, label="Transfer ellipse")
    _annot(ax, r1*np.cos(th1), r1*np.sin(th1), "Burn 1 (Δv1) + plane rot.")
    _annot(ax, r2*np.cos(th2), r2*np.sin(th2), "Burn 2 (Δv2)")

    # ----- Phasing overlay at r2 -----
    # Solve phasing at the target radius r2 with requested initial along-track error
    a_d, N, dv_out, dv_back, du_res = _solve_phasing(r2, np.deg2rad(du0_deg))
    rp_p, ra_p = (min(r2, a_d), max(r2, a_d))
    a_ho_p, e_ho_p = _hohmann_params(rp_p, ra_p)

    th_tan, th_opp = (th2, (th2+np.pi)%(2*np.pi))  # Phasing burns tangent at the arrival point

    # Outbound ellipse (to drift)
    if r2 == rp_p:
        xpo, ypo = _sample_ellipse(a_ho_p, e_ho_p, th_tan, th_opp)
    else:
        xpo, ypo = _sample_ellipse(a_ho_p, e_ho_p, th_opp, 2*np.pi+th_tan)
    ax.plot(xpo, ypo, alpha=0.8, label="Phasing transfer (out)")

    # Inbound ellipse (back from drift)
    if r2 == rp_p:
        xpb, ypb = _sample_ellipse(a_ho_p, e_ho_p, th_opp, 2*np.pi+th_tan)
    else:
        xpb, ypb = _sample_ellipse(a_ho_p, e_ho_p, th_tan, th_opp)
    ax.plot(xpb, ypb, alpha=0.8, label="Phasing transfer (back)")

    # Phasing burns and drift arc
    _annot(ax, r2*np.cos(th_tan), r2*np.sin(th_tan), "Phasing Burn 1")
    _annot(ax, a_d*np.cos(th_opp), a_d*np.sin(th_opp), "Phasing Burn 2")
    xd, yd = _sample_circle(a_d, th_opp, th_opp+2*np.pi)
    ax.plot(xd, yd, linewidth=2, alpha=0.55, label=f"Drift {N} rev (1 shown)")
    ax.text(a_d*np.cos(th_opp+0.25), a_d*np.sin(th_opp+0.25), f"× {N} rev")
    _annot(ax, a_d*np.cos(th_opp), a_d*np.sin(th_opp), "Phasing Burn 3")
    _annot(ax, r2*np.cos(th_tan), r2*np.sin(th_tan), "Phasing Burn 4")

    # Cosmetics
    ax.set_aspect("equal"); ax.grid(True)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    box = (f"r2 = {r2:,.0f} km | a_d = {a_d:,.0f} km (Δa/a={(a_d/r2-1)*100:+.2f}%)\n"
           f"N = {N} rev | |Δu| ≈ {np.rad2deg(du_res):.3f}° | Δv_phase ≈ {dv_out+dv_back:.3f} km/s")
    ax.annotate(box, xy=(0.02, 0.02), xycoords="axes fraction",
                va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.5", alpha=0.85))
    plt.show()


def _solve_phasing(r_tgt, du0_rad, s_span=0.05, Nmax=24):
    n_t = np.sqrt(MU / r_tgt**3)
    best = None
    step = 0.0025
    grid = np.linspace(-s_span, s_span, int(2*s_span/step)+1)
    for N in range(1, Nmax+1):
        for s in grid:
            if abs(s) < 1e-8:
                continue
            a_d = r_tgt*(1+s)
            n_d = np.sqrt(MU / a_d**3)
            T_d = 2*np.pi / n_d
            t_d = N*T_d
            du_res = abs(wrap_pi(du0_rad - (n_d - n_t)*t_d))
            at = 0.5*(r_tgt + a_d)
            vt1 = np.sqrt(MU*(2/r_tgt - 1/at))
            dv_out = abs(vt1 - v_circ(r_tgt))
            v_back_entry = np.sqrt(MU*(2/a_d - 1/at))
            dv_back = abs(v_circ(r_tgt) - v_back_entry)
            dv_tot = dv_out + dv_back
            score = (du_res, dv_tot)
            if (best is None) or (score < best[0]):
                best = (score, a_d, N, dv_out, dv_back, du_res)
    (_, _), a_d, N, dv_out, dv_back, du_res = best
    return a_d, N, dv_out, dv_back, du_res

def plot_phasing_geometry(r_tgt, du0_deg=120.0, title="Phasing at target altitude"):
    du0 = np.deg2rad(du0_deg)
    a_d, N, dv_out, dv_back, du_res = _solve_phasing(r_tgt, du0)
    rp, ra = (min(r_tgt, a_d), max(r_tgt, a_d))
    a_ho, e_ho = _hohmann_params(rp, ra)

    fig, ax = plt.subplots(figsize=(6.8,6.8))
    x,y = _sample_circle(r_tgt); ax.plot(x,y, label="Target circle r_tgt")
    xd,yd = _sample_circle(a_d); ax.plot(xd,yd, linestyle="--", label="Drift circle a_d")

    th_tan, th_opp = 0.0, np.pi
    raise_case = (r_tgt == rp)

    if raise_case:
        xt, yt = _sample_ellipse(a_ho, e_ho, th_tan, th_opp)
    else:
        xt, yt = _sample_ellipse(a_ho, e_ho, th_opp, 2*np.pi+th_tan)
    ax.plot(xt, yt, label="Transfer (out)")

    if raise_case:
        xt2, yt2 = _sample_ellipse(a_ho, e_ho, th_opp, 2*np.pi+th_tan)
    else:
        xt2, yt2 = _sample_ellipse(a_ho, e_ho, th_tan, th_opp)
    ax.plot(xt2, yt2, label="Transfer (back)")

    _annot(ax, r_tgt*np.cos(th_tan), r_tgt*np.sin(th_tan), "Burn 1: to drift")
    _annot(ax, a_d*np.cos(th_opp),   a_d*np.sin(th_opp),   "Burn 2: circ @ a_d")

    x1,y1 = _sample_circle(a_d, th_opp, th_opp+2*np.pi)
    ax.plot(x1,y1, linewidth=2, alpha=0.55, label=f"Drift {N} rev (1 shown)")
    ax.text(a_d*np.cos(th_opp+0.25), a_d*np.sin(th_opp+0.25), f"× {N} rev")

    _annot(ax, a_d*np.cos(th_opp),   a_d*np.sin(th_opp),   "Burn 3: leave drift")
    _annot(ax, r_tgt*np.cos(th_tan), r_tgt*np.sin(th_tan), "Burn 4: recirc")

    ax.set_aspect("equal"); ax.grid(True)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    box = (f"r_tgt = {r_tgt:,.0f} km\n"
           f"a_d = {a_d:,.0f} km (Δa/a = {(a_d/r_tgt-1)*100:+.2f}%)\n"
           f"N = {N} rev, |Δu| ≈ {np.rad2deg(du_res):.3f}°\n"
           f"Δv_phase ≈ {dv_out+dv_back:.3f} km/s")
    ax.annotate(box, xy=(0.02, 0.02), xycoords="axes fraction",
                va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.5", alpha=0.85))
    plt.show()

def plot_leo2leo_apogee_lambert(rA, rB, r_ap_factor=4.0,
                                title="LEO→LEO: apogee plane-match + coplanar Lambert",
                                show_lambert=True):
    fig, ax = plt.subplots(figsize=(7,6))
    x,y = _sample_circle(rA); ax.plot(x,y, label="LEO A")
    x,y = _sample_circle(rB); ax.plot(x,y, linestyle="--", label="LEO B")
    r_ap = r_ap_factor * max(rA, rB)
    a1,e1 = _hohmann_params(min(rA,r_ap), max(rA,r_ap))
    xt, yt = _sample_ellipse(a1,e1, 0.0, np.pi)
    ax.plot(xt, yt, label="Raise ellipse to r_ap")
    a2,e2 = _hohmann_params(min(rB,r_ap), max(rB,r_ap))
    xt2, yt2 = _sample_ellipse(a2,e2, np.pi, 2*np.pi)
    ax.plot(xt2, yt2, label="Lower ellipse to LEO B")
    _annot(ax, rA, 0.0,            "Burn 1 (raise)")
    _annot(ax, -r_ap, 0.0,         "Burn 2 (apogee plane match)")
    _annot(ax, -rB, 0.0,           "Burn 3 (circularize LEO B)")
    th_dep, th_arr = 0.25*np.pi, 2.3
    xD, yD = rB*np.cos(th_dep), rB*np.sin(th_dep)
    xA, yA = rB*np.cos(th_arr), rB*np.sin(th_arr)
    ax.scatter([xD,xA],[yD,yA], zorder=6)
    ax.text(xD, yD, "  Lambert dep", va="bottom")
    ax.text(xA, yA, "  Lambert arr", va="bottom")
    if show_lambert:
        ctrl = np.array([(xD+xA)/2, (yD+yA)/2])*1.15
        t = np.linspace(0,1,150)
        bez = (1-t)[:,None]**2*np.array([xD,yD]) + 2*(1-t)[:,None]*t[:,None]*ctrl + t[:,None]**2*np.array([xA,yA])
        ax.plot(bez[:,0], bez[:,1], label="Lambert (schematic)")
    ax.set_aspect("equal"); ax.grid(True)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title(title); ax.legend(loc="upper right")
    plt.show()

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

# Build targets (dicts) and order: start on INTELSAT, then FREGAT, then 33393U, then 52939U
T = [target_from_tle_fields(o["name"], o["epoch"], o["inc"], o["raan"], o["ecc"], o["argp"], o["M"], o["n"]) for o in objs]
start = next(t for t in T if "INTELSAT 33E" in t["name"])
freg  = next(t for t in T if "FREGAT" in t["name"])
leoA  = next(t for t in T if "33393U" in t["name"])
leoB  = next(t for t in T if "52939U" in t["name"])

# -------------------------
# Run (minimal state bookkeeping)
# -------------------------
if __name__ == "__main__":
    # Common start at latest epoch
    t0 = max(t["epoch"] for t in T)
    cur_t = t0
    cur_r = start["a"]
    cur_i = start["i"]
    cur_O = start["Omega"]
    total_dv = 0.0

    print("=== MISSION START ===")
    print("Start on INTELSAT, dwell then visit: FREGAT -> 33393U -> 52939U")

    # Leg 0: rendezvous/dwell on starting orbit (phasing only)
    cur_t, cur_i, cur_O, dv = leg_hohmann_plane_then_phasing(cur_t, cur_r, cur_i, cur_O, start,
                                                             "Rendezvous at INTELSAT (starting orbit)")
    total_dv += dv
    cur_r = start["a"]

    # Plot phasing at INTELSAT altitude
    plot_phasing_geometry(start["a"], du0_deg=120.0,
                          title="Phasing at INTELSAT altitude (geometry & burns)")

    # Leg 1: to FREGAT (Hohmann+plane, then phasing)
    cur_t, cur_i, cur_O, dv = leg_hohmann_plane_then_phasing(cur_t, cur_r, cur_i, cur_O, freg,
                                                             f"{start['name']} -> {freg['name']}")
    total_dv += dv
    cur_r = freg["a"]

    # Plots for leg 1
    plot_hohmann_plus_phasing(start["a"], freg["a"], du0_deg=120.0,
                          title="INTELSAT → FREGAT: transfer + phasing overlay")
    #plot_phasing_geometry(freg["a"], du0_deg=120.0,
                      #    title="Phasing at FREGAT altitude (post-transfer)")

    # Leg 2: to 33393U (Hohmann+plane, then phasing)
    cur_t, cur_i, cur_O, dv = leg_hohmann_plane_then_phasing(cur_t, cur_r, cur_i, cur_O, leoA,
                                                             f"{freg['name']} -> {leoA['name']}")
    total_dv += dv
    cur_r = leoA["a"]

    # Plots for leg 2
    plot_hohmann_plus_phasing(freg["a"], leoA["a"], du0_deg=120.0,
                          title="FREGAT → 33393U: transfer + phasing overlay")
    #plot_phasing_geometry(leoA["a"], du0_deg=120.0,
     #                     title="Phasing at 33393U altitude (post-transfer)")

    # Leg 3: LEO to LEO with apogee plane-match + coplanar Lambert
    cur_t, cur_i, cur_O, dv = leg_leo_to_leo_full3d_lambert(
    cur_t, cur_r, cur_i, cur_O, leoB, f"{leoA['name']} -> {leoB['name']} (3D Lambert)")
    total_dv += dv
    cur_r = leoB["a"]

    # Plot for leg 3
    plot_leo2leo_apogee_lambert(leoA["a"], leoB["a"],
                                title="33393U → 52939U: apogee plane-match + LamberAt")

    print("\n=== SUMMARY ===")
    print(f"Total DeltaV          : {total_dv:.3f} km/s")
    print(f"Completion (UTC)               : {cur_t.to_datetime(timezone=timezone.utc)}")
