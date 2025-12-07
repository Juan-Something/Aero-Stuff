
from math import pi, sqrt, sin, cos
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Constants
# -------------------------
MU = 398600.4418  # km^3/s^2
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
    dt = datetime(year,1,1,tzinfo=timezone.utc) + timedelta(days=day_int-1, seconds=frac_day*SEC_PER_DAY)
    return dt

def R3(th):
    return np.array([[ np.cos(th), -np.sin(th), 0.0],
                     [ np.sin(th),  np.cos(th), 0.0],
                     [ 0.0,         0.0,        1.0]])

def R1(th):
    return np.array([[1.0, 0.0, 0.0 ],
                     [0.0, np.cos(th), -np.sin(th)],
                     [0.0, np.sin(th),  np.cos(th)]])

def Q_eci_from_pqw(raan_deg, inc_deg, argp_deg):
    Ω = np.deg2rad(raan_deg); i = np.deg2rad(inc_deg); ω = np.deg2rad(argp_deg)
    return R3(Ω) @ R1(i) @ R3(ω)

# Target accessors
def target_from_tle_fields(name, epoch, inc, raan, e, argp, Mdeg, n_rev_per_day):
    n_rad_s = n_rev_per_day * 2*np.pi / SEC_PER_DAY
    a = (MU / n_rad_s**2)**(1/3)
    return dict(name=name, epoch=tle_epoch_to_Time(epoch), a=a, e=float(e),
                i=float(inc), Omega=float(raan), omega=float(argp),
                M0=float(Mdeg), n=n_rad_s)

def M_at(tgt, t):
    dt = (t - tgt["epoch"]).total_seconds()
    return wrap2pi(np.deg2rad(tgt["M0"]) + tgt["n"]*dt)

def nu_at(tgt, t): return M_to_nu(M_at(tgt, t), tgt["e"])
def u_at(tgt, t): return wrap2pi(np.deg2rad(tgt["omega"]) + nu_at(tgt, t))

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

# -------------------------
# Lambert + RK4 propagator
# -------------------------
def C(z):
    if z > 0:
        sz=np.sqrt(z); return (1-np.cos(sz))/z
    if z < 0:
        sz=np.sqrt(-z); return (1-np.cosh(sz))/z
    return 0.5

def S(z):
    if z > 0:
        sz=np.sqrt(z); return (sz-np.sin(sz))/(sz**3)
    if z < 0:
        sz=np.sqrt(-z); return (np.sinh(sz)-sz)/(sz**3)
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

def rk4_sample(r0, v0, tof, steps=800, mu=MU):
    def acc(r): return -mu*r/np.linalg.norm(r)**3
    rs = np.zeros((steps,3)); ts = np.linspace(0, tof, steps)
    r = r0.copy(); v = v0.copy()
    for i,t in enumerate(ts):
        rs[i]=r
        h = tof/(steps-1) if steps>1 else tof
        k1r = v;            k1v = acc(r)
        k2r = v + 0.5*h*k1v; k2v = acc(r + 0.5*h*k1r)
        k3r = v + 0.5*h*k2v; k3v = acc(r + 0.5*h*k2r)
        k4r = v + h*k3v;     k4v = acc(r + h*k3r)
        r = r + (h/6.0)*(k1r + 2*k2r + 2*k3r + k4r)
        v = v + (h/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    return ts, rs, r, v

# -------------------------
# Combined-plane Hohmann + propagated phasing
# -------------------------
def unit(v): v = np.array(v, dtype=float); return v/np.linalg.norm(v)

def plane_normal(i_deg, Omega_deg):
    Q = R3(np.deg2rad(Omega_deg)) @ R1(np.deg2rad(i_deg))
    return Q @ np.array([0.0,0.0,1.0])

def node_line(i1, O1, i2, O2):
    n1 = plane_normal(i1,O1); n2 = plane_normal(i2,O2)
    l = np.cross(n1, n2)
    if np.linalg.norm(l) < 1e-12:
        l = R3(np.deg2rad(O1)) @ np.array([1.0, 0.0, 0.0])
    return unit(l), unit(n1), unit(n2)

def circ_state_on_line(a_km, k_hat, rhat):
    r = a_km * rhat
    vdir = np.cross(k_hat, rhat)  # prograde tangent
    v = v_circ(a_km) * unit(vdir)
    return r, v

def hohmann_with_planechange_propagated(a1, i1, O1, a2, i2, O2):
    rhat, k1, k2 = node_line(i1, O1, i2, O2)
    r0, v0 = circ_state_on_line(a1, k1, rhat)
    at = 0.5*(a1+a2)
    v1_speed = sqrt(MU*(2/a1 - 1/at))
    v1_dir   = unit(np.cross(k2, rhat))  # tangent in target plane
    v1       = v1_speed * v1_dir
    dv1      = np.linalg.norm(v1 - v0)
    tof = pi*sqrt(at**3/MU)
    ts, rs, r_end, v_end = rk4_sample(r0, v1, tof, steps=1000)
    v2_circ = v_circ(a2) * unit(np.cross(k2, -rhat))
    dv2 = np.linalg.norm(v2_circ - v_end)
    return {
        "r_path": rs, "tof": tof,
        "r_dep": r0, "v_dep_pre": v0, "v_dep_post": v1, "dv1": dv1,
        "r_arr": r_end, "v_arr_pre": v_end, "v_arr_post": v2_circ, "dv2": dv2,
        "a1": a1, "a2": a2, "k2": k2, "rhat": rhat
    }

def choose_drift(a_tgt, du0_deg=120.0, s_span=0.05, Nmax=20):
    n_t = np.sqrt(MU / a_tgt**3)
    best = None
    du0 = np.deg2rad(du0_deg)
    for N in range(1, Nmax+1):
        for s in np.linspace(-s_span, s_span, 81):
            if abs(s) < 1e-8: continue
            a_d = a_tgt*(1+s)
            n_d = np.sqrt(MU / a_d**3); T_d = 2*np.pi/n_d
            du_res = abs(wrap_pi(du0 - (n_d - n_t)*N*T_d))
            at = 0.5*(a_tgt + a_d)
            dv_out = abs(sqrt(MU*(2/a_tgt - 1/at)) - v_circ(a_tgt))
            v_back_entry = sqrt(MU*(2/a_d - 1/at))
            dv_back = abs(v_circ(a_tgt) - v_back_entry)
            dv_tot = dv_out + dv_back
            score = (du_res, dv_tot)
            if (best is None) or (score < best[0]):
                best = (score, a_d, N)
    return best[1], best[2]

def phasing_sequence_propagated(r_start_on_target, k2, a_tgt, N, a_drift):
    tracks = []
    # To drift
    at_out = 0.5*(a_tgt+a_drift)
    v_out = sqrt(MU*(2/a_tgt - 1/at_out)) * unit(np.cross(k2, unit(r_start_on_target)))
    _, rs1, rE, vE = rk4_sample(r_start_on_target, v_out, pi*sqrt(at_out**3/MU), steps=500)
    tracks.append(rs1)
    # Drift circ
    rhat_E = unit(rE); v_cE = v_circ(a_drift) * unit(np.cross(k2, rhat_E))
    T_d = 2*pi*sqrt(a_drift**3/MU)
    _, rs2, rF, vF = rk4_sample(rE, v_cE, N*T_d, steps=max(600, int(300*N)))
    tracks.append(rs2)
    # Back
    at_back = at_out
    v_back = sqrt(MU*(2/a_drift - 1/at_back)) * unit(np.cross(k2, rhat_E))
    _, rs3, rG, vG = rk4_sample(rF, v_back, pi*sqrt(at_back**3/MU), steps=500)
    tracks.append(rs3)
    # Recirc target
    rhat_G = unit(rG); v_recirc = v_circ(a_tgt) * unit(np.cross(k2, rhat_G))
    return {"segments": tracks, "end_state": (rG, v_recirc)}

# -------------------------
# Leg 3: direct LEO plane change at departure + coplanar Lambert (propagated)
# -------------------------
def leg3_direct_plane_then_lambert_propagated(t0, A, B):
    # A,B are target dicts (leoA -> leoB)
    r1 = A["a"]; r2 = B["a"]
    # Build departure geometry at the intersection line; do the plane change in LEO
    rhat, kA, kB = node_line(A["i"], A["Omega"], B["i"], B["Omega"])
    r_dep, v_dep_pre = circ_state_on_line(r1, kA, rhat)
    v_dep_post = v_circ(r1) * unit(np.cross(kB, rhat))  # same speed, new plane
    dv_plane = np.linalg.norm(v_dep_post - v_dep_pre)

    # Choose a reasonable Lambert TOF scale (half-ellipse at target radius)
    tof0 = 0.5*pi*sqrt(r2**3/MU)
    best = None
    for s in np.linspace(0.4, 1.6, 49):
        tof = max(60.0, s*tof0)
        t_arr = t0 + timedelta(seconds=tof)
        r_tgt_arr = r_eci(B, t_arr)
        v_tgt_arr = v_eci(B, t_arr)
        try:
            v1_tr, v2_tr = lambert_universal(r_dep, r_tgt_arr, tof, mu=MU, prograde=True)
        except Exception:
            continue
        # Departure burn includes injection from post-plane-change circular velocity
        dv_dep = np.linalg.norm(v1_tr - v_dep_post)
        dv_arr = np.linalg.norm(v_tgt_arr - v2_tr)
        dv_tot = dv_plane + dv_dep + dv_arr
        if (best is None) or (dv_tot < best[0]):
            best = (dv_tot, tof, t_arr, v1_tr, v2_tr, dv_plane, dv_dep, dv_arr, r_tgt_arr, v_tgt_arr)

    if best is None:
        raise RuntimeError("Lambert scan failed for direct plane-change leg.")

    dv_total, tof_best, t_arrival, v1_tr, v2_tr, dv_plane, dv_dep, dv_arr, r_tgt_arr, v_tgt_arr = best

    # Propagate the Lambert arc from (r_dep, v1_tr)
    _, rs_lam, r_end, v_end = rk4_sample(r_dep, v1_tr, tof_best, steps=900)

    return {
        "path": rs_lam,
        "dv_plane": dv_plane, "dv_dep": dv_dep, "dv_arr": dv_arr, "dv_tot": dv_total,
        "start_circle": r1, "target_circle": r2
    }

def plot_leg3_direct(leg3, title):
    fig, ax = plt.subplots(figsize=(7.6,6.7))
    ax.plot(leg3["path"][:,0], leg3["path"][:,1], linewidth=2, label="Lambert arc (propagated)")
    th = np.linspace(0,2*np.pi,500)
    r1 = leg3["start_circle"]; r2 = leg3["target_circle"]
    ax.plot(r1*np.cos(th), r1*np.sin(th), linestyle="--", label="LEO A circle (proj)")
    ax.plot(r2*np.cos(th), r2*np.sin(th), linestyle="--", label="LEO B circle (proj)")
    ax.set_aspect("equal"); ax.grid(True)
    ax.set_xlabel("ECI x (km)"); ax.set_ylabel("ECI y (km)")
    ax.set_title(title + f"\nΔv_plane@LEO={leg3['dv_plane']:.3f} km/s, Δv_Lambert dep/arr={leg3['dv_dep']:.3f}/{leg3['dv_arr']:.3f} km/s, "
                         f"Total={leg3['dv_tot']:.3f} km/s")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    plt.show()

# -------------------------
# PLOTTING
# -------------------------
def plot_leg_with_phasing(leg, phase, title):
    fig, ax = plt.subplots(figsize=(7.6,6.7))
    ax.plot(leg["r_path"][:,0], leg["r_path"][:,1], linewidth=2, label="Transfer (propagated)")
    ax.scatter([leg["r_dep"][0]],[leg["r_dep"][1]]); ax.text(leg["r_dep"][0],leg["r_dep"][1]," dep", va="bottom", fontsize=9)
    ax.scatter([leg["r_arr"][0]],[leg["r_arr"][1]]); ax.text(leg["r_arr"][0],leg["r_arr"][1]," arr", va="bottom", fontsize=9)
    th = np.linspace(0,2*np.pi,500)
    ax.plot(leg["a1"]*np.cos(th), leg["a1"]*np.sin(th), linestyle="--", label="Start circle (proj)")
    ax.plot(leg["a2"]*np.cos(th), leg["a2"]*np.sin(th), linestyle="--", label="Target circle (proj)")
    for j,seg in enumerate(phase["segments"], start=1):
        ax.plot(seg[:,0], seg[:,1], alpha=0.9, label=f"Phasing seg {j}")
    ax.set_aspect("equal"); ax.grid(True)
    ax.set_xlabel("ECI x (km)"); ax.set_ylabel("ECI y (km)")
    ax.set_title(title + f"\nΔv1={leg['dv1']:.3f} km/s, Δv2={leg['dv2']:.3f} km/s")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    plt.show()

# -------------------------
# Objects and run
# -------------------------
objs = [
 dict(name="INTELSAT 33E DEB (61998)", epoch=25316.14507882, inc=0.9979,  raan=85.7856,  ecc=0.0009167, argp=252.6559, M=21.5383,  n=1.00437626),
 dict(name="FREGAT R/B (39192)",      epoch=25316.66227286, inc=0.0773,  raan=183.5260, ecc=0.0008031, argp=266.3417, M=270.2279, n=5.20983481235749),
 dict(name="OBJECT 33393U 08048A",     epoch=25316.56274537, inc=9.3468,  raan=320.0996, ecc=0.0009937, argp=135.6367, M=224.4630, n=14.952780489390428),
 dict(name="OBJECT 52939U 22072E",     epoch=25316.14146735, inc=9.9523,  raan=276.8703, ecc=0.0026188, argp=221.3777, M=138.4452, n=15.20682587186371),
]
T = [target_from_tle_fields(o["name"], o["epoch"], o["inc"], o["raan"], o["ecc"], o["argp"], o["M"], o["n"]) for o in objs]
start = next(t for t in T if "INTELSAT 33E" in t["name"])
freg  = next(t for t in T if "FREGAT" in t["name"])
leoA  = next(t for t in T if "33393U" in t["name"])
leoB  = next(t for t in T if "52939U" in t["name"])

if __name__ == "__main__":
    # Common start at latest epoch
    t0 = max(t["epoch"] for t in T)

    print("=== MISSION START (propagated legs 1–3) ===")
    print("Start on INTELSAT; visit FREGAT, then 33393U, then 52939U.")

    # ---- Leg 1: INTELSAT -> FREGAT (combined-plane Burn 1) + propagated phasing ----
    leg1 = hohmann_with_planechange_propagated(start["a"], start["i"], start["Omega"],
                                               freg["a"],  freg["i"],  freg["Omega"])
    a_d1, N1 = choose_drift(freg["a"])
    phase1 = phasing_sequence_propagated(leg1["r_arr"], leg1["k2"], freg["a"], N1, a_d1)
    plot_leg_with_phasing(leg1, phase1, "Leg 1: INTELSAT → FREGAT (HT+plane on Burn 1) + phasing")

    # Dwell 5 periods on FREGAT
    dwell1 = 5*period_from_a(freg["a"])

    # ---- Leg 2: FREGAT -> 33393U (combined-plane Burn 1) + propagated phasing ----
    leg2 = hohmann_with_planechange_propagated(freg["a"], freg["i"], freg["Omega"],
                                               leoA["a"],  leoA["i"],  leoA["Omega"])
    a_d2, N2 = choose_drift(leoA["a"])
    phase2 = phasing_sequence_propagated(leg2["r_arr"], leg2["k2"], leoA["a"], N2, a_d2)
    plot_leg_with_phasing(leg2, phase2, "Leg 2: FREGAT → 33393U (HT+plane on Burn 1) + phasing")

    # Dwell 5 periods on 33393U
    dwell2 = 5*period_from_a(leoA["a"])

    # ---- Leg 3: 33393U -> 52939U (direct LEO plane change + Lambert; no apogee raise) ----
    # Departure epoch bookkeeping (so target position is consistent)
    t3_dep = t0 + timedelta(seconds=leg1["tof"] + dwell1 + leg2["tof"] + dwell2)
    leg3 = leg3_direct_plane_then_lambert_propagated(t3_dep, leoA, leoB)
    plot_leg3_direct(leg3, "Leg 3: 33393U → 52939U (direct LEO plane change + Lambert)")

    # Δv tallies (transfer-only for legs 1–2; full for leg 3)
    dv_leg1 = leg1["dv1"] + leg1["dv2"]
    dv_leg2 = leg2["dv1"] + leg2["dv2"]
    dv_leg3 = leg3["dv_plane"] + leg3["dv_dep"] + leg3["dv_arr"]
    print("\\n=== Δv SUMMARY ===")
    print(f"Leg 1 (transfer only): {dv_leg1:.3f} km/s")
    print(f"Leg 2 (transfer only): {dv_leg2:.3f} km/s")
    print(f"Leg 3 (plane@LEO + Lambert dep/arr): {dv_leg3:.3f} km/s")
    print("(Phasing Δv not included; add if you need the full mission budget.)")
