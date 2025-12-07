from math import pi, sqrt, sin, cos
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Constants
# -------------------------
MU = 398600  # km^3/s^2
SEC_PER_DAY = 86400.0

# -------------------------
# Tiny helpers
# -------------------------
def vCirc(r): return sqrt(MU / r)
def periodFromA(a): return 2*np.pi*np.sqrt(a**3/MU)
def wrap2pi(x): return np.mod(x, 2*np.pi)
def wrapPi(x): return (x + np.pi) % (2*np.pi) - np.pi

def solveEFromM(M, e, tol=1e-12, it=100):
    M = (M + np.pi) % (2*np.pi) - np.pi
    E = M if e < 0.8 else np.pi
    for _ in range(it):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        dE = -f/fp
        E += dE
        if abs(dE) < tol: break
    return E

def MToNu(M, e):
    E = solveEFromM(M, e)
    sv = sqrt(1-e**2)*sin(E)/(1-e*cos(E))
    cv = (cos(E)-e)/(1-e*cos(E))
    return np.arctan2(sv, cv)

def tleEpochToTime(epochField):
    yy = int(epochField // 1000)
    doy = epochField - yy*1000
    year = (2000 + yy)                  # yy=25 -> 2025
    dayInt = int(doy)
    fracDay = doy - dayInt
    dt = datetime(year,1,1,tzinfo=timezone.utc) + timedelta(days=dayInt-1, seconds=fracDay*SEC_PER_DAY)
    return dt

def R3(th):
    return np.array([[ np.cos(th), -np.sin(th), 0.0],
                     [ np.sin(th),  np.cos(th), 0.0],
                     [ 0.0,         0.0,        1.0]])

def R1(th):
    return np.array([[1.0, 0.0, 0.0 ],
                     [0.0, np.cos(th), -np.sin(th)],
                     [0.0, np.sin(th),  np.cos(th)]])

def QEciFromPqw(raanDeg, incDeg, argpDeg):
    Ω = np.deg2rad(raanDeg); i = np.deg2rad(incDeg); ω = np.deg2rad(argpDeg)
    return R3(Ω) @ R1(i) @ R3(ω)

def planeAngleDeg(i1Deg, Omega1Deg, i2Deg, Omega2Deg):
    i1, i2 = np.deg2rad([i1Deg, i2Deg])
    dO = np.deg2rad((Omega2Deg - Omega1Deg + 180) % 360 - 180)
    cth = np.cos(i1)*np.cos(i2) + np.sin(i1)*np.sin(i2)*np.cos(dO)
    return float(np.rad2deg(np.arccos(np.clip(cth, -1.0, 1.0))))

# -------------------------
# Rp/Ra timing helpers
# -------------------------
def a_from_rp_ra(rp_km, ra_km):
    return 0.5*(rp_km + ra_km)

def e_from_rp_ra(rp_km, ra_km):
    return (ra_km - rp_km) / (ra_km + rp_km)

def n_from_a(a):
    return np.sqrt(MU / a**3)

def a_for_time(tgt):
    """Semimajor axis used for time (prefers Rp/Ra if present)."""
    if ("rp_km" in tgt) and ("ra_km" in tgt) and (tgt["rp_km"] is not None) and (tgt["ra_km"] is not None):
        return a_from_rp_ra(tgt["rp_km"], tgt["ra_km"])
    return tgt["a"]

def periodFromTarget(tgt):
    return 2*np.pi*np.sqrt(a_for_time(tgt)**3 / MU)

# -------------------------
# Target accessors
# -------------------------
def targetFromTleFields(name, epoch, inc, raan, e, argp, Mdeg, nRevPerDay, rp_km=None, ra_km=None):
    """
    If rp_km/ra_km provided, timing will use a=(rp+ra)/2 via a_for_time().
    Geometry (r,v) still uses the stored a,e unless you also choose to derive e from rp/ra externally.
    """
    if (rp_km is not None) and (ra_km is not None):
        a_mm = a_from_rp_ra(rp_km, ra_km)   # for 'n' storage (not strictly required)
        nRadS = n_from_a(a_mm)
        a_store = (MU / nRadS**2)**(1/3)    # keep consistency; won't be used for timing
    else:
        nRadS = nRevPerDay * 2*np.pi / SEC_PER_DAY
        a_store = (MU / nRadS**2)**(1/3)

    return dict(
        name=name, epoch=tleEpochToTime(epoch),
        a=a_store, e=float(e),
        i=float(inc), Omega=float(raan), omega=float(argp),
        M0=float(Mdeg), n=nRadS,
        rp_km=rp_km, ra_km=ra_km
    )

def MAt(tgt, t):
    dt = (t - tgt["epoch"]).total_seconds()
    return wrap2pi(np.deg2rad(tgt["M0"]) + tgt["n"]*dt)

def MToU(M, omega_deg, e):
    return wrap2pi(np.deg2rad(omega_deg) + MToNu(M, e))

def nuAt(tgt, t): return MToNu(MAt(tgt, t), tgt["e"])
def uAt(tgt, t): return wrap2pi(np.deg2rad(tgt["omega"]) + nuAt(tgt, t))

def rEci(tgt, t):
    nu = nuAt(tgt, t); a, e = tgt["a"], tgt["e"]
    p = a*(1 - e**2); rmag = p/(1 + e*np.cos(nu))
    rPqw = np.array([rmag*np.cos(nu), rmag*np.sin(nu), 0.0])
    Q = QEciFromPqw(tgt["Omega"], tgt["i"], tgt["omega"])
    return Q @ rPqw

def vEci(tgt, t):
    nu = nuAt(tgt, t); a, e = tgt["a"], tgt["e"]
    p = a*(1 - e**2); h = sqrt(MU*p)
    vPqw = np.array([-np.sin(nu), e + np.cos(nu), 0.0]) * (MU/h)
    Q = QEciFromPqw(tgt["Omega"], tgt["i"], tgt["omega"])
    return Q @ vPqw

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

def lambertUniversal(r1, r2, tof, mu=MU, prograde=True, it=60, tol=1e-9):
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
        tofZ = (x**3*Sz + A*np.sqrt(y))/np.sqrt(mu)
        if abs(tofZ - tof) < tol: break
        z += 0.1 if (tofZ < tof) else -0.1
    Cz, Sz = C(z), S(z)
    y = r1n + r2n + A*(z*Sz - 1)/np.sqrt(Cz)
    f = 1 - y/r1n; g = A*np.sqrt(y/mu); gdot = 1 - y/r2n
    v1 = (r2 - f*r1)/g
    v2 = (gdot*r2 - r1)/g
    return v1, v2

def rk4Sample(r0, v0, tof, steps=800, mu=MU):
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
# Utilities
# -------------------------
def unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v/n if n != 0.0 else v

def planeNormal(iDeg, OmegaDeg):
    Q = R3(np.deg2rad(OmegaDeg)) @ R1(np.deg2rad(iDeg))
    return Q @ np.array([0.0,0.0,1.0])

def nodeLine(i1, O1, i2, O2):
    n1 = planeNormal(i1,O1); n2 = planeNormal(i2,O2)
    l = np.cross(n1, n2)
    if np.linalg.norm(l) < 1e-12:
        l = R3(np.deg2rad(O1)) @ np.array([1.0, 0.0, 0.0])
    return unit(l), unit(n1), unit(n2)

def circStateOnLine(aKm, kHat, rhat):
    r = aKm * rhat
    vdir = np.cross(kHat, rhat)  # prograde tangent
    v = vCirc(aKm) * unit(vdir)
    return r, v

# -------------------------
# Leg 1/2: Hohmann with combined plane on Burn 1 (+ propagated phasing)
# -------------------------
def hohmannWithPlanechangePropagated(a1, i1, O1, a2, i2, O2):
    rhat, k1, k2 = nodeLine(i1, O1, i2, O2)
    r0, v0 = circStateOnLine(a1, k1, rhat)

    at = 0.5*(a1+a2)
    v1Speed = sqrt(MU*(2/a1 - 1/at))
    v1Dir   = unit(np.cross(k2, rhat))  # tangent in target plane
    v1      = v1Speed * v1Dir
    dv1_vec = v1 - v0
    dv1     = np.linalg.norm(dv1_vec)

    tof = np.pi*np.sqrt(at**3/MU)
    _, rs, rEnd, vEnd = rk4Sample(r0, v1, tof, steps=1000)

    v2Circ  = vCirc(a2) * unit(np.cross(k2, -rhat))
    dv2_vec = v2Circ - vEnd
    dv2     = np.linalg.norm(dv2_vec)

    return {
        "r_path": rs, "tof": tof,
        "r_dep": r0, "v_dep_pre": v0, "v_dep_post": v1, "dv1": dv1, "dv1_vec": dv1_vec,
        "r_arr": rEnd, "v_arr_pre": vEnd, "v_arr_post": v2Circ, "dv2": dv2, "dv2_vec": dv2_vec,
        "a1": a1, "a2": a2, "k2": k2, "rhat": rhat
    }

def chooseDrift(aTgt, du0Deg=120.0, sSpan=0.05, Nmax=20):
    nT = np.sqrt(MU / aTgt**3)
    best = None
    du0 = np.deg2rad(du0Deg)
    for N in range(1, Nmax+1):
        for s in np.linspace(-sSpan, sSpan, 81):
            if abs(s) < 1e-8: continue
            aD = aTgt*(1+s)
            nD = np.sqrt(MU / aD**3); TD = 2*np.pi/nD
            duRes = abs(wrapPi(du0 - (nD - nT)*N*TD))
            at = 0.5*(aTgt + aD)
            dvOut = abs(sqrt(MU*(2/aTgt - 1/at)) - vCirc(aTgt))
            vBackEntry = sqrt(MU*(2/aD - 1/at))
            dvBack = abs(vCirc(aTgt) - vBackEntry)
            dvTot = dvOut + dvBack
            score = (duRes, dvTot)
            if (best is None) or (score < best[0]):
                best = (score, aD, N)
    return best[1], best[2]

def phasingSequencePropagated(rStartOnTarget, k2, aTgt, N, aDrift):
    tracks = []
    dvTerms = {}
    durations = {}

    vCircTgt = vCirc(aTgt)
    # To drift (Burn 3)
    atOut = 0.5*(aTgt+aDrift); tOut = np.pi*np.sqrt(atOut**3/MU)
    vOut = sqrt(MU*(2/aTgt - 1/atOut)) * unit(np.cross(k2, unit(rStartOnTarget)))
    dvOut = abs(np.linalg.norm(vOut) - vCircTgt)
    _, rs1, rE, vE = rk4Sample(rStartOnTarget, vOut, tOut, steps=500)
    tracks.append(rs1)

    # Circularize at drift (Burn 4)
    rhatE = unit(rE); vCe = vCirc(aDrift) * unit(np.cross(k2, rhatE))
    dvCircDrift = np.linalg.norm(vCe - vE)

    # Drift N revs (coast)
    TD = 2*np.pi*np.sqrt(aDrift**3/MU); tDrift = N*TD
    _, rs2, rF, vF = rk4Sample(rE, vCe, tDrift, steps=max(600, int(300*N)))
    tracks.append(rs2)

    # Leave drift (Burn 5)
    atBack = atOut; tBack = np.pi*np.sqrt(atBack**3/MU)
    vBack = sqrt(MU*(2/aDrift - 1/atBack)) * unit(np.cross(k2, rhatE))
    dvLeave = np.linalg.norm(vBack - vF)
    _, rs3, rG, vG = rk4Sample(rF, vBack, tBack, steps=500)
    tracks.append(rs3)

    # Recircularize at target (Burn 6)
    rhatG = unit(rG); vRecirc = vCirc(aTgt) * unit(np.cross(k2, rhatG))
    dvRecirc = np.linalg.norm(vRecirc - vG)

    dvTerms["dv_out"] = dvOut
    dvTerms["dv_circ_drift"] = dvCircDrift
    dvTerms["dv_leave"] = dvLeave
    dvTerms["dv_recirc"] = dvRecirc
    dvTerms["dv_total"] = dvOut + dvCircDrift + dvLeave + dvRecirc

    durations["t_out"] = tOut
    durations["T_d"] = TD
    durations["N"] = N
    durations["t_drift"] = tDrift
    durations["t_back"] = tBack

    return {"segments": tracks, "end_state": (rG, vRecirc), "dv": dvTerms, "dur": durations}

# -------------------------
# Leg 3: 3D Lambert, arrive-and-circularize (no rendezvous), with Earth clearance
# -------------------------
def leg33dLambertPropagated(
    t0, A, B,
    diag=True,
    clearanceKm=100.0,
    earthRadiusKm=6378.137,
    maxWaitRevs=2.0,          # wait on A before departing, in revs
    waitSteps=36,             # resolution of wait scan
    tofScale=(0.20, 3.50),    # TOF window multiples of 0.5*pi*sqrt(B.a^3/MU)
    tofSteps=141              # resolution of TOF scan
):
    def periapsis_radius(r0, v0, mu=MU):
        r0n = np.linalg.norm(r0)
        hvec = np.cross(r0, v0); h = np.linalg.norm(hvec)
        if h == 0.0: return 0.0
        evec = (np.cross(v0, hvec) / mu) - (r0 / r0n)
        e = np.linalg.norm(evec)
        p = h**2 / mu
        return p / (1.0 + e)

    TA   = periodFromTarget(A)                         # Rp/Ra-aware
    tof0 = 0.5 * np.pi * np.sqrt(a_for_time(B)**3 / MU)
    kB   = planeNormal(B["i"], B["Omega"])

    best = None
    cands = []

    for tw in np.linspace(0.0, maxWaitRevs*TA, waitSteps):
        tDep = t0 + timedelta(seconds=float(tw))
        rDep = rEci(A, tDep)
        vDep = vEci(A, tDep)
        prograde = (np.cross(rDep, vDep)[2] >= 0.0)

        for s in np.linspace(tofScale[0], tofScale[1], tofSteps):
            tof = max(60.0, s*tof0)
            tArr = tDep + timedelta(seconds=float(tof))
            rArr = rEci(B, tArr)

            try:
                v1Tr, v2Tr = lambertUniversal(rDep, rArr, tof, mu=MU, prograde=prograde)
            except Exception:
                continue

            rp = periapsis_radius(rDep, v1Tr, mu=MU)
            if rp < (earthRadiusKm + clearanceKm):
                continue

            rhat = unit(rArr)
            vCirc_Bplane_arr = vCirc(a_for_time(B)) * unit(np.cross(kB, rhat))

            dv_dep_vec  = v1Tr - vDep
            dv_arr_vec  = vCirc_Bplane_arr - v2Tr
            dv_dep      = np.linalg.norm(dv_dep_vec)
            dv_arr_circ = np.linalg.norm(dv_arr_vec)
            dv_tot      = dv_dep + dv_arr_circ

            cand = (dv_tot, tw, tof, tDep, tArr, rDep, rArr, v1Tr, v2Tr, dv_dep_vec, dv_arr_vec, rp)
            cands.append(cand)
            if (best is None) or (dv_tot < best[0]):
                best = cand

    if best is None:
        raise RuntimeError("Leg 3: no collision-free arrive-&-circularize solution. Increase maxWaitRevs or widen tofScale.")

    (dvTot, twBest, tofBest, tDepBest, tArrBest,
     rDepBest, rArrBest, v1Best, v2Best, dvDepVec, dvArrCircVec, rpBest) = best

    # Propagate chosen arc for plotting/verification
    _, rsLam, _, _ = rk4Sample(rDepBest, v1Best, tofBest, steps=900)

    if diag and cands:
        print("\n--- Leg 3 Lambert diagnostics (rp > Earth + clearance) ---")
        print(f"{'wait[min]':>9} {'TOF[min]':>9} {'dv_dep':>9} {'dv_arr':>11} {'dv_tot':>9} {'rp[km]':>10}")

    return {
        "path": rsLam,
        "start_circle": a_for_time(A), "target_circle": a_for_time(B),
        "t_depart": tDepBest, "t_arrival": tArrBest,
        "tof": tofBest, "wait_used_s": twBest,
        "r_dep": rDepBest, "r_arr": rArrBest,
        "rp": rpBest, "clearance_km": clearanceKm,
        "dv_dep_vec": dvDepVec, "dv_arr_circ_vec": dvArrCircVec,
        "dv_dep": np.linalg.norm(dvDepVec),
        "dv_arr_circularize": np.linalg.norm(dvArrCircVec),
        "dv_tot_circularize": dvTot
    }

# -------------------------
# Plotting
# -------------------------
def plotLegWithPhasing(leg, phase, title):
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
    ax.set_title(title + f"\nΔv1={leg['dv1']:.3f} km/s, Δv2={leg['dv2']:.3f} km/s, Δv_phase={phase['dv']['dv_total']:.3f} km/s")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    plt.show()

def plotLeg33d(leg3, title):
    fig, ax = plt.subplots(figsize=(7.6,6.7))
    ax.plot(leg3["path"][:,0], leg3["path"][:,1], linewidth=2, label="Lambert arc (propagated)")
    th = np.linspace(0,2*np.pi,500)
    r1 = leg3["start_circle"]; r2 = leg3["target_circle"]
    ax.plot(r1*np.cos(th), r1*np.sin(th), linestyle="--", label="LEO A circle (proj)")
    ax.plot(r2*np.cos(th), r2*np.sin(th), linestyle="--", label="LEO B circle (proj)")
    ax.scatter([leg3["r_dep"][0]],[leg3["r_dep"][1]], s=30); ax.text(leg3["r_dep"][0], leg3["r_dep"][1], " dep", va="bottom", fontsize=9)
    ax.scatter([leg3["r_arr"][0]],[leg3["r_arr"][1]], s=30); ax.text(leg3["r_arr"][0], leg3["r_arr"][1], " arr", va="bottom", fontsize=9)
    ax.set_aspect("equal"); ax.grid(True)
    ax.set_xlabel("ECI x (km)"); ax.set_ylabel("ECI y (km)")
    ax.set_title(
        title + f"\nΔv dep + circ = {leg3['dv_dep']:.3f} + {leg3['dv_arr_circularize']:.3f} = {leg3['dv_tot_circularize']:.3f} km/s"
                f"   (rp={leg3['rp']:.1f} km, clr≥{leg3['clearance_km']:.0f} km)"
    )
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    plt.show()

# -------------------------
# Reporting helpers
# -------------------------
def fmtTime(dtObj): return dtObj.strftime("%Y-%m-%d %H:%M:%S UTC")

def fmtVec3(v, unit_lbl):
    return f"[{v[0]: .3f}, {v[1]: .3f}, {v[2]: .3f}] {unit_lbl}"

def reportObjectsRV(objs, t, header=None):
    if header:
        print(f"\n-- {header} @ {fmtTime(t)} --")
    for tgt in objs:
        r = rEci(tgt, t)
        v = vEci(tgt, t)
        print(f"{tgt['name']:<30} r= {fmtVec3(r, 'km')}   v= {fmtVec3(v, 'km/s')}")

def reportLegHtPlanePhasing(legLabel, tNow, rStart, iStart, oStart, tgt, leg, phase, allObjs=None):
    theta = planeAngleDeg(iStart, oStart, tgt["i"], tgt["Omega"])
    tArr = tNow + timedelta(seconds=leg["tof"])  # arrival at target ring
    tInt = tArr + timedelta(seconds=phase["dur"]["t_out"] + phase["dur"]["t_drift"] + phase["dur"]["t_back"])  # intercept instant
    tAfter = tInt + timedelta(seconds=5*periodFromTarget(tgt))  # Rp/Ra-aware dwell
    dvLeg = leg["dv1"] + leg["dv2"]

    print(f"\n=== {legLabel} ===")
    print(f"Start (r,i,Ω)                  : ({rStart:,.1f} km, {iStart:.3f}°, {oStart:.3f}°) @ {fmtTime(tNow)}")
    print(f"Target (r,i,Ω,ω,e)             : ({tgt['a']:,.1f} km, {tgt['i']:.3f}°, {tgt['Omega']:.3f}°, {tgt['omega']:.3f}°, {tgt['e']:.5f})")
    print(f"Plane-rotation θ               : {theta:.3f}°")
    print(f"Transfer burns (dv1+dv2)       : {leg['dv1']:.3f} + {leg['dv2']:.3f} = {dvLeg:.3f} km/s")
    print(f"Arrival                        : {fmtTime(tArr)}")
    print(f"Phasing dv (out+circ+leave+recirc) : {phase['dv']['dv_total']:.3f} km/s; intercept {fmtTime(tInt)}")
    print(f"Dwell 5 periods                : until {fmtTime(tAfter)}")

    if allObjs is not None:
        reportObjectsRV(allObjs, tNow, header="Event: leg start (pre-burn)")
        reportObjectsRV(allObjs, tArr, header="Event: leg arrival on target ring")
        reportObjectsRV(allObjs, tInt, header="Event: post-phasing intercept")
    return tAfter

def reportLeg3dLambert(legLabel, tNow, rStart, iStart, oStart, tgt, leg3, allObjs=None):
    theta = planeAngleDeg(iStart, oStart, tgt["i"], tgt["Omega"])
    tDep = leg3["t_depart"]; tArr = leg3["t_arrival"]
    tAfter = tArr + timedelta(seconds=5*periodFromTarget(tgt))  # Rp/Ra-aware dwell
    print(f"\n=== {legLabel} ===")
    print(f"Start (r,i,Ω)                  : ({rStart:,.1f} km, {iStart:.3f}°, {oStart:.3f}°) @ {fmtTime(tDep)}")
    print(f"Target (r,i,Ω,ω,e)             : ({tgt['a']:,.1f} km, {tgt['i']:.3f}°, {tgt['Omega']:.3f}°, {tgt['omega']:.3f}°, {tgt['e']:.5f})")
    print(f"Plane-rotation θ               : {theta:.3f}°  (implicit in Lambert)")
    print(f"Transfer (dep + arrive&circularize): {leg3['dv_dep']:.3f} + {leg3['dv_arr_circularize']:.3f} = {leg3['dv_tot_circularize']:.3f} km/s")
    print(f"Arrival                        : {fmtTime(tArr)}   (rp={leg3['rp']:.1f} km, clr≥{leg3['clearance_km']:.0f} km)")
    print(f"Dwell 5 periods                : until {fmtTime(tAfter)}")

    if allObjs is not None:
        reportObjectsRV(allObjs, tDep, header="Event: Lambert departure")
        reportObjectsRV(allObjs, tArr, header="Event: Lambert arrival & circularize")
    return tAfter

# ===== 3D helpers =====
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _set_axes_equal_3d(ax):
    xs, ys, zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xmid, ymid, zmid = [(a+b)/2 for a,b in (xs, ys, zs)]
    max_range = max(abs(xs[1]-xs[0]), abs(ys[1]-ys[0]), abs(zs[1]-zs[0]))/2
    ax.set_xlim3d([xmid-max_range, xmid+max_range])
    ax.set_ylim3d([ymid-max_range, ymid+max_range])
    ax.set_zlim3d([zmid-max_range, zmid+max_range])

def _orthonormal_basis_from_normal(n):
    n = unit(n)
    trial = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(n, trial))
    v = unit(np.cross(n, u))
    return u, v

def _plot_ring(ax, radius_km, normal_vec, center=np.zeros(3), npts=400, **kwargs):
    u, v = _orthonormal_basis_from_normal(normal_vec)
    th = np.linspace(0.0, 2*np.pi, npts)
    ring = center[None,:] + radius_km*np.cos(th)[:,None]*u[None,:] + radius_km*np.sin(th)[:,None]*v[None,:]
    ax.plot(ring[:,0], ring[:,1], ring[:,2], **kwargs)

def _plot_earth_wireframe(ax, R=6378.137, n=32, **kwargs):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = R*np.outer(np.cos(u), np.sin(v))
    y = R*np.outer(np.sin(u), np.sin(v))
    z = R*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=0.5, alpha=0.3, **kwargs)

# ===== 3D plotters =====
def plotLegWithPhasing3D(leg, phase, iStart_deg, OStart_deg, iTgt_deg, OTgt_deg, title, show_earth=True):
    fig = plt.figure(figsize=(8.4, 7.6))
    ax = fig.add_subplot(111, projection='3d')

    if show_earth:
        _plot_earth_wireframe(ax)

    ax.plot(leg["r_path"][:,0], leg["r_path"][:,1], leg["r_path"][:,2], linewidth=2, label="Transfer (propagated)")
    for j, seg in enumerate(phase["segments"], start=1):
        ax.plot(seg[:,0], seg[:,1], seg[:,2], alpha=0.9, label=f"Phasing seg {j}")

    ax.scatter([leg["r_dep"][0]], [leg["r_dep"][1]], [leg["r_dep"][2]], s=24)
    ax.text(leg["r_dep"][0], leg["r_dep"][1], leg["r_dep"][2], " dep", fontsize=9)
    ax.scatter([leg["r_arr"][0]], [leg["r_arr"][1]], [leg["r_arr"][2]], s=24)
    ax.text(leg["r_arr"][0], leg["r_arr"][1], leg["r_arr"][2], " arr", fontsize=9)

    kStart = planeNormal(iStart_deg, OStart_deg)
    kTgt   = planeNormal(iTgt_deg,   OTgt_deg)
    _plot_ring(ax, leg["a1"], kStart, label="Start circle", linestyle='--')
    _plot_ring(ax, leg["a2"], kTgt,   label="Target circle", linestyle='--')

    ax.set_xlabel("ECI x (km)")
    ax.set_ylabel("ECI y (km)")
    ax.set_zlabel("ECI z (km)")
    ax.set_title(title + f"\nΔv1={leg['dv1']:.3f} km/s, Δv2={leg['dv2']:.3f} km/s, Δv_phase={phase['dv']['dv_total']:.3f} km/s")
    _set_axes_equal_3d(ax)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plotLeg33d3D(leg3, A, B, title, show_earth=True):
    fig = plt.figure(figsize=(8.4, 7.6))
    ax = fig.add_subplot(111, projection='3d')

    if show_earth:
        _plot_earth_wireframe(ax)

    ax.plot(leg3["path"][:,0], leg3["path"][:,1], leg3["path"][:,2], linewidth=2, label="Lambert arc (propagated)")

    ax.scatter([leg3["r_dep"][0]], [leg3["r_dep"][1]], [leg3["r_dep"][2]], s=24)
    ax.text(leg3["r_dep"][0], leg3["r_dep"][1], leg3["r_dep"][2], " dep", fontsize=9)
    ax.scatter([leg3["r_arr"][0]], [leg3["r_arr"][1]], [leg3["r_arr"][2]], s=24)
    ax.text(leg3["r_arr"][0], leg3["r_arr"][1], leg3["r_arr"][2], " arr", fontsize=9)

    kA = planeNormal(A["i"], A["Omega"])
    kB = planeNormal(B["i"], B["Omega"])
    _plot_ring(ax, leg3["start_circle"], kA, label="Start circle", linestyle='--')
    _plot_ring(ax, leg3["target_circle"], kB, label="Target circle", linestyle='--')

    ax.set_xlabel("ECI x (km)")
    ax.set_ylabel("ECI y (km)")
    ax.set_zlabel("ECI z (km)")
    ax.set_title(
        title + f"\nΔv dep + circ = {leg3['dv_dep']:.3f} + {leg3['dv_arr_circularize']:.3f} = {leg3['dv_tot_circularize']:.3f} km/s"
                f"   (rp={leg3['rp']:.1f} km, clr≥{leg3['clearance_km']:.0f} km)"
    )
    _set_axes_equal_3d(ax)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# -------------------------
# Objects and run
# -------------------------
# If you have known perigee/apogee *radii* (km), include them below to drive timing via Rp/Ra.
# The example here uses altitudes (487–589 km, 7666–7689 km, 35722–35759 km) converted to radii by adding Earth radius 6378.137 km.
objs = [
 dict(name="INTELSAT 33E DEB (61998)", epoch=25316.14507882, inc=0.9979,  raan=85.7856,  ecc=0.0009167, argp=252.6559, M=21.5383,  n=1.00437626,
      rp_km=6378.137+35722.0, ra_km=6378.137+35759.0),
 dict(name="FREGAT R/B (39192)",      epoch=25316.66227286, inc=0.0773,  raan=183.5260, ecc=0.0008031, argp=266.3417, M=270.2279, n=5.20983481235749,
      rp_km=6378.137+7666.0,  ra_km=6378.137+7689.0),
 dict(name="OBJECT 33393U 08048A",     epoch=25316.56274537, inc=9.3468,  raan=320.0996, ecc=0.0009937, argp=135.6367, M=224.4630, n=14.952780489390428,
      rp_km=6378.137+487.0,   ra_km=6378.137+523.0),
 dict(name="OBJECT 52939U 22072E",     epoch=25316.14146735, inc=9.9523,  raan=276.8703, ecc=0.0026188, argp=221.3777, M=138.4452, n=15.20682587186371,
      rp_km=6378.137+575.0,   ra_km=6378.137+589.0),
]

T = [targetFromTleFields(
        o["name"], o["epoch"], o["inc"], o["raan"], o["ecc"], o["argp"], o["M"], o["n"],
        rp_km=o.get("rp_km"), ra_km=o.get("ra_km")
    ) for o in objs]

start = next(t for t in T if "INTELSAT 33E" in t["name"])
freg  = next(t for t in T if "FREGAT" in t["name"])
leoA  = next(t for t in T if "33393U" in t["name"])
leoB  = next(t for t in T if "52939U" in t["name"])

if __name__ == "__main__":
    # Common start at latest epoch
    t0 = max(t["epoch"] for t in T)

    print("=== MISSION START (propagated legs 1–3; timing via Rp/Ra) ===")
    print("Start on INTELSAT; visit FREGAT, then 33393U, then 52939U.")

    # Initial epochs for each object
    print("\n=== Initial epochs (per TLE) ===")
    for t in T:
        print(f"{t['name']:<30} epoch: {fmtTime(t['epoch'])}")

    # r/v for each object at t0
    reportObjectsRV(T, t0, header="Initial state at common start time (t0)")

    # ---- Leg 0: dwell on INTELSAT (5 periods, Rp/Ra-based) ----
    T0 = periodFromTarget(start)
    tAfter0 = t0 + timedelta(seconds=5*T0)
    print("\n=== Dwell at INTELSAT (5 periods, Rp/Ra timing) ===")
    print(f"Start (r,i,Ω)                  : ({start['a']:,.1f} km, {start['i']:.3f}°, {start['Omega']:.3f}°) @ {fmtTime(t0)}")
    print(f"Dwell 5 periods                : until {fmtTime(tAfter0)}")
    reportObjectsRV(T, tAfter0, header="Event: end of dwell on INTELSAT")

    # ---- Leg 1 ----
    aStart = a_for_time(start)
    aFreg  = a_for_time(freg)
    leg1 = hohmannWithPlanechangePropagated(aStart, start["i"], start["Omega"],
                                            aFreg,  freg["i"],  freg["Omega"])
    aTgt1 = a_for_time(freg)
    aD1, N1 = chooseDrift(aTgt1)
    phase1 = phasingSequencePropagated(leg1["r_arr"], leg1["k2"], aTgt1, N1, aD1)
    tAfter1 = reportLegHtPlanePhasing(
        "INTELSAT → FREGAT (HT+plane on Burn 1) + phasing",
        tAfter0, aStart, start["i"], start["Omega"], freg, leg1, phase1, allObjs=T
    )
    plotLegWithPhasing(leg1, phase1, "Leg 1: INTELSAT → FREGAT (HT+plane on Burn 1) + phasing")

    # ---- Leg 2 ----
    aLeoA  = a_for_time(leoA)
    leg2 = hohmannWithPlanechangePropagated(aFreg,  freg["i"],  freg["Omega"],
                                            aLeoA,  leoA["i"],  leoA["Omega"])
    aTgt2 = a_for_time(leoA)
    aD2, N2 = chooseDrift(aTgt2)
    phase2 = phasingSequencePropagated(leg2["r_arr"], leg2["k2"], aTgt2, N2, aD2)
    tAfter2 = reportLegHtPlanePhasing(
        "FREGAT → 33393U (HT+plane on Burn 1) + phasing",
        tAfter1, aFreg, freg["i"], freg["Omega"], leoA, leg2, phase2, allObjs=T
    )
    plotLegWithPhasing(leg2, phase2, "Leg 2: FREGAT → 33393U (HT+plane on Burn 1) + phasing")

    # ---- Leg 3 ----
    leg3 = leg33dLambertPropagated(tAfter2, leoA, leoB, diag=True, clearanceKm=100.0)
    tAfter3 = reportLeg3dLambert(
        "33393U → 52939U (3D Lambert; arrive & circularize)",
        tAfter2, aLeoA, leoA["i"], leoA["Omega"], leoB, leg3, allObjs=T
    )
    plotLeg33d(leg3, "Leg 3: 33393U → 52939U (arrive & circularize)")

    reportObjectsRV(T, tAfter3, header="Final positions (after final dwell)")

    # Δv tallies
    dvLeg1 = leg1["dv1"] + leg1["dv2"] + phase1["dv"]["dv_total"]
    dvLeg2 = leg2["dv1"] + leg2["dv2"] + phase2["dv"]["dv_total"]
    dvLeg3 = leg3["dv_tot_circularize"]
    dvTotal = dvLeg1 + dvLeg2 + dvLeg3

    print("\n=== Δv SUMMARY ===")
    print(f"Leg 1 (transfer + phasing): {dvLeg1:.3f} km/s  "
          f"[transfer={leg1['dv1']+leg1['dv2']:.3f}, phasing={phase1['dv']['dv_total']:.3f}]")
    print(f"Leg 2 (transfer + phasing): {dvLeg2:.3f} km/s  "
          f"[transfer={leg2['dv1']+leg2['dv2']:.3f}, phasing={phase2['dv']['dv_total']:.3f}]")
    print(f"Leg 3 (dep + circularize):  {dvLeg3:.3f} km/s")
    print(f"TOTAL mission Δv:           {dvTotal:.3f} km/s")

    # 3D visualizations
    plotLegWithPhasing3D(
        leg1, phase1,
        iStart_deg=start["i"], OStart_deg=start["Omega"],
        iTgt_deg=freg["i"],   OTgt_deg=freg["Omega"],
        title="Leg 1: INTELSAT → FREGAT (HT+plane on Burn 1) + phasing"
    )
    plotLegWithPhasing3D(
        leg2, phase2,
        iStart_deg=freg["i"],  OStart_deg=freg["Omega"],
        iTgt_deg=leoA["i"],    OTgt_deg=leoA["Omega"],
        title="Leg 2: FREGAT → 33393U (HT+plane on Burn 1) + phasing"
    )
    plotLeg33d3D(
        leg3, A=leoA, B=leoB,
        title="Leg 3: 33393U → 52939U (arrive & circularize)"
    )

    # --- Optional timestamp check block (manual snapshot) ---
    manual_dt = 10.861*24*3600  # 938390.4 s
    t_manual = t0 + timedelta(seconds=manual_dt)
    print("\n=== Timestamp check ===")
    print("t_manual (10.861 d after t0):", fmtTime(t_manual))
    print("tAfter3 (final dwell end):   ", fmtTime(tAfter3))
    print("Δt (s):", (t_manual - tAfter3).total_seconds())

    print("\n=== Final positions (manual 10.861 d after t0) ===")
    for tgt in T:
        r = rEci(tgt, t_manual); v = vEci(tgt, t_manual)
        print(f"{tgt['name']:<30} r= [{r[0]: .6f}, {r[1]: .6f}, {r[2]: .6f}] km")
        print(f"{'':30} v= [{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}] km/s")
