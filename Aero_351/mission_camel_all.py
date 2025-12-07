
from math import pi, sqrt, sin, cos
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Constants
# -------------------------
MU = 398600.4418  # km^3/s^2
SEC_PER_DAY = 86400.0

"""
I learned my lesson about not building up my library as the quarter went on, this project had me worked to the bone building up all
the definitions - Juan
"""

# -------------------------
# Tiny helpers
# -------------------------
def vCirc(r): return sqrt(MU / r)
def periodFromA(a): return 2*pi*sqrt(a**3/MU)
def wrap2pi(x): return np.mod(x, 2*np.pi)
def wrapPi(x): return (x + np.pi) % (2*np.pi) - np.pi

def solveEFromM(M, e, tol=1e-12, it=60):
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
    year = (2000 + yy) if yy < 57 else (1900 + yy)
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

# Target accessors
def targetFromTLEFields(name, epoch, inc, raan, e, argp, Mdeg, nRevPerDay):
    MU = 398600.4418
    SEC_PER_DAY = 86400.0
    nRadS = nRevPerDay * 2*np.pi / SEC_PER_DAY
    a = (MU / nRadS**2)**(1/3)
    return dict(name=name, epoch=tleEpochToTime(epoch), a=a, e=float(e),
                i=float(inc), Omega=float(raan), omega=float(argp),
                M0=float(Mdeg), n=nRadS)

def MAt(tgt, t):
    dt = (t - tgt["epoch"]).totalSeconds()
    return wrap2pi(np.deg2rad(tgt["M0"]) + tgt["n"]*dt)

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
# Combined-plane Hohmann + propagated phasing
# -------------------------
def unit(v): v = np.array(v, dtype=float); return v/np.linalg.norm(v)

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

def hohmannWithPlanechangePropagated(a1, i1, O1, a2, i2, O2):
    rhat, k1, k2 = nodeLine(i1, O1, i2, O2)
    r0, v0 = circStateOnLine(a1, k1, rhat)
    at = 0.5*(a1+a2)
    v1Speed = sqrt(MU*(2/a1 - 1/at))
    v1Dir   = unit(np.cross(k2, rhat))  # tangent in target plane
    v1       = v1Speed * v1Dir
    dv1      = np.linalg.norm(v1 - v0)
    tof = pi*sqrt(at**3/MU)
    ts, rs, rEnd, vEnd = rk4Sample(r0, v1, tof, steps=1000)
    v2Circ = vCirc(a2) * unit(np.cross(k2, -rhat))
    dv2 = np.linalg.norm(v2Circ - vEnd)
    return {
        "r_path": rs, "tof": tof,
        "r_dep": r0, "v_dep_pre": v0, "v_dep_post": v1, "dv1": dv1,
        "r_arr": rEnd, "v_arr_pre": vEnd, "v_arr_post": v2Circ, "dv2": dv2,
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
    atOut = 0.5*(aTgt+aDrift); tOut = pi*sqrt(atOut**3/MU)
    vOut = sqrt(MU*(2/aTgt - 1/atOut)) * unit(np.cross(k2, unit(rStartOnTarget)))
    dvOut = abs(np.linalg.norm(vOut) - vCircTgt)
    _, rs1, rE, vE = rk4Sample(rStartOnTarget, vOut, tOut, steps=500)
    tracks.append(rs1)
    # Circularize at drift (Burn 4)
    rhatE = unit(rE); vCe = vCirc(aDrift) * unit(np.cross(k2, rhatE))
    dvCircDrift = np.linalg.norm(vCe - vE)
    # Drift N revs (coast)
    TD = 2*pi*sqrt(aDrift**3/MU); tDrift = N*TD
    _, rs2, rF, vF = rk4Sample(rE, vCe, tDrift, steps=max(600, int(300*N)))
    tracks.append(rs2)
    # Leave drift (Burn 5)
    atBack = atOut; tBack = pi*sqrt(atBack**3/MU)
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
# Leg 3: fully 3D Lambert (no explicit plane change) with diagnostics
# -------------------------
def leg33dLambertPropagated(t0, A, B, diag=True):
    r1 = A["a"]; r2 = B["a"]
    rhat, kA, kB = nodeLine(A["i"], A["Omega"], B["i"], B["Omega"])
    rDep, vDep = circStateOnLine(r1, kA, rhat)  # pre-burn circular in A

    # orbital sense for "prograde" hint
    hz = np.cross(rDep, vDep)[2]
    prograde = True if hz >= 0 else False

    tof0 = 0.5*pi*sqrt(r2**3/MU)
    candidates = []
    best = None
    for s in np.linspace(0.25, 2.0, 71):  # widened scan window
        tof = max(60.0, s*tof0)
        tArr = t0 + timedelta(seconds=tof)
        rTgtArr = rEci(B, tArr)
        vTgtArr = vEci(B, tArr)
        try:
            v1Tr, v2Tr = lambertUniversal(rDep, rTgtArr, tof, mu=MU, prograde=prograde)
        except Exception:
            continue
        # diagnostics
        cd = np.clip(np.dot(rDep, rTgtArr)/(np.linalg.norm(rDep)*np.linalg.norm(rTgtArr)), -1.0, 1.0)
        ang = np.degrees(np.arccos(cd))
        dvDep = np.linalg.norm(v1Tr - vDep)
        dvArr = np.linalg.norm(vTgtArr - v2Tr)
        dvTot = dvDep + dvArr
        candidates.append((tof, ang, dvDep, dvArr, dvTot))
        if (best is None) or (dvTot < best[0]):
            best = (dvTot, tof, tArr, v1Tr, v2Tr, dvDep, dvArr)

    if best is None:
        raise RuntimeError("Lambert scan failed for 3D leg.")

    # Print diagnostics table
    if diag and candidates:
        print("\n--- Leg 3 Lambert diagnostics (scan over TOF) ---")
        print(f"{'TOF [min]':>10}  {'Δθ[r1→r2] [deg]':>15}  {'dv_dep [km/s]':>13}  {'dv_arr [km/s]':>13}  {'dv_tot [km/s]':>13}")
        for tof, ang, dvDep, dvArr, dvTot in candidates:
            print(f"{tof/60:10.2f}  {ang:15.2f}  {dvDep:13.3f}  {dvArr:13.3f}  {dvTot:13.3f}")

    dvTotal, tofBest, tArrival, v1Tr, v2Tr, dvDep, dvArr = best
    _, rsLam, rEnd, vEnd = rk4Sample(rDep, v1Tr, tofBest, steps=900)

    return {
        "path": rsLam,
        "dv_dep": dvDep, "dv_arr": dvArr, "dv_tot": dvTotal,
        "start_circle": r1, "target_circle": r2,
        "tof": tofBest, "t_arrival": tArrival
    }

def plotLeg33d(leg3, title):
    fig, ax = plt.subplots(figsize=(7.6,6.7))
    ax.plot(leg3["path"][:,0], leg3["path"][:,1], linewidth=2, label="Lambert arc (propagated)")
    th = np.linspace(0,2*np.pi,500)
    r1 = leg3["start_circle"]; r2 = leg3["target_circle"]
    ax.plot(r1*np.cos(th), r1*np.sin(th), linestyle="--", label="LEO A circle (proj)")
    ax.plot(r2*np.cos(th), r2*np.sin(th), linestyle="--", label="LEO B circle (proj)")
    ax.setAspect("equal"); ax.grid(True)
    ax.setXlabel("ECI x (km)"); ax.setYlabel("ECI y (km)")
    ax.setTitle(title + f"\nΔv_Lambert dep/arr={leg3['dv_dep']:.3f}/{leg3['dv_arr']:.3f} km/s, Total={leg3['dv_tot']:.3f} km/s")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    plt.show()

# -------------------------
# PLOTTING
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
    ax.setAspect("equal"); ax.grid(True)
    ax.setXlabel("ECI x (km)"); ax.setYlabel("ECI y (km)")
    ax.setTitle(title + f"\nΔv1={leg['dv1']:.3f} km/s, Δv2={leg['dv2']:.3f} km/s, Δv_phase={phase['dv']['dv_total']:.3f} km/s")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    plt.show()

# -------------------------
# Reporting helpers
# -------------------------
def fmtTime(dtObj):
    return dtObj.strftime("%Y-%m-%d %H:%M:%S UTC")

def reportLegHtPlanePhasing(legLabel, tNow, rStart, iStart, oStart, tgt, leg, phase):
    theta = planeAngleDeg(iStart, oStart, tgt["i"], tgt["Omega"])
    tArr = tNow + timedelta(seconds=leg["tof"])
    tInt = tArr + timedelta(seconds=phase["dur"]["t_out"] + phase["dur"]["t_drift"] + phase["dur"]["t_back"])
    tAfter = tInt + timedelta(seconds=5*periodFromA(tgt["a"]))
    dvLeg = leg["dv1"] + leg["dv2"]

    print(f"\n=== {legLabel} ===")
    print(f"Start (r,i,Ω)                  : ({rStart:,.1f} km, {iStart:.3f}°, {OSError:.3f}°) @ {fmtTime(tNow)}")
    print(f"Target (r,i,Ω,ω,e)             : ({tgt['a']:,.1f} km, {tgt['i']:.3f}°, {tgt['Omega']:.3f}°, {tgt['omega']:.3f}°, {tgt['e']:.5f})")
    print(f"Plane-rotation θ               : {theta:.3f}°")
    print(f"Transfer burns (dv1+dv2)       : {leg['dv1']:.3f} + {leg['dv2']:.3f} = {dvLeg:.3f} km/s")
    print(f"Arrival                        : {fmtTime(tArr)}")
    print(f"Phasing dv (out+circ+leave+recirc) : {phase['dv']['dv_total']:.3f} km/s; intercept {fmtTime(tInt)}")
    print(f"Dwell 5 periods                : until {fmtTime(tAfter)}")
    return tAfter

def reportLeg3dLambert(legLabel, tNow, rStart, iStart, oStart, tgt, leg3):
    theta = planeAngleDeg(iStart, oStart, tgt["i"], tgt["Omega"])
    tArr = tNow + timedelta(seconds=leg3["tof"])
    tAfter = tArr + timedelta(seconds=5*periodFromA(tgt["a"]))
    print(f"\n=== {legLabel} ===")
    print(f"Start (r,i,Ω)                  : ({rStart:,.1f} km, {iStart:.3f}°, {oStart:.3f}°) @ {fmtTime(tNow)}")
    print(f"Target (r,i,Ω,ω,e)             : ({tgt['a']:,.1f} km, {tgt['i']:.3f}°, {tgt['Omega']:.3f}°, {tgt['omega']:.3f}°, {tgt['e']:.5f})")
    print(f"Plane-rotation θ               : {theta:.3f}°  (implicit in Lambert burns)")
    print(f"Transfer burns (dep+arr)       : {leg3['dv_dep']:.3f} + {leg3['dv_arr']:.3f} = {leg3['dv_tot']:.3f} km/s")
    print(f"Arrival                        : {fmtTime(tArr)}")
    print(f"Dwell 5 periods                : until {fmtTime(tAfter)}")
    return tAfter

# -------------------------
# Objects and run
# -------------------------
objs = [
 dict(name="INTELSAT 33E DEB (61998)", epoch=25316.14507882, inc=0.9979,  raan=85.7856,  ecc=0.0009167, argp=252.6559, M=21.5383,  n=1.00437626),
 dict(name="FREGAT R/B (39192)",      epoch=25316.66227286, inc=0.0773,  raan=183.5260, ecc=0.0008031, argp=266.3417, M=270.2279, n=5.20983481235749),
 dict(name="OBJECT 33393U 08048A",     epoch=25316.56274537, inc=9.3468,  raan=320.0996, ecc=0.0009937, argp=135.6367, M=224.4630, n=14.952780489390428),
 dict(name="OBJECT 52939U 22072E",     epoch=25316.14146735, inc=9.9523,  raan=276.8703, ecc=0.0026188, argp=221.3777, M=138.4452, n=15.20682587186371),
]
T = [targetFromTLEFields(o["name"], o["epoch"], o["inc"], o["raan"], o["ecc"], o["argp"], o["M"], o["n"]) for o in objs]
start = next(t for t in T if "INTELSAT 33E" in t["name"])
freg  = next(t for t in T if "FREGAT" in t["name"])
leoA  = next(t for t in T if "33393U" in t["name"])
leoB  = next(t for t in T if "52939U" in t["name"])

if __name__ == "__main__":
    # Common start at latest epoch
    t0 = max(t["epoch"] for t in T)

    print("=== MISSION START (propagated legs 1–3) ===")
    print("Start on INTELSAT; visit FREGAT, then 33393U, then 52939U.")

    # ---- Leg 1 ----
    leg1 = hohmannWithPlanechangePropagated(start["a"], start["i"], start["Omega"],
                                               freg["a"],  freg["i"],  freg["Omega"])
    aD1, N1 = chooseDrift(freg["a"])
    phase1 = phasingSequencePropagated(leg1["r_arr"], leg1["k2"], freg["a"], N1, aD1)
    tAfter1 = reportLegHtPlanePhasing(
        "INTELSAT → FREGAT (HT+plane on Burn 1) + phasing",
        t0, start["a"], start["i"], start["Omega"], freg, leg1, phase1
    )
    plotLegWithPhasing(leg1, phase1, "Leg 1: INTELSAT → FREGAT (HT+plane on Burn 1) + phasing")

    # ---- Leg 2 ----
    leg2 = hohmannWithPlanechangePropagated(freg["a"], freg["i"], freg["Omega"],
                                               leoA["a"],  leoA["i"],  leoA["Omega"])
    aD2, N2 = chooseDrift(leoA["a"])
    phase2 = phasingSequencePropagated(leg2["r_arr"], leg2["k2"], leoA["a"], N2, aD2)
    tAfter2 = reportLegHtPlanePhasing(
        "FREGAT → 33393U (HT+plane on Burn 1) + phasing",
        tAfter1, freg["a"], freg["i"], freg["Omega"], leoA, leg2, phase2
    )
    plotLegWithPhasing(leg2, phase2, "Leg 2: FREGAT → 33393U (HT+plane on Burn 1) + phasing")

    # ---- Leg 3 ----
    leg3 = leg33dLambertPropagated(tAfter2, leoA, leoB, diag=True)
    tAfter3 = reportLeg3dLambert(
        "33393U → 52939U (3D Lambert; no explicit plane change)",
        tAfter2, leoA["a"], leoA["i"], leoA["Omega"], leoB, leg3
    )
    plotLeg33d(leg3, "Leg 3: 33393U → 52939U (3D Lambert; no explicit plane change)")

    # Δv tallies
    dvLeg1 = leg1["dv1"] + leg1["dv2"] + phase1["dv"]["dv_total"]
    dvLeg2 = leg2["dv1"] + leg2["dv2"] + phase2["dv"]["dv_total"]
    dvLeg3 = leg3["dv_dep"] + leg3["dv_arr"]
    dvTotal = dvLeg1 + dvLeg2 + dvLeg3

    print("\\n=== Δv SUMMARY ===")
    print(f"Leg 1 (transfer + phasing): {dvLeg1:.3f} km/s  [transfer={leg1['dv1']+leg1['dv2']:.3f}, phasing={phase1['dv']['dv_total']:.3f}]")
    print(f"Leg 2 (transfer + phasing): {dvLeg2:.3f} km/s  [transfer={leg2['dv1']+leg2['dv2']:.3f}, phasing={phase2['dv']['dv_total']:.3f}]")
    print(f"Leg 3 (3D Lambert dep/arr): {dvLeg3:.3f} km/s")
    print(f"TOTAL mission Δv: {dvTotal:.3f} km/s")
