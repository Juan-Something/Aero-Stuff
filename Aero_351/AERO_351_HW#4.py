import matplotlib as mpL
import scipy as sciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ==========================
# GLOBAL CONSTANTS
# ==========================

muSun   = 1.32712e11      # km^3/s^2
muJup   = 1.26686e8       # km^3/s^2
muEarth = 398600          # km^3/s^2
muVenus = 324859
muMars  = 42828           # km^3/s^2

RS2J = 778.6e6            # km
RS2E = 149.6e6            # km

rEarth   = 6378           # km
rSaturn  = 1.433e9        # km
rNeptune = 4.495e9        # km
rUranus  = 2.872e9        # km
rPluto = 1187             # km
rMars    = 3390           # km
rVenus = 6052

massSaturn  = 568.25e24   # kg
massSun     = 1.989e30    # kg
massNeptune = 102.4e24    # kg
massUranus  = 86.83e24    # kg
massPluto = 13.03e21      # kg

# ==========================
# 8.2
# ==========================
print("-----Question 8.2-----")

RS2M = 227.9e6
deltaVMars = np.sqrt(muSun/RS2M) * (np.sqrt(2*RS2J/(RS2M+RS2J)) - 1)
deltaVJup  = np.sqrt(muSun/RS2J) * (1 - np.sqrt(2*RS2M/(RS2M+RS2J)))

totalDeltaV1 = deltaVMars + deltaVJup
print(f"Total Delta V for Hohmann Transfer from Mars Orbit to Jupiter Orbit: {totalDeltaV1: .4f} km/s")

# ==========================
# 8.4
# ==========================
print("-----Question 8.4-----")

MarsPeriodToEarth    = 687.054
JupiterPeriodToEarth = 4.3319e3
synodicPeriod = ((MarsPeriodToEarth * JupiterPeriodToEarth) /
                 np.abs(MarsPeriodToEarth - JupiterPeriodToEarth))
print(f"Synodic Period of Mars and Jupiter is: {synodicPeriod: .4f} days")
print(synodicPeriod / 365.25)

# ==========================
# 8.6
# ==========================
print("-----Question 8.6-----")

saturnSphereOfInfluence = rSaturn  * (massSaturn/massSun)**(2/5)
uranusSphereOfInfluence = rUranus  * (massUranus/massSun)**(2/5)
neptuneSphereOfInfluence = rNeptune * (massNeptune/massSun)**(2/5)
plutoSphereOfInfluence = rPluto * (massPluto/massSun)**(2/5)

print(f"Saturn's Sphere of Influence: {saturnSphereOfInfluence: .4f} km ")
print(f"Uranus's Sphere of Influence: {uranusSphereOfInfluence: .4f} km ")
print(f"Neptune's Sphere of Influence: {neptuneSphereOfInfluence: .4f} km ")
print(f"Pluto's Sphere of Influence: {plutoSphereOfInfluence: .4f} km ")


# ==========================
# 8.7
# ==========================
print("-----Question 8.7-----")

rPSC = 120e6
rE2S = 147.4e6

vHelioSC   = np.sqrt(muSun*(2/rE2S - 2/(rE2S + rPSC)))
vEarth     = np.sqrt(muSun/rE2S)
vInfExcess = vEarth - vHelioSC
vP         = np.sqrt(vInfExcess**2 + 2*muEarth/(rEarth + 200))
vC         = np.sqrt(muEarth / (rEarth + 200))
deltaV     = vP - vC

print(f"Delta V required for spacecraft to enter Hohmann Transfer Orbit from Earth Parking Orbit {deltaV: .4f} km/s")
print(f"Hyperbolic Excess Speed: {vInfExcess: .4f} km/s")

# ==========================
# 8.12
# ==========================
print("-----Question 8.12-----")

radiusJupiter = 71490
flybyAlt      = 200000 + radiusJupiter

eccTransfer = (RS2J - RS2E) / (RS2J + RS2E)
vPlanet     = np.sqrt(muSun/RS2J)
a           = (RS2J + RS2E) / 2
h           = np.sqrt(a * muSun * (1 - eccTransfer**2))
vTrans      = h / RS2J

vInf    = vTrans - vPlanet
vInfVec = np.array([vInf, 0.0])

vPHyper = np.sqrt(vInf**2 + 2 * muJup/flybyAlt)
hHyper  = vPHyper * flybyAlt
eccHyper = (hHyper**2 / (flybyAlt * muJup)) - 1
d        = -2 * np.asin(1/eccHyper)

# Delta V should be ~10.57 km/s
vInf2 = np.array([vInf*np.cos(d), vInf*np.sin(d)])

deltaVDir = vInf2 - vInfVec
deltaVMag = np.linalg.norm(deltaVDir)

print(f"Delta V imparted by Jupiter Fly-By {deltaVMag: .4f} km/s")

# ==========================
# 8.16
# ==========================
print("-----Question 8.16-----")

# --------------------------
# CONSTANTS FOR 8.16
# --------------------------
AU_KM = 149597870.7  # km

EMB_ELEMS = dict(
    a0=1.00000261,  adot=0.00000562,
    e0=0.01671123,  edot=-0.00004392,
    I0=-0.00001531, Idot=-0.01294668,
    L0=100.46457166, Ldot=35999.37244981,
    varpi0=102.93768193, varpidot=0.32327364,
    Omega0=0.0, Omegadot=0.0
)

MARS_ELEMS = dict(
    a0=1.52371034,  adot=0.00001847,
    e0=0.09339410,  edot=0.00007882,
    I0=1.84969142,  Idot=-0.00813131,
    L0=-4.55343205, Ldot=19140.30268499,
    varpi0=-23.94362959, varpidot=0.44441088,
    Omega0=49.55953891, Omegadot=-0.29257343
)

VENUS_ELEMS = dict(
    a0=0.72333566,  adot=-0.00000390,
    e0=0.00677672,  edot=-0.00004107,
    I0=3.39467605,  Idot=-0.00078890,
    L0=181.97909950, Ldot=58517.81538729,
    varpi0=131.60246718, varpidot=0.002683,
    Omega0=76.67984255, Omegadot=-0.27769418,
)

# --------------------------
# TIME / ELEMENT PROPAGATION
# --------------------------
def julian_date(year, month, day, hour=0, minute=0, second=0.0):
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + A // 4
    jd_day = int(365.25*(year + 4716)) + int(30.6001*(month + 1)) + day + B - 1524.5
    frac = (hour + minute/60 + second/3600) / 24.0
    return jd_day + frac

def planetary_elements(elem, jd):
    T = (jd - 2451545.0)/36525.0  # Julian centuries from J2000

    a_au = elem['a0'] + elem['adot']*T
    e    = elem['e0'] + elem['edot']*T
    I    = (elem['I0'] + elem['Idot']*T) * math.pi/180.0
    L    = (elem['L0'] + elem['Ldot']*T) * math.pi/180.0
    varpi = (elem['varpi0'] + elem['varpidot']*T) * math.pi/180.0
    Omega = (elem['Omega0'] + elem['Omegadot']*T) * math.pi/180.0

    omega = varpi - Omega
    M = L - varpi
    M = (M + math.pi) % (2*math.pi) - math.pi  # normalize

    a_km = a_au * AU_KM
    return a_km, e, I, Omega, omega, M

def solve_kepler(M, e, tol=1e-10, maxiter=50):
    E = M if abs(e) < 0.8 else math.pi
    for _ in range(maxiter):
        f  = E - e*math.sin(E) - M
        fp = 1 - e*math.cos(E)
        dE = -f/fp
        E += dE
        if abs(dE) < tol:
            break
    return E

def coe_to_rv(a, e, I, Omega, omega, M, mu):
    E = solve_kepler(M, e)
    cosE = math.cos(E)
    sinE = math.sin(E)

    r_mag = a*(1 - e*cosE)

    cos_nu = (cosE - e)/(1 - e*cosE)
    sin_nu = (math.sqrt(1 - e**2)*sinE)/(1 - e*cosE)
    nu = math.atan2(sin_nu, cos_nu)

    p = a*(1 - e**2)

    r_pf = np.array([r_mag*math.cos(nu), r_mag*math.sin(nu), 0.0])
    v_pf = np.array([
        -math.sqrt(mu/p)*math.sin(nu),
        math.sqrt(mu/p)*(e + math.cos(nu)),
        0.0
    ])

    cO, sO = math.cos(Omega), math.sin(Omega)
    ci, si = math.cos(I), math.sin(I)
    co, so = math.cos(omega), math.sin(omega)

    R = np.array([
        [cO*co - sO*so*ci, -cO*so - sO*co*ci, sO*si],
        [sO*co + cO*so*ci, -sO*so + cO*co*ci, -cO*si],
        [so*si,            co*si,             ci]
    ])

    r_vec = R @ r_pf
    v_vec = R @ v_pf
    return r_vec, v_vec
def stumpff_C(z):
    if z > 0:
        sz = math.sqrt(z)
        return (1 - math.cos(sz))/z
    elif z < 0:
        sz = math.sqrt(-z)
        return (math.cosh(sz) - 1)/(-z)
    else:
        return 0.5

def stumpff_S(z):
    if z > 0:
        sz = math.sqrt(z)
        return (sz - math.sin(sz))/(sz**3)
    elif z < 0:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz)/(sz**3)
    else:
        return 1.0/6.0

def lambertVallado(r1_vec, r2_vec, dt, mu, prograde=True,
                    maxiter=100, tol=1e-8):
    """
    Vallado-style Lambert solver (single revolution, short way).
    Returns v1, v2 in same frame as r1_vec, r2_vec, my original bisection method really did not want to cooperate so I went hunting for a better functioning one
    """
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    cross_r1r2 = np.cross(r1_vec, r2_vec)
    cos_dnu = np.dot(r1_vec, r2_vec)/(r1*r2)
    cos_dnu = max(-1.0, min(1.0, cos_dnu))
    sin_dnu = np.linalg.norm(cross_r1r2)/(r1*r2)

    if prograde:
        if cross_r1r2[2] < 0:
            sin_dnu = -sin_dnu
    else:
        if cross_r1r2[2] > 0:
            sin_dnu = -sin_dnu

    dnu = math.atan2(sin_dnu, cos_dnu)

    A = sin_dnu * math.sqrt(r1*r2/(1 - cos_dnu))
    if abs(A) < 1e-12:
        raise RuntimeError("Transfer angle too small or ~180, A≈0.")

    def y_z(z):
        C = stumpff_C(z)
        S = stumpff_S(z)
        return r1 + r2 + A*(z*S - 1)/math.sqrt(C)

    def F_z(z):
        C = stumpff_C(z)
        S = stumpff_S(z)
        y = y_z(z)
        if y < 0:
            return 1e9 + y
        return (y/C)**1.5 * S + A*math.sqrt(y) - math.sqrt(mu)*dt

    z = 0.0
    for _ in range(maxiter):
        F = F_z(z)
        if abs(F) < tol:
            break
        dz = 1e-5
        Fp = F_z(z + dz)
        Fm = F_z(z - dz)
        dFdz = (Fp - Fm)/(2*dz)
        if abs(dFdz) < 1e-14:
            break
        z_new = z - F/dFdz
        if abs(z_new) > 1e3:
            z_new = 0.0
        z = z_new

    C = stumpff_C(z)
    y = y_z(z)
    f    = 1 - y/r1
    g    = A*math.sqrt(y/mu)
    gdot = 1 - y/r2

    v1 = (r2_vec - f*r1_vec)/g
    v2 = (gdot*r2_vec - r1_vec)/g
    return v1, v2


# ==========================
# MAIN PROBLEM
# ==========================

# 1) Epochs
jd_dep = julian_date(2005, 8, 15)
jd_arr = julian_date(2006, 3, 15)
tof    = (jd_arr - jd_dep) * 86400.0   # s

jd_dep2 = julian_date(2025,1, 1)
jd_arr2 = julian_date(2025,9,1)

tof2 = (jd_arr2 - jd_dep2) * 86400.0

# 2) Earth & Mars heliocentric states
aE, eE, iE, OE, omE, ME = planetary_elements(EMB_ELEMS, jd_dep)
aM, eM, iM, OM, omM, MM = planetary_elements(MARS_ELEMS, jd_arr)

aE2, eE2, iE2, OE2, omE2, ME2 = planetary_elements(EMB_ELEMS, jd_dep2)
aV, eV, iV, OV, omV, MV = planetary_elements(VENUS_ELEMS, jd_arr2)

rE2, vE2 = coe_to_rv(aE2, eE2, iE2, OE2, omE2, ME2, muSun)
rV, vV = coe_to_rv(aV, eV, iV, OV, omV, MV, muSun)

rE, vE = coe_to_rv(aE, eE, iE, OE, omE, ME, muSun)
rM, vM = coe_to_rv(aM, eM, iM, OM, omM, MM, muSun)

# 3) Lambert Sun-centered transfer
v_dep, v_arr = lambertVallado(rE, rM, tof, muSun, prograde=True)
v_dep2, v_arr2 = lambertVallado(rE2, rV, tof2, muSun, prograde = True)

# 4) Earth departure Δv: 190 km LEO
r_LEO       = rEarth + 190.0
v_circ_LEO  = math.sqrt(muEarth / r_LEO)
v_inf_dep_vec = v_dep - vE
v_inf_dep     = np.linalg.norm(v_inf_dep_vec)
v_peri_hyp_E  = math.sqrt(v_inf_dep**2 + 2*muEarth/r_LEO)
dv1           = v_peri_hyp_E - v_circ_LEO

r_LEO2 = rEarth + 500
v_circ_LEO2 = np.sqrt(muEarth / r_LEO2)
v_inf_dep_vec2 = v_dep2 - vE2
v_inf_dep2     = np.linalg.norm(v_inf_dep_vec2)
v_peri_hyp_E2  = math.sqrt(v_inf_dep2**2 + 2*muEarth/r_LEO2)
dv1_2          = v_peri_hyp_E2 - v_circ_LEO

# 5) Mars arrival Δv: into 35 h orbit with rp = rMars + 300 km
r_p_M = rMars + 300.0
r_p_V = rVenus + 2000
v_inf_arr_vec2 = v_arr2 - vV
v_inf_arr2     = np.linalg.norm(v_inf_arr_vec2)
v_peri_hyp_V  = math.sqrt(v_inf_arr2**2 + 2*muVenus/r_p_V)



v_inf_arr_vec = v_arr - vM
v_inf_arr     = np.linalg.norm(v_inf_arr_vec)
rp = rVenus + 2000    # km
ra = rVenus + 10000   # km
a  = 0.5 * (rp + ra)

v_peri_V = math.sqrt(muVenus * (2/rp - 1/a))

# Periapsis velocity of elliptical orbit
e = (ra - rp) / (ra + rp)
v_peri_ell = math.sqrt(muVenus * (1 + e) / rp)
# (equivalently: sqrt(muVenus * (2/rp - 1/a)))


print(v_peri_ell)
v_peri_hyp_M  = math.sqrt(v_inf_arr**2 + 2*muMars/r_p_M)

T_cap = 35.0 * 3600.0
a_cap = (muMars * (T_cap/(2*math.pi))**2)**(1.0/3.0)
e_cap = 1.0 - r_p_M/a_cap

v_peri_ell_M = math.sqrt(muMars*(1 + e_cap)/r_p_M)
dv2 = abs(v_peri_hyp_M - v_peri_ell_M)

dv2_2 = abs(v_peri_hyp_V - v_peri_ell)

dv_total = dv1 + dv2
dv_total2 = dv1_2 + dv2_2


# ==========================
# OUTPUT
# ==========================

print("Earth departure Δv  (LEO -> hyperbola): {:.4f} km/s".format(dv1))
print("Mars arrival Δv     (hyperbola -> 35 h capture orbit): {:.4f} km/s".format(dv2))
print("Total mission Δv: {:.4f} km/s".format(dv_total))

print(dv_total2)
