import matplotlib as mpL
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize as op 
from scipy.integrate import solve_ivp 
from Aero_351.COES import ClassicOrbitalElementsCalculator as COES


# 2.23
print("----- Question 1 -----")

rPerigeeAltitude = 500 # km
rEarth = 6378 # km
muEarth = 398600
rPerigee = rPerigeeAltitude + rEarth
vSpeed = 10 # km/s
TA = 120 # degrees
radTA = np.deg2rad(TA)

h = rPerigee * vSpeed # km^2 / s

ecc = (h**2 / (rPerigee * muEarth)) - 1

flightPathAngle = np.arctan((ecc*np.sin(radTA))/(1+ecc*np.cos(radTA)))
flightPathAngleDeg = np.rad2deg(flightPathAngle)
print(f"Flight path angle: {flightPathAngleDeg} degrees")

r = (h**2 / muEarth) * (1 / (1 + ecc*np.cos(radTA)))

altitude = r - rEarth

print(f"Altitude of satellite: {altitude: .4f} km")


# 2.36
print("----- Question 2 -----")
perigeeSpeed = 11
rPerigreeAltitude2 = 250
rPerigee2 = rEarth + rPerigreeAltitude2
TA2 = np.deg2rad(100)

specificEnergy = ((perigeeSpeed)**2 / 2) - (muEarth / rPerigee2)
excessSpeed = np.sqrt(2*specificEnergy)

print(f"Hyperbolic excess speed: {excessSpeed: .4f} m/s")

h2 = rPerigee2 * perigeeSpeed

ecc2 = (h2**2 / (rPerigee2 * muEarth)) - 1
r2 = (h2**2 / muEarth) * (1 / (1 + ecc2*np.cos(TA2)))

vAzimuthal = h2 / r2
vRadial = (muEarth/h2)*ecc2*np.sin(TA2)
print(f"Radius when True Anomoly is 100 degrees: {r2: .4f} km")
print(f"Azimuthal speed when True Anomaly is 100 degrees: {vAzimuthal: .4f} km/s")
print(f"Radial speed when True Anomaly is 100 degrees {vRadial: .4f} km/s")


# 2.37
print("----- Question 3 -----")

rAltitudeMeteor = 402000 # km
vMeteor = 2.23
TAMeteor = np.deg2rad(150)
 
rMeteor = rAltitudeMeteor - rEarth

specificEnergy2 = ((vMeteor**2) / 2) - (muEarth/rAltitudeMeteor)
aMeteor = muEarth/(2*specificEnergy2)

C = rAltitudeMeteor + aMeteor
B = rAltitudeMeteor*np.cos(TAMeteor)
A = -aMeteor

quadInside = B**2 - 4 * A * C

# Quadratic equation shenanigans
ecc3 = (-B - np.sqrt(quadInside)) / (2 * A)
print(f"Eccentricity of hyperbolic trajectory: {ecc3: .4f}")

rMeteorPerigee = aMeteor*(ecc3 - 1) 
rMeteorPerigeeAltitude = rMeteorPerigee - rEarth
print(f"Altitude at perigee: {rMeteorPerigeeAltitude: .4f} km")

h3 = np.sqrt((rMeteorPerigee * muEarth * (1+ecc3)))
vMeteorPerigee = h3 / rMeteorPerigee

print(f"Velocity at closest approach: {vMeteorPerigee: .4f} km/s")



# 3.8
print("----- Question 4 -----")
rSatellitePerigee = 200 + rEarth
rSatelliteApogee = 600 + rEarth
rSatelliteApogeeMin = 400 + rEarth

def orbitalElements(perigeeRadius, apogeeRadius, muEarth, rEarth):
    ecc = (apogeeRadius - perigeeRadius) / (apogeeRadius + perigeeRadius)
    a = (apogeeRadius + perigeeRadius) / 2
    period = ((2*np.pi) / np.sqrt(muEarth)) * a ** (3/2)
    specificEnergy = -(1/2)*(muEarth)/a
    h = np.sqrt((a)*muEarth*(1 - ecc**2))
    p = h**2 / muEarth

    return ecc, a, period, specificEnergy, h, p

eccSatellite, aSatellite, periodSatellite, seSatellite, hSatellite, pSatellite = orbitalElements(rSatellitePerigee, rSatelliteApogee, muEarth, rEarth)

TASatellite = np.arccos((hSatellite**2/ (rSatelliteApogeeMin * muEarth) - 1) / eccSatellite)

eccentricAnomaly = 2* np.arctan(np.sqrt((1-eccSatellite) / (1 + eccSatellite)) * np.tan(TASatellite/2))

meanAnomaly = eccentricAnomaly - eccSatellite*np.sin(eccentricAnomaly)

timeAbove = (meanAnomaly / (2 * np.pi)) *periodSatellite

timeInterval = (periodSatellite - 2*timeAbove) * (1/60)

print(f"Time spent above or at 400 km {timeInterval: .2f} minutes")

# 3.10
print("----- Question 5 -----")
periodSatellite2 = 14 * 60 * 60
rSatellite2Perigee = 10000
tSatellite = 10 * 60 * 60

aSatellite2 = (muEarth * periodSatellite2**2 / (4*(np.pi**2)))**(1/3)
eccSatellite2 = (2*aSatellite2 - 2*rSatellite2Perigee) / (2 * aSatellite2)

meanEccentricity = (2 *np.pi / periodSatellite2) * tSatellite

M = np.mod(meanEccentricity, 2*np.pi)          # normalize M
f  = lambda E: E - eccSatellite2*np.sin(E) - M
fp = lambda E: 1 - eccSatellite2*np.cos(E)

E0 = M if eccSatellite2 < 0.8 else np.pi*np.sign(M)
E  = op.newton(f, x0=E0, fprime=fp, tol=1e-12, maxiter=100)

# report in [0, 2π)
if E < 0:
    E += 2*np.pi

TASatellite2 = 2*np.arctan2(np.sqrt(1+eccSatellite2)*np.sin(E/2),np.sqrt(1-eccSatellite2)*np.cos(E/2))
radiusSat = aSatellite2*(1- eccSatellite2* np.cos(E))
print(f"Radius of satellite {radiusSat: .8} km")

specificEnergy3 = -muEarth / (2 * aSatellite2)
speedSat = np.sqrt(2+(specificEnergy3 + (muEarth / radiusSat)))
print(f"Speed {speedSat: .4} km/s")
hSat2 = np.sqrt(rSatellite2Perigee * muEarth * (1+eccSatellite2))
vRadialSat = muEarth/(hSat2) * eccSatellite2 *np.sin(TASatellite2)
print(f"Radial component of speed {vRadialSat: .4} km/s")


# 3.20 (check with ODE45)
print("----- Question 6 -----")
rVector = np.array([20000, -105000, -19000])
vVector = np.array([.9000, -3.4000, -1.5000])

rMag = np.sqrt(rVector * rVector)
vMag = np.sqrt(vVector * vVector)

timeLapsed  = 2 * 60 * 60

def stumpff_C(z):
    if abs(z) < 1e-8:
        return 0.5 - z/24 + z*z/720
    if z > 0:
        s = np.sqrt(z)
        return (1 - np.cos(s)) / z
    s = np.sqrt(-z)
    return (np.cosh(s) - 1) / (-z)

def stumpff_S(z):
    if abs(z) < 1e-8:
        return 1/6 - z/120 + z*z/5040
    if z > 0:
        s = np.sqrt(z)
        return (s - np.sin(s)) / (s**3)
    s = np.sqrt(-z)
    return (np.sinh(s) - s) / (s**3)

def solve_chi(r0_vec, v0_vec, dt, mu, tol=1e-12, maxit=100):
    r0 = np.linalg.norm(r0_vec)
    v0 = np.linalg.norm(v0_vec)
    vr0 = np.dot(r0_vec, v0_vec) / r0
    alpha = 2.0/r0 - (v0*v0)/mu

    # initial guess
    chi = (np.sqrt(mu)*abs(alpha)*dt) if abs(alpha) > 1e-12 else (np.sqrt(mu)*dt/r0)

    for _ in range(maxit):
        z = alpha * chi*chi
        C = stumpff_C(z)
        S = stumpff_S(z)

        F  = (r0*vr0/np.sqrt(mu))*chi*chi*C + (1 - alpha*r0)*chi**3*S + r0*chi - np.sqrt(mu)*dt
        dF = (r0*vr0/np.sqrt(mu))*chi*(1 - z*S) + (1 - alpha*r0)*chi*chi*C + r0

        delta = -F/dF
        chi  += delta
        f = 1 - (chi*chi/r0)*C
        g = dt - (chi**3/np.sqrt(mu))*S
        r_vec = f*r0_vec + g*v0_vec
        r = np.linalg.norm(r_vec)
        fdot = (np.sqrt(mu)/(r*r0))*(alpha*chi**3*S - chi)
        gdot = 1 - (chi*chi/r)*C
        v_vec = fdot*r0_vec + gdot*v0_vec

        if abs(delta) < tol:
            return chi, r_vec, v_vec
        

        
    


chi, newR, newV = solve_chi(rVector, vVector, timeLapsed, muEarth, tol= 1e-12, maxit= 500)

print(f"New position: {newR} km")
print(f"New velocity: {newV} km/s")

# --- ode45-style propagation (RK45) and comparison ---
def two_body_ode(t, y, mu):
    r = y[:3]
    v = y[3:]
    rnorm = np.linalg.norm(r)
    a = -mu * r / (rnorm**3)
    return np.hstack((v, a))

y0 = np.hstack((rVector, vVector))
sol = solve_ivp(
    fun=lambda t, y: two_body_ode(t, y, muEarth),
    t_span=(0.0, timeLapsed),
    y0=y0,
    method="RK45",           
    rtol=1e-10,
    atol=1e-13,
    dense_output=False,
)

r_ode = sol.y[:3, -1]
v_ode = sol.y[3:, -1]

# Compare to universal-variable solution computed above: newR, newV
dr = r_ode - newR
dv = v_ode - newV

print("----- ode45 vs universal-variable (2 h) -----")
print(f"ode45 position  : {r_ode} km")
print(f"ode45 velocity  : {v_ode} km/s")
print(f"Δr (ode45 - UV) : {dr} km  | ||Δr|| = {np.linalg.norm(dr):.6e} km")
print(f"Δv (ode45 - UV) : {dv} km/s| ||Δv|| = {np.linalg.norm(dv):.6e} km/s")

# 4.5
print("----- Question 7 -----")
rVector2 =  [6500, -7500, -2500]
vVector2 = [4, 3, -3]

variables = COES(rVector2, vVector2)


# 4.7
print("----- Question 8 -----")

rVector3 = np.array([-6600, -1300, -5200])
eVec = np.array([-0.4, -0.5, -0.6])

normR = np.linalg.norm(rVector3)
normE = np.linalg.norm(eVec)

theta2 = 2*np.pi - np.arccos(np.dot(rVector3, eVec)/(normR * normE))
thetaDeg = np.rad2deg(theta2)
h4 = np.sqrt((normR * muEarth) * (1 + (normE)*np.cos(theta2)))
magH4 = np.linalg.norm(h4)

h5 = np.cross(rVector3, eVec)

normh5 = np.linalg.norm(h5)
angularMomentum = h5 / normh5


newH = angularMomentum * magH4
theta3 = np.arccos(newH / magH4)
print(f"Inclination of Orbit is: {np.rad2deg(theta3[2]): .4f} degrees")



def twoBodyODE(t, state, muEarth): 
    x, y, z, dx, dy, dz =state
    rMag = np.linalg.norm([x, y, z])

    ddx = -muEarth * x / rMag**3
    ddy = -muEarth * y/ rMag**3
    ddz = -muEarth * z / rMag**3

    return [dx, dy, dz, ddx, ddy, ddz]

def ClassicOrbitalElementsCalculator(R_Vector, V_Vector):
    """
    Calculates classical orbital elements from position and velocity vectors.

    Inputs:
        R_Vector : array-like, shape (3,)
            Position vector [km]
        V_Vector : array-like, shape (3,)
            Velocity vector [km/s]
    Outputs:
        a    : Semi-major axis [km]
        e    : Eccentricity (scalar)
        nu   : True anomaly [deg]
        i    : Inclination [deg]
        RAAN : Right Ascension of Ascending Node [deg]
        w    : Argument of Periapsis [deg]
    """

    R_Vector = np.array(R_Vector, dtype=float)
    V_Vector = np.array(V_Vector, dtype=float)

    R = np.linalg.norm(R_Vector)
    V = np.linalg.norm(V_Vector)
    mu = 398600  # km^3/s^2

    # Angular momentum vector
    h_Vector = np.cross(R_Vector, V_Vector)

    # Unit vectors
    ihat = np.array([1, 0, 0])
    jhat = np.array([0, 1, 0])
    khat = np.array([0, 0, 1])

    #COE 0 - h
    hMag = np.linalg.norm(h_Vector)

    # COE 1 - Semi-major axis
    SME = (V**2 / 2) - (mu / R)  # Specific Mechanical Energy
    a = -mu / (2 * SME)

    # COE 2 - Eccentricity
    e_Vector = (1 / mu) * (((V**2 - (mu / R)) * R_Vector) - (np.dot(R_Vector, V_Vector) * V_Vector))
    e = np.linalg.norm(e_Vector)

    # COE 3 - Inclination
    i = np.degrees(np.arccos(np.dot(khat, h_Vector) / (np.linalg.norm(khat) * np.linalg.norm(h_Vector))))

    # COE 4 - RAAN
    n_vector = np.cross(khat, h_Vector)
    n_norm = np.linalg.norm(n_vector)
    RAAN = np.degrees(np.arctan2(n_vector[1], n_vector[0]))
    if RAAN < 0:
        RAAN += 360.0

    # COE 5 - Argument of Periapsis
    w1 = np.arccos(np.dot(n_vector, e_Vector) / (n_norm * e))
    if e_Vector[2] < 0:
        w = np.degrees(2 * np.pi - w1)
    else:
        w = np.degrees(w1)

    # COE 6 - True Anomaly
    nu1 = np.arccos(np.dot(e_Vector, R_Vector) / (e * R))
    if np.dot(R_Vector, V_Vector) < 0:
        nu = np.degrees(2 * np.pi - nu1)
    else:
        nu = np.degrees(nu1)

        print("\n--- Classical Orbital Elements ---")
    print(f"Specific Angular Momentum (h): {hMag:12.4f} km²/s")
    print(f"Semi-Major Axis (a):           {a:12.4f} km")
    print(f"Eccentricity (e):              {e:12.6f}")
    print(f"True Anomaly (ν):              {nu:12.4f}°")
    print(f"Inclination (i):               {i:12.4f}°")
    print(f"RAAN (Ω):                      {RAAN:12.4f}°")
    print(f"Argument of Periapsis (ω):     {w:12.4f}°")
    print("----------------------------------\n")
    return float(hMag), float(a), float(e), float(nu), float(i), float(RAAN), float(w)

COES = ClassicOrbitalElementsCalculator(rVector, vVector)



