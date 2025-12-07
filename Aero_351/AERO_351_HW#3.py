import matplotlib as mpL
import scipy as sciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



# 4.15
print("-----Q1----")

ecc1 = 1.5                                         
perigeeAlt = 300 + 6378                            # [km]
inc1 = np.deg2rad(35)                              # [rad]
RAAN1 = np.deg2rad(130)                            # [rad]
w1 = np.deg2rad(115)                               # [rad]
muEarth = 398600                                   # [km^3/s^2]

R3W = np.array([[np.cos(w1), np.sin(w1), 0],       
                [-np.sin(w1), np.cos(w1), 0],
                [0, 0, 1]])

R3RAAN = np.array([[np.cos(RAAN1), np.sin(RAAN1), 0],   
                   [-np.sin(RAAN1), np.cos(RAAN1), 0],
                   [0, 0, 1]])

R1Inc = np.array([[1,0,0],                           
                  [0, np.cos(inc1), np.sin(inc1)],
                  [0, -np.sin(inc1), np.cos(inc1)]])

h1 = np.sqrt(perigeeAlt*muEarth*(1+ecc1))           # [km^2/s]
perigeeVelo = h1 / perigeeAlt                       # [km/s]

rotMatrix =  R3W @ R1Inc @ R3RAAN                   
perigeeR = np.array([perigeeAlt, 0, 0])             # [km]
perigeeV = np.array([0, perigeeVelo, 0])            # [km/s]

equatorialR = np.transpose(rotMatrix) @ perigeeR    # [km]
equatorialV = np.transpose(rotMatrix) @ perigeeV    # [km/s]

print(equatorialR)
print(equatorialV)

# 5.6
print("-----Q2----")

geoR1 = np.array([5644, 2830, 4170])               # [km]
geoR2 = np.array([-2240, 7320, 4980])              # [km]
dt = 20 * 60                                       # [s]

# Pulled from HW#2
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
        # (sinh - s)/s^3
    return (np.sinh(s) - s) / (s**3)

def timeOfFlight(z, r1n, r2n, A, mu= muEarth):     # returns t [s]
    C = stumpff_C(z)
    S = stumpff_S(z)
    y = r1n + r2n + A * (z*S - 1) / C              # [km]
    if y < 0:
        return math.inf, y, C, S
    chi = math.sqrt(y / C)                          
    t = (chi**3 * S + A * chi) / math.sqrt(mu)      # [s]
    return t, y, C, S

def lambertBisection(r1, r2, dt, mu=muEarth, prograde=True, long_way=False, tol=1e-9, maxiter=200):
    r1 = np.asarray(r1, float)                      # [km]
    r2 = np.asarray(r2, float)                      # [km]
    r1Norm = np.linalg.norm(r1)                     # [km]
    r2Norm = np.linalg.norm(r2)                     # [km]

    transferAngle = np.dot(r1, r2) / (r1Norm * r2Norm)   
    transferAngleDegree = math.acos(transferAngle)       # [rad]
    crz = np.cross(r1, r2)[2]                      # [km^2]
    if prograde:
        if crz < 0: transferAngleDegree = 2*np.pi - transferAngleDegree
    else:
        if crz >= 0: transferAngleDegree = 2*np.pi - transferAngleDegree
    if long_way:
        transferAngleDegree = 2*np.pi - transferAngleDegree

    if abs(1 - transferAngleDegree) < 1e-12:
        raise ValueError("A is undefined for collinear geometry")
    A = np.sin(transferAngleDegree) * np.sqrt(r1Norm * r2Norm / (1 - transferAngle))  # [km]
    if abs(A) < 1e-14:
        raise ValueError("A = 0; no orbit possible with this method.")

    z_low  = -4*np.pi**2 + 1e-6                     
    z_high =  4*np.pi**2 - 1e-6                     

    it = 0
    z_mid = 0.0
    while it < maxiter:
        z_mid = 0.5*(z_low + z_high)
        t_mid, y_mid, C_mid, S_mid = timeOfFlight(z_mid, r1Norm, r2Norm, A, mu)
        if not math.isfinite(t_mid):
            z_low = z_mid; it += 1; continue
        if abs(t_mid - dt) < tol:
            break
        if t_mid < dt:
            z_low = z_mid
        else:
            z_high = z_mid
        it += 1

    y = y_mid                                       # [km]
    f    = 1 - y / r1Norm                           
    g    = A * np.sqrt(y / mu)                      # [s]
    gdot = 1 - y / r2Norm                           
    fdot = (f * gdot - 1.0) / g                     # [1/s]

    v1 = (r2 - f * r1) / g                          # [km/s]
    v2 = (gdot * r2 - r1) / g                       # [km/s]
    return v1, v2

v1, v2= lambertBisection(geoR1, geoR2, dt, prograde= True, long_way= False)
v1Mag = np.linalg.norm(v1)                          # [km/s]
v2Mag = np.linalg.norm(v2)                          # [km/s]
print("v1 [km/s] =", np.round(v1, 6))
print("v2 [km/s] =", np.round(v2, 6))
print(v1Mag)
print(v2Mag)

# 6.8
print("-----Q3----")

rEarth = 6378                                       # [km]
altitudeOrbit = 300                                 # [km]
rCoplanar = 3000 + rEarth                           # [km]
rPOrbit = altitudeOrbit + rEarth                    # [km]
aTransfer = (rCoplanar + rPOrbit) / 2               # [km]

vOrbit = np.sqrt(muEarth / rPOrbit)                 # [km/s]
vCoplanar = np.sqrt(muEarth / rCoplanar)            # [km/s]
print(f"Orbit Speed: {vOrbit: .4f} km/s")
print(f"Coplanar Orbit Speed: {vCoplanar: .4f} km/s")

hInitial = vOrbit * rPOrbit                         # [km^2/s]

eccTrans = (rCoplanar - rPOrbit) / (rCoplanar + rPOrbit)  
h3 = np.sqrt((rPOrbit * muEarth * (1+eccTrans)))    # [km^2/s]
print(f"h value: {h3: .4f} km^2/s")

vTransferPerigee = h3 / rPOrbit                     # [km/s]
vTransferApogee = h3 / rCoplanar                    # [km/s]

initialDeltaV = (vTransferPerigee - vOrbit)         # [km/s]
print(f"Initial Transfer Delta V {initialDeltaV: .4f} km/s")
finalDeltaV = (vCoplanar - vTransferApogee)         # [km/s]
deltaV = (finalDeltaV + initialDeltaV)              # [km/s]
print(f"Transfer Delta V {deltaV: .4f} km/s")

periodTransfer = ((2 * np.pi) / (np.sqrt(muEarth))) * (aTransfer**1.5)  # [s]
timeToTransfer = (periodTransfer / 2) / 60          # [min]
print(f"Time to Transfer {timeToTransfer: .4f} min")

# 6.23
print("-----Q4-----")
orbit1Rp = 8100                                     # [km]
orbit1Ra = 18900                                    # [km]

def orbitalElements(perigeeRadius, apogeeRadius, muEarth, rEarth):
    ecc = (apogeeRadius - perigeeRadius) / (apogeeRadius + perigeeRadius)    
    a = (apogeeRadius + perigeeRadius) / 2                                   # [km]
    period = ((2*np.pi) / np.sqrt(muEarth)) * a ** (3/2)                     # [s]
    specificEnergy = -(1/2)*(muEarth)/a                                      # [km^2/s^2]
    h = np.sqrt((a)*muEarth*(1 - ecc**2))                                    # [km^2/s]
    p = h**2 / muEarth                                                       # [km]
    return ecc, a, period, specificEnergy, h, p

orbit1Ecc, orbit1a, orbit1Period, orbit1SpecificEnergy, orbit1H, orbit1P = orbitalElements(orbit1Rp, orbit1Ra, muEarth= muEarth, rEarth= 6378)

def EFromTA (nu, e):                             
    return 2.0 * np.atan2(np.sqrt(1 - e) * np.sin(nu / 2.0), np.sqrt(1 + e) * np.cos(nu / 2.0))

def MFromE (E, e):                                 
    return E - e*np.sin(E)

nuB = np.deg2rad(45)                               # [rad]
nuC = np.deg2rad(150)                              # [rad]
E0 = EFromTA(nuC, orbit1Ecc)                       # [rad]
EF = EFromTA(nuB, orbit1Ecc)                       # [rad]
M0 = MFromE(E0, orbit1Ecc)                         # [rad]
MF = MFromE(EF, orbit1Ecc)                         # [rad]

P0 = ((2 * np.pi) / np.sqrt(muEarth)) * orbit1a**1.5  # [s]
P1 = orbit1Rp*(1 + orbit1Ecc)                      # [km]

rB = P1 / (1 + orbit1Ecc * np.cos(nuB))            # [km]
timeTo = orbit1Period * ((MF - M0) %(2*np.pi)) / (2*np.pi)  # [s]
orbit2a = (muEarth * (timeTo/(2*np.pi))**2)**(1/3)          # [km]
orbit2Ecc = 1 - rB / orbit2a                       
V1 = np.sqrt(muEarth*(2/rB - 1/orbit1a))           # [km/s]
V2 = np.sqrt(muEarth*(2/rB - 1/orbit2a))           # [km/s]

gamma = np.arctan2(orbit1Ecc*np.sin(nuB), 1 + orbit1Ecc*np.cos(nuB))   # [rad]
deltaOneWay =  np.sqrt(V1**2 + V2**2 - 2*V1*V2*np.cos(gamma))          # [km/s]
deltaTotal = 2 * deltaOneWay                                           # [km/s]
print(f"Total Delta V: {deltaTotal: .4f} km/s")

# 6.25
print("-----Q5-----")
perigeeAlt2 = 1270                                   # [km]
perigeeV2 = 9                                        # [km/s]
perigeeR2 = 1270 + 6378                              # [km]
TA2 = np.deg2rad(100)                                # [rad]

e2 = 0.4                                             

e1 = (perigeeV2**2 *perigeeR2) / muEarth - 1   # Moved the vis-viva equation around to pull ecc without solving for h      
a1 = perigeeR2 / (1 - e1)                            # [km]
p1 = perigeeR2*(1 + e1)                              # [km]

rTA = p1 / (1 + e1*np.cos(TA2))                      # [km]

p2 = rTA * (1 + e2*np.cos(TA2))                      # [km]
a2 = p2 / (1 - e2**2)                                # [km]

maneuverVelo = np.sqrt(muEarth*(2/rTA - 1/a1))       # [km/s]
maneuverVelo2 = np.sqrt(muEarth*(2/rTA - 1/a2))      # [km/s]

flightPathAngle1 = np.arctan2(e1*np.sin(TA2), (1 + e1*np.cos(TA2)))   # [rad]
flightPathAngle2 = np.arctan2(e2*np.sin(TA2), (1 + e2*np.cos(TA2)))   # [rad]

deltaFlightPath = flightPathAngle2 - flightPathAngle1                  # [rad]
deltaV2 = np.sqrt(maneuverVelo**2 + maneuverVelo2**2 - 2*maneuverVelo2*maneuverVelo*np.cos(deltaFlightPath))  # [km/s]
print(f"Delta V Magnitude: {deltaV2: .4f} km/s")
print(f"Delta Flght Path Angle: {np.rad2deg(deltaFlightPath): .4f} degrees")

# 6.31, see handwritten

# 6.44
print("-----Q6-----")
r1 = rEarth + 300.0                               # [km]
r2 = rEarth + 600.0                               # [km]
aTransfer = 0.5*(r1 + r2)                         # [km]

spaceCraftVelo        = np.sqrt(muEarth / r1)                     # [km/s]
orbit2Velo            = np.sqrt(muEarth / r2)                     # [km/s]
spaceCraftPerigeeVelo = np.sqrt(muEarth*(2.0/r1 - 1.0/aTransfer)) # [km/s]
spaceCraftApogeeVelo  = np.sqrt(muEarth*(2.0/r2 - 1.0/aTransfer)) # [km/s]

dInc = np.deg2rad(20.0)                                          # [rad]

# (a) plane change after circularizing at 600 km (three burns)
spaceCraftDeltaV_a = (abs(spaceCraftPerigeeVelo - spaceCraftVelo)
                      + abs(orbit2Velo - spaceCraftApogeeVelo)
                      + 2.0*orbit2Velo*np.sin(dInc/2.0))          # [km/s]

# (b) combine plane change with circularization at apogee (two burns)
spaceCraftDeltaV_b = (abs(spaceCraftPerigeeVelo - spaceCraftVelo)
                      + np.sqrt(spaceCraftApogeeVelo**2 + orbit2Velo**2
                               - 2.0*spaceCraftApogeeVelo*orbit2Velo*np.cos(dInc)))  # [km/s]

# (c) combine plane change with departure from 300-km orbit (two burns)
spaceCraftDeltaV_c = (np.sqrt(spaceCraftVelo**2 + spaceCraftPerigeeVelo**2
                              - 2.0*spaceCraftVelo*spaceCraftPerigeeVelo*np.cos(dInc))
                      + abs(orbit2Velo - spaceCraftApogeeVelo))   # [km/s]

print(spaceCraftDeltaV_a, spaceCraftDeltaV_b, spaceCraftDeltaV_c)

# 6.47
print("-----Q7-----")
# Inputs
initialMass = 1000.0                                # [kg]
thrustForce = 10000.0                               # [N]
gravity     = 9.80665                               # [m/s^2]
initialR    = np.array([436.0, 6083.0, 2529.0])     # [km]
initialV    = np.array([-7.340, -0.5125, 2.497])    # [km/s]
deltaT      = 89 * 60.0                             # [s]
ISP         = 300.0                                 # [s]
burnTime    = 120.0                                 # [s]
thrustTime  = deltaT + burnTime                     # [s]

# Guidance: thrust along velocity
def thrustDirAlongV(r, v, t):
    vNorm = np.linalg.norm(v)                       # [km/s]
    return v / vNorm if vNorm > 0.0 else np.zeros(3)

def gravAcc(r):
    rNorm = np.linalg.norm(r)                       # [km]
    return -muEarth * r / rNorm**3                  # [km/s^2]

# Thrust (active only in [deltaT, thrustTime)) + mass flow
def thrustAccAndMdot(r, v, m, t):
    if not (deltaT <= t < thrustTime) or thrustForce <= 0.0 or m <= 0.0:
        return np.zeros(3), 0.0
    u    = thrustDirAlongV(r, v, t)                 
    aT   = (thrustForce / m) / 1000.0 * u           # [km/s^2]  
    mDot = -thrustForce / (ISP * gravity)           # [kg/s]
    return aT, mDot

# RK4 propagation w/ history (for plotting)
def propagateWithHistory(r0, v0, m0, tMax, dt, startCheckAfter=None):
    t = 0.0                                         # [s]
    state = np.hstack((r0.astype(float), v0.astype(float), float(m0)))  # [km, km/s, kg]

    times, alts, masses = [], [], []                # [s], [km], [kg]
    rHist = []                                       # [km]
    prevVr = None
    steps = int(np.ceil(tMax / dt))                 

    def deriv(st, tt):
        r, v, m = st[:3], st[3:6], st[6]
        aT, mDot = thrustAccAndMdot(r, v, m, tt)
        a = gravAcc(r) + aT                         # [km/s^2]
        return np.hstack((v, a, mDot))

    for _ in range(steps):
        r, v, m = state[:3], state[3:6], state[6]
        times.append(t)                              # [s]
        alts.append(np.linalg.norm(r) - rEarth)      # [km]
        masses.append(m)                             # [kg]
        rHist.append(r.copy())                       # [km]

        k1 = deriv(state, t)
        k2 = deriv(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = deriv(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = deriv(state + dt * k3, t + dt)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

        if startCheckAfter is not None and t > startCheckAfter:
            rNow, vNow = state[:3], state[3:6]
            vRadial = np.dot(rNow, vNow) / np.linalg.norm(rNow)   # [km/s]
            if prevVr is not None and prevVr > 0 and vRadial <= 0:
                break
            prevVr = vRadial

    return (np.array(times), np.array(alts), np.array(masses),
            np.vstack(rHist), state[6], t)

tMax = 6 * 3600.0                                   # [s]
dt = 1.0                                            # [s]
startCheckAfter = thrustTime + 600.0                # [s]

times, alts, masses, rHist, finalMass, elapsedSec = propagateWithHistory(
    initialR, initialV, initialMass, tMax, dt, startCheckAfter
)

plt.figure()
theta = np.linspace(0.0, 2*np.pi, 400)
plt.plot(rEarth*np.cos(theta), rEarth*np.sin(theta))  # Earth [km]
plt.plot(rHist[:,0], rHist[:,1])                      # trajectory [km]
startIdx = 0
ignIdx   = min(int(deltaT // dt), len(rHist)-1)
cutIdx   = min(int(thrustTime // dt), len(rHist)-1)
endIdx   = len(rHist)-1
plt.scatter(rHist[startIdx,0], rHist[startIdx,1], marker='o', label="start")
plt.scatter(rHist[ignIdx,0],   rHist[ignIdx,1],   marker='^', label="ignition")
plt.scatter(rHist[cutIdx,0],   rHist[cutIdx,1],   marker='s', label="cutoff")
plt.axis('equal')
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("Changing Orbit (2D Inertial x-y Projection)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"peakAltitudeKm: {float(alts.max()) : .4f}")
print(f"timeAtPeakMin: {float(times[np.argmax(alts)]/60) : .4f}")
print(f"finalMassKg: {float(finalMass) : .4f}")
