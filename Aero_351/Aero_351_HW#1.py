import matplotlib as mpL
import scipy as sciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## Question #1
# Calculating Julian Time for September 22, 2025 4:00:00 UT

def julian_date(year, month, day, hour, minute, second):
    if month <= 2:
        year -= 1
        month += 12

    
    j0 = (367*year - np.floor((7*(year + np.floor((month + 9)/12))) / 4) + np.floor((275*month)/9) + day + 1721013.5)

    jD = j0 + (hour + minute/60 + second/3600)/24
    return jD 

jd = julian_date(2025, 9, 22, 4, 0, 0)
print(f"Julian Date: {jd}")

## Question #2

# Calculating Greenwich Sidereal Time (GST) for the given Julian Date
melbourneJd = julian_date(2007, 12, 21, 10, 0, 0)
melbourneLong = 144 + 58/60 # degrees

def greenwichSiderealTime(jd):
    T = (jd - 2451545.0) / 36525
    GST = 100.4606184 + 26000.77004*T + 0.000387933*T**2 - 2.58e-8*T**3

    GST = GST % 360
    return GST

gst = greenwichSiderealTime(melbourneJd)
lst = (gst + melbourneLong) 
print(f"Greenwich Sidereal Time: {gst} degrees")
print(f"Local Sidereal Time in Melbourne: {lst} degrees")

## Question #3
rVect = np.array([3207, 5459,  2714])
vVect = np.array([-6.532, 0.7835, 6.142])
state1 = np.concatenate([rVect, vVect])
rTol = 1e-8
aTol = 1e-8

muEarth = 398600
tSpan = [0, 5*60*60]

def twoBodyODE(t, state, muEarth): 
    x, y, z, dx, dy, dz =state
    rMag = np.linalg.norm([x, y, z])

    ddx = -muEarth * x / rMag**3
    ddy = -muEarth * y/ rMag**3
    ddz = -muEarth * z / rMag**3

    return [dx, dy, dz, ddx, ddy, ddz]

solution = sciPy.integrate.solve_ivp(twoBodyODE, tSpan, state1, args=(muEarth,), rtol=rTol, atol=aTol)

x, y, z = solution.y[0], solution.y[1], solution.y[2]
# Find position and speed magnitude at five hours
final_pos = solution.y[:3, -1]
final_vel = solution.y[3:, -1]
pos_mag = np.linalg.norm(final_pos)
vel_mag = np.linalg.norm(final_vel)
print(f"Position magnitude at 5 hours: {pos_mag:.3f} km")
print(f"Speed magnitude at 5 hours: {vel_mag:.3f} km/s")

# Plot
fig =   plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x, y, z)
ax.set_xlabel('x, km')
ax.set_ylabel('y, km')
ax.set_zlabel('z, km')
plt.show()

# Question 4, Chapter 2, 2.9

# Question 5, Chapter 2, 2.16

# Question 6, Chapter 2, 2.20

perigeeRadius = 10000 # km
apogeeRadius = 100000 # km
muEarth = 398600 # km^3/s^2
rEarth = 6378 # km
rAltitude = 10000 + rEarth # km

def orbitalElements(perigeeRadius, apogeeRadius, muEarth, rEarth):
    ecc = (apogeeRadius - perigeeRadius) / (apogeeRadius + perigeeRadius)
    a = (apogeeRadius + perigeeRadius) / 2
    period = ((2*np.pi) / np.sqrt(muEarth)) * a ** (3/2)
    specificEnergy = -(1/2)*(muEarth)/a
    h = np.sqrt((a)*muEarth*(1 - ecc**2))
    p = h**2 / muEarth

    return ecc, a, period, specificEnergy, h, p

ecc, a, period, specificEnergy, h, p = orbitalElements(perigeeRadius, apogeeRadius, muEarth, rEarth)
periodHours = period / 3600
print(f"Eccentricity: {ecc:.4f}")
print(f"Semi-major axis: {a:.2f} km")
print(f"Orbital period: {periodHours:.2f} hours")
print(f"Specific orbital energy: {specificEnergy:.2f} km^2/s^2")
print(f"h: {h:.2f} km^2/s^2")

trueAnomaly = np.degrees(np.arccos((p/rAltitude - 1)/ecc))
print(f"True Anomaly at 10,000 km altitude: {trueAnomaly:.2f} degrees")









