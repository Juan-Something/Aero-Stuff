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

"""
Heart Check: 2460940.6666666665 is very close to the online value of 2460940.666667

"""

## Question #2

# Calculating Greenwich Sidereal Time (GST) for the given Julian Date
melbourneJd = julian_date(2007, 12, 21, 10, 0, 0)
melbourneLong = 144 + 58/60 # degrees

def greenwichSiderealTime(jd):
    # Calculate number of days (including fraction) since J2000.0
    D = jd - 2451545.0
    # Calculate centuries since J2000.0
    T = D / 36525
    # Calculate GST at 0h UT
    GST_0 = 100.46061837 + 36000.770053608 * T + 0.000387933 * T**2 - (T**3) / 38710000
    # Find UT in hours from the fractional part of JD
    UT = (jd + 0.5) % 1 * 24
    # Complete GST calculation
    GST = GST_0 + 360.98564724 * UT / 24
    GST = GST % 360
    return GST

gst = greenwichSiderealTime(melbourneJd)
lst = (gst + melbourneLong) % 360
print(f"Greenwich Sidereal Time: {gst} degrees")
print(f"Local Sidereal Time in Melbourne: {lst} degrees")

# SLO Sidereal Time
sloJd = julian_date(2025, 7, 4, 12, 30, 22)

westToEast = 360 - 120.653 # degrees
sloGst = greenwichSiderealTime(sloJd)
sloLst = (sloGst + westToEast) % 360
print(f"Local Sidereal Time in San Luis Obispo: {sloLst} degrees")


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
ax.scatter(x[0], y[0], z[0], color='green', label='Start')
ax.scatter(x[-1], y[-1], z[-1], color='red', label='End')
ax.legend()
ax.set_xlabel('x, km')
ax.set_ylabel('y, km')
ax.set_zlabel('z, km')
plt.show()

"""
This velocity is within the range of expected values for an apoapse at this altitude.

"""

# Question 4, Chapter 2, 2.9




# Question 5, Chapter 2, 2.16
marsAltitude = 200 # km
marsMu = 42828 # km^3/s^2
marsRadius = 3396 # km

satelliteRadius = marsAltitude + marsRadius # km
vCircular = np.sqrt(marsMu / satelliteRadius)
print(f"Circular Orbit Velocity at 200 km altitude around Mars: {vCircular:.4f} km/s")

orbitalPeriod = 2 * np.pi * np.sqrt(satelliteRadius**3 / marsMu)
orbitalPeriodHours = orbitalPeriod / 3600
print(f"Orbital Period at 200 km altitude around Mars: {orbitalPeriodHours:.4f} hours")

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
print(f"Orbital period: {periodHours:.5} hours")
print(f"Specific orbital energy: {specificEnergy:.5} km^2/s^2")
print(f"h: {h:.5} km^2/s^2")

trueAnomaly = np.degrees(np.arccos((p/rAltitude - 1)/ecc))
print(f"True Anomaly at 10,000 km altitude: {trueAnomaly:.4} degrees")

vRadial = np.sqrt(muEarth/p) * ecc * np.sin(np.radians(trueAnomaly))
print(f"Radial Velocity at 10,000 km altitude: {vRadial:.5} km/s")
vAzimuth = np.sqrt(muEarth/p) * (1 + ecc * np.cos(np.radians(trueAnomaly)))
print(f"Azimuthal Velocity at 10,000 km altitude: {vAzimuth:.5} km/s")
vPeriapsis = np.sqrt(muEarth * (1 + ecc) / (perigeeRadius))
print(f"Velocity at Periapsis: {vPeriapsis:.5} km/s")
vApoapsis = np.sqrt(muEarth * (1 - ecc) / (apogeeRadius))
print(f"Velocity at Apoapsis: {vApoapsis:.5} km/s")

"""
Heart Check: All values match the ones we calculated in class.
"""








