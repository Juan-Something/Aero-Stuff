"""
TRISTIN KOURY
AERO 351: INTRODUCTION TO ORBITAL MECHANICS
SATELLITE GROUP PROJECT | DEBRIS COLLECTION MISSION
"""

# standard imports:
import numpy as np
import fruise.time_tools as tu
import fruise.orbit_tools as f
import fruise.constants as c
from utils import homework_helpers as hh
import utils.linearAlgebra_utils as la


"""
DIRECTIONS: 

STEP 1:
NEED to find the position of each satellite from when their TLE was recorded:
USE TLE data to find true anomaly (TA): M, ecc:
    M => E => TA
USE ORBIT EQUATION to find r and v vectors at t_TLE:
    TA, ecc, p => r & v
USE RAAN, inc, and AoP to transform the r vector from perifocal reference frame
into ECI reference frame (that way we can more easily compute transfer shit)
    r_ECI, v_ECI = (Cz(RAAN) x Cx(inc) x Cz(AoP)) x (r, v)

STEP 2:
NEED to use TLE epoch to figure out "when" each satellite is, and then propagate
their positions and velocities to the time of the most recently recorded TLE:
USE ECI position and velocity...
"""

# custom function definitions:
def z2rabtearth(z): 
    return c.r_earth + z

def TLEdata2RV0 (M, ecc, RAAN, inc, AoP, ra, rp, mu=c.mu_earth):
    """
    INPUTS:
    M = Mean anomaly (degrees)
    ecc = Eccentricity
    RAAN = Right ascension of ascending node (degrees)
    inc = Inclination (degrees)
    AoP = Argumento of perigee (degrees)
    ra = Radius at apogess (km)
    rp = Radius at perigee (km)
    mu = Gravitational parameter of body in focus (idrk lol)

    OUTPUTS:
    r_ECI = position in ECI [km]
    v_ECI = velocity in ECI [km]
    """

    # get the true anomaly:
    E = f.M2E(M, ecc)
    TA = np.deg2rad(f.E2TA(E, ecc))# in radians for computation

    # compute r using orbit equation:
    a = (ra + rp) / 2
    r = (a * (1 - ecc ** 2) / (1 + ecc * np.cos(TA)))
    r_perifocal = r * np.array([np.cos(TA), np.sin(TA), 0])

    # compute v using stupid random bullshit equations:
    p = a * (1 - ecc ** 2)
    # print(p)
    coeff = np.sqrt(mu / p)
    v_perifocal = coeff * np.array([ - np.sin(TA), (ecc +  np.cos(TA)), 0])

    # convert r & v to ECI:
    rotix = f.perifocal2ECI(RAAN, inc, AoP)
    r_ECI = rotix @ r_perifocal
    v_ECI = rotix @ v_perifocal

    return r_ECI, v_ECI

# declare constants:
mu = c.mu_earth

# declare TLE values...
# format: [M, ecc, RAAN, inc, AoP, ra, rp, n]

# GEO: INTELSAT-33E-DEB:
GEO = np.array([21.5383, 0.0, 85.7856, 0.9979, 252.6559, z2rabtearth(35759), z2rabtearth(35722), 1.00437626])

# MEO: FREGAT:
MEO = np.array([270.2279, 0.0008031, 183.5260, 0.0773, 266.3417, z2rabtearth(7689), z2rabtearth(7666), 5.20983481])

# LEO1: POEM:
LEO1 = np.array([138.4452, 0.0, 276.8703, 9.9523, 221.3777, z2rabtearth(523), z2rabtearth(487), 15.20682587])

# LEO2: DEMOSAT/FALCON1:
LEO2 = np.array([224.4630, 0.0, 320.0996, 9.3468, 135.6367, z2rabtearth(589), z2rabtearth(575), 14.95278048])

"""
BEHOLD... IT IS TIME TO GET OUR ECI r & v VECTORS!!!

NOTE: all of these goobers are expressed in ECI--just wanted to make that clear.
"""

# GEO satellite:
r_GEO, v_GEO = TLEdata2RV0(GEO[0],GEO[1],GEO[2],GEO[3],GEO[4],GEO[5],GEO[6])

# MEO satellite:
r_MEO, v_MEO = TLEdata2RV0(MEO[0],MEO[1],MEO[2],MEO[3],MEO[4],MEO[5],MEO[6])

# LEO1 satellite:
r_LEO1, v_LEO1 = TLEdata2RV0(LEO1[0],LEO1[1],LEO1[2],LEO1[3],LEO1[4],LEO1[5],LEO1[6])

# LEO2 satellite:
r_LEO2, v_LEO2 = TLEdata2RV0(LEO2[0],LEO2[1],LEO2[2],LEO2[3],LEO2[4],LEO2[5],LEO2[6])

"""
NOW IT'S PROPAGATIN' TIME... WELL THE FIRST ONE.

We need to figure out which TLE was the most recent, and then how much time there is (in seconds... oh god) between
that time and the time all of the other TLEs were recorded... FUN!
"""

# epochs: TLEs all recorded on the same day, 25 (year) 316 (day of year) (YAY!!!! YAYYYYYY!!! THIS MAKES MY LIFE SO MUCH EASIER! SERIOUSLY! YAY!!!!!!!!!!!!!!)
# anyways... storing each time on that day as a FRACTION OF DAY

dayseconds = 24 * 3600
t0_GEO = 0.14507882 * dayseconds
t0_MEO = 0.66227286 * dayseconds # <= this one is the most recent one... we want to zero everything to this instant
t0_LEO1 = 0.14146735 * dayseconds
t0_LEO2 = 0.56274537 * dayseconds

# figure out time spacing between other satellites' TLEs and the MEO satellite's TLE...
delt_GEO = t0_MEO - t0_GEO
delt_LEO1 = t0_MEO - t0_LEO1
delt_LEO2 = t0_MEO - t0_LEO2

# propogate GEO &  LEO satellites to the same instant as the MEO satellite (most recent timestamp)...
r0_GEO, v0_GEO = f.chi_propogator(r_GEO, v_GEO, mu, delt_GEO)
r0_LEO1, v0_LEO1 = f.chi_propogator(r_LEO1, v_LEO1, mu, delt_LEO1)
r0_LEO2, v0_LEO2 = f.chi_propogator(r_LEO2, v_LEO2, mu, delt_LEO2)
r0_MEO, v0_MEO = r_MEO, v_MEO # I just wanted him to feel special, ok? poor little guy could use a break. ;)

print("\n===== GEO =====")
print("r0_GEO [km]:", r0_GEO)
print("v0_GEO [km/s]:", v0_GEO)

print("\n===== LEO1 =====")
print("r0_LEO1 [km]:", r0_LEO1)
print("v0_LEO1 [km/s]:", v0_LEO1)

print("\n===== LEO2 =====")
print("r0_LEO2 [km]:", r0_LEO2)
print("v0_LEO2 [km/s]:", v0_LEO2)

print("\n===== MEO =====")
print("r0_MEO [km]:", r0_MEO)
print("v0_MEO [km/s]:", v0_MEO)

# so now all of our satellites are zeroed to the same instant... YAY!!!

"""
TRANSFER 1 MANEUVER STRATEGY
- Combined inclination and Hohmann burn 1/2 at chosen apogee
- Wait a convenient amount of time before first Hohmann transfer burn
  to negate having to perform an actual phasing maneuver

ASSUMPTIONS:
- Take magnitude of initial r vector for all circular orbits as radius
- Eccentricity of circular orbits (ecc <= 0.001) approximated as 0
"""

deltav = []

# FOR ALL CIRCULAR ORBITS: take magnitudes 'R' as radius (based on initial position vector)
RGEO = np.linalg.norm(r0_GEO)
RMEO = np.linalg.norm(r0_MEO)
RLEO2 = np.linalg.norm(r_LEO2) # LEO1 gets left out :'(

# CHARACTERIZE GEO ORBIT
T_GEO = f.circ_period_eq1(RGEO, mu) # period of GEO orbit [sec]

# CHARACTERIZE MEO ORBIT
T_MEO = f.circ_period_eq1(RMEO, mu)

# CHARACTERIZE TRANSFER ORBIT
ra_trans = RGEO
rp_trans = RMEO

e_trans = f.ell_eccentricity(ra_trans, rp_trans)
a_trans = f.ell_semimajoraxis(ra_trans, rp_trans)
T_trans = f.ell_period(a_trans, mu) # period of transfer orbit [sec]

"""
Wait a sufficient amount of time to negate having to do a phasing maneuver...
THEOREM:
- Find T of parking and transfer orbit
- Find tH--the amount of time relative to perigee that the satellite in the lower orbit
  must wait to be in the appropriate position
    - For this to work using the Hohmann transfer method, the apse lines of both orbits 
      must be parallel with one another (not antiparallel either... need true anomaly to
      be based on the same starting point for calculations to be easier)

Okay, so walking through it, I realize that the geometry would work out way better
if I made these two orbits coplanar. That way, I can just take the difference of 
the two satellites' true anomalies and use that to describe the total angle 
between them (instead of also having to worry about using the inclination as well;
I am curious about this though to avoid having to do an inclination change... something
I'll do later)
"""

# DEFINE PROBE'S INITIAL ORBIT:
PROBE = GEO # define orbit of the probe collecting the debris (inherits GEO's properties)
r0_PROBE, v0_PROBE = r0_GEO, v0_GEO

r0_GEO_mag = np.linalg.norm(r0_GEO)
r0_MEO_mag = np.linalg.norm(r0_MEO)
r0_PROBE_mag = r0_GEO_mag
v0_PROBE_mag = np.linalg.norm(v0_GEO)

# want to do inclination change along node line, find out where to do this burn:
n1 = np.cross(r0_MEO, v0_MEO) / np.linalg.norm(np.cross(r0_MEO, v0_MEO) )
n2 = np.cross(r0_GEO, v0_GEO) / np.linalg.norm(np.cross(r0_GEO, v0_GEO) )

N = np.cross(n1, n2) / np.linalg.norm(np.cross(n1, n2) ) # unit vector along node line
inc_burn_position = RGEO * N # determine probe position at one of the nodes

# PROPOGATE until satellite is in position for the inc change burn at NODE LINE position...
# declare initial conditions:
x = inc_burn_position[0]
y = inc_burn_position[1]
z = inc_burn_position[2]

# declare other variables:
theta_error = np.inf
theta_tol = 1e-6
total_timestep = 60779 # final time of propogation[sec]
dt = 0.0001
r_PROBE = r0_PROBE
v_PROBE = v0_PROBE

# once the two vectors are within a negligable angular displacement (theta_error), consider us in position for inclination change burn...
while (theta_error) > theta_tol:
    total_timestep += dt
    r_PROBE, v_PROBE = f.chi_propogator(r0_PROBE, v0_PROBE, mu, total_timestep)
    theta_error = np.rad2deg(np.arccos((np.dot(r_PROBE, inc_burn_position) / ((np.linalg.norm(r_PROBE) * np.linalg.norm(inc_burn_position))))))

# have r and v for probe; need them for all other satellites at new time:
r_GEO, v_GEO = f.chi_propogator(r0_GEO, v0_GEO, mu, total_timestep)
r_MEO, v_MEO = f.chi_propogator(r0_MEO, v0_MEO, mu, total_timestep)
r_LEO1, v_LEO1 = f.chi_propogator(r0_LEO1, v0_LEO1, mu, total_timestep)
r_LEO2, v_LEO2 = f.chi_propogator(r0_LEO2, v0_LEO2, mu, total_timestep)

# PERFORM inclination burn:
inc_change_angle = PROBE[3] - MEO[3]
deltav.append(2 * (np.linalg.norm(v_PROBE) * np.sin((inc_change_angle/2))))

# update PROBE's orbit:
PROBE[2] = MEO[2] # RAAN
PROBE[3] = MEO[3] # inc
PROBE[4] = MEO[4] # AoP

# rotate r and v, same magnitudes since burn was performed along the node line:
inc_change_rotix = la.rotix_thetaz_getRHR(inc_change_angle) @ la.rotix_thetax_getRHR(0) @ la.rotix_thetaz_getRHR(0)
r_PROBE = inc_change_rotix @ r_PROBE
v_PROBE = inc_change_rotix @ v_PROBE
RPROBE = np.linalg.norm(r_PROBE)

# format: [M, ecc, RAAN, inc, AoP, ra, rp, n]

# PREPARE for Hohmann transfer:

# FIND angle necessary between these two satellites for them to rendezvous at perigee without a phasing maneuver:
# time relative to perigee for MEO satellite to reach perigee (rendezvous point):
tH = T_MEO - 0.5 * T_trans # [sec]
n_MEO = MEO[7] # mean motion of MEO satellite
M = n_MEO * tH # mean anomaly from anomaly from mean motion

TA_coastStart = np.rad2deg(n_MEO * tH % (2 * np.pi)) # position of satellite for coast to start [deg]
theta_diff = TA_coastStart - 180 # the angle between the GEO and MEO satellites... will help us find what point we should propogate to

print(theta_diff)

"""
Now that we have the angular displacement we want between the two orbits,
we can propogate forward until the soonest point in time that the two objects
are at the proper angular displacement, theta.
"""

theta_error = np.inf
theta_tol = 1
dt = 0.1
total_timestep = T_MEO - T_MEO / 27.7715

# initial conditions:
r0_PROBE = r_PROBE
v0_PROBE = v_PROBE
r0_MEO = r_MEO
v0_MEO = v_MEO

# # we're in position once close to proper position (within a tolerance):
# while theta_error > theta_tol:
#     total_timestep += dt
#     r_PROBE, v_PROBE = f.chi_propogator(r0_PROBE, v0_PROBE, mu, total_timestep)
#     r_MEO, v_MEO = f.chi_propogator(r0_MEO, v0_MEO, mu, total_timestep)
#     theta_current = np.rad2deg((np.dot(r_MEO, r_PROBE) / (RMEO * RPROBE)))
#     theta_error = abs(theta_current - theta_diff)
#     print(theta_error)
    
# # check to see that we arrive in the right place
# va_trans = [f.ell_vtransverse(180, ra_trans, rp_trans, mu), 0, 0]
# va_trans = f.perifocal2ECI(PROBE[2], PROBE[3], PROBE[4]) @ va_trans

# r_trans, v_trans = f.chi_propogator(r_PROBE, va_trans, mu, T_trans / 2)
# r_MEO, v_MEO = f.chi_propogator(r_MEO, v_MEO, mu, T_trans / 2)
# print(r_trans)
# print(r_MEO)

# # have all other satellites catch up
# r_GEO, v_GEO = f.chi_propogator(r_GEO, v_GEO, mu, total_timestep)
# r_LEO1, v_LEO1 = f.chi_propogator(r_LEO1, v_LEO1, mu, total_timestep)
# r_LEO2, v_LEO2 = f.chi_propogator(r_LEO2, v_LEO2, mu, total_timestep)

# # # propogate to the point of alignment:
# theta_actual = np.rad2deg(np.arccos(np.dot(r_GEO, r_MEO) / (RGEO * RMEO)))
# print(" ")
# print(theta_actual)
# print(theta_diff)

# dt = 0.0001 # timestep [sec]
# total_delta_t = T_MEO - T_MEO / 11.4 + 6# initial guess for time instant based on previous trials/geometry
# while abs(theta_actual - theta_diff) > theta_tol:
#     total_delta_t += dt
#     # propogate orbits to the next timestep
#     r_GEO, v_GEO = f.chi_propogator(r0_GEO, v0_GEO, mu, total_delta_t)
#     r_MEO, v_MEO = f.chi_propogator(r0_MEO, v0_MEO, mu, total_delta_t)

#     r_GEO_mag = np.linalg.norm(r_GEO)
#     r_MEO_mag = np.linalg.norm(r_MEO)

#     # use the definition of the dot product to find the angle between the two vectors
#     theta_actual = np.rad2deg(np.arccos(np.dot(r_GEO, r_MEO) / (r_GEO_mag * r_MEO_mag)))
#     # print(abs(theta_actual - theta_diff))

# va_trans = [f.ell_vtransverse(180, ra_trans, rp_trans, mu), 0, 0]
# va_trans = f.perifocal2ECI(PROBE[2], PROBE[3], PROBE[4]) @ va_trans

# r_trans, v_trans = f.chi_propogator(r_PROBE, va_trans, mu, T_trans / 2)
# r_MEO, v_MEO = f.chi_propogator(r_MEO, v_MEO, mu, T_trans / 2)
# print(r_trans)
# print(r_MEO)
