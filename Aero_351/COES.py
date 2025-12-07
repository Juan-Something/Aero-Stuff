import matplotlib as mpL
import scipy as sciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rVector = np.array([9031.5,   -5316.9,  -1647.2]) #hm
vVector = np.array([-2.8640, 5.1112, -5.0805]) # km/s

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


