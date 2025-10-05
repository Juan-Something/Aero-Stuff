import matplotlib as mpL
import scipy as sciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Question #1
def airViscositySutherland(TCelsius):


    tRef = 275.15
    muRef = 1.716e-5
    sAir = 110.4

    TKelvin = TCelsius + 273.15

    muAir = muRef * (TKelvin/tRef)**1.5 * (tRef + sAir) / (TKelvin + sAir)
    return muAir

tempArray = np.linspace(-50, 500, 200)

viscosityArray = airViscositySutherland(tempArray)

plt.figure(1)
plt.plot(tempArray, viscosityArray)
plt.xlabel('Temperature, K')
plt.ylabel('Viscosity, kg/(m*s)')
plt.title("Air Viscosity vs Temperature")
plt.grid()
plt.show(block = False)

# FS1

def standardAtmshpereModel(h):

    R = 287.0
    g = 9.80665

     # Calculate T, P, Rho based on height
    if 0 <= h <= 11:
        a = -0.0065
        T = 288.16 + a * (1000*h)
        P = 101.325 * (T / 288.16) ** (-g / (a * R))
        Rho = 1.2250 * (T / 288.16) ** (-(g / (a * R) + 1))
    elif 11 < h <= 25:
        T = 216.66
        P = 22.6272 * np.exp(-g / (T*R) * (1000*h - 11000))
        Rho = 0.36383 * np.exp(-g / (T*R) * (1000*h - 11000))
    elif 25 < h <= 47:
        a = 0.0030
        T = 216.66 + a * (1000*h - 25000)
        P = 2.48733 * (T / 216.66) ** (-g / (a * R))
        Rho = 0.039995 * (T / 216.66) ** (-(g / (a * R) + 1))
    elif 47 < h <= 53:
        T = 282.66
        P = 0.12033 * np.exp(-g / (T*R) * (1000*h - 47000))
        Rho = 0.0014831 * np.exp(-g / (T*R) * (1000*h - 47000))
    elif 53 < h <= 79:
        a = -0.0045
        T = 282.66 + a * (1000*h - 53000)
        P = 0.058261 * (T / 282.66) ** (-g / (a * R))
        Rho = 0.000718077 * (T / 282.66) ** (-(g / (a * R) + 1))
    elif 79 < h <= 90:
        T = 165.66
        P = 0.0010078 * np.exp(-g / (T*R) * (1000*h - 79000))
        Rho = 0.0000211951 * np.exp(-g / (T*R) * (1000*h - 79000))
    elif 90 < h <= 100:
        a = 0.0040
        T = 165.66 + a * (1000*h - 90000)
        P = 0.000104233 * (T / 165.66) ** (-g / (a * R))
        Rho = 0.00000219214 * (T / 165.66) ** (-(g / (a * R) + 1))
    else:
        raise ValueError("Height out of range (0-100 km)")
    
    # Sutherland's law for dynamic viscosity
    mu0 = 1.716e-5  # kg/(m*s)
    T0 = 273.15     # K
    S = 110.4       # K
    mu = mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
    return T, P, Rho, mu

altitudeArray = np.linspace(0, 100, 200)

results = np.array([standardAtmshpereModel(h) for h in altitudeArray])
tempArray = results[:, 0]
pressureArray = results[:, 1]
densityArray = results[:, 2]
viscosityArray = results[:, 3]
plt.figure(2, figsize=(10, 8))
plt.subplot(1, 3, 1)
plt.plot(altitudeArray, pressureArray)
plt.xlabel('Altitude, km')
plt.ylabel('Pressure, kPa')
plt.title('Pressure vs Altitude')
plt.grid()
plt.subplot(1, 3, 2)
plt.plot(altitudeArray, densityArray)
plt.xlabel('Altitude, km')
plt.ylabel('Density, kg/m^3')
plt.title('Density vs Altitude')
plt.grid()
plt.subplot(1, 3, 3)
plt.plot(altitudeArray, viscosityArray)
plt.xlabel('Altitude, km')
plt.ylabel('Viscosity, kg/(m*s)')
plt.title('Viscosity vs Altitude')
plt.grid()
plt.tight_layout()
plt.show()

# FS2

