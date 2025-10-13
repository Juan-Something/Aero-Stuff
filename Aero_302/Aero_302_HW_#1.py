import matplotlib as mpL
import scipy as sp
import scipy.integrate as spi
import numpy as np
import pandas as pd
from Aero_302.standardAtmosphereModel import standardAtmshpereModel
import matplotlib.pyplot as plt

# Question #1

"""
Sutherland's Law is used, assumptions made are that air behaves as an ideal gas and that the temperature range is valid for Sutherland's law.
We will ignore affects such as ionization and vibrations of molecules at high temperatures. 
"""
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

    h = np.array(h, ndmin=1)
    R = 287.0
    g = 9.80665

    T = np.zeros_like(h)
    P = np.zeros_like(h)
    Rho = np.zeros_like(h)

     # Calculate T, P, Rho based on height
    mask = (h >= 0) & (h <= 11)
    a = -0.0065
    T[mask] = 288.16 + a * (1000*h[mask])
    P[mask] = 101.325 * (T[mask] / 288.16) ** (-g / (a * R))
    Rho[mask] = 1.2250 * (T[mask] / 288.16) ** (-(g / (a * R) + 1))
    # --- Layer 2: 11–25 km ---
    mask = (h > 11) & (h <= 25)
    T[mask] = 216.66
    P[mask] = 22.6272 * np.exp(-g / (T[mask]*R) * (1000*h[mask] - 11000))
    Rho[mask] = 0.36383 * np.exp(-g / (T[mask]*R) * (1000*h[mask] - 11000))
   
    # --- Layer 3: 25–47 km ---
    mask = (h > 25) & (h <= 47)
    a = 0.0030
    T[mask] = 216.66 + a * (1000*h[mask] - 25000)
    P[mask] = 2.48733 * (T[mask] / 216.66) ** (-g / (a * R))
    Rho[mask] = 0.039995 * (T[mask] / 216.66) ** (-(g / (a * R) + 1))

    # --- Layer 4: 47–53 km ---
    mask = (h > 47) & (h <= 53)
    T[mask] = 282.66
    P[mask] = 0.12033 * np.exp(-g / (T[mask]*R) * (1000*h[mask] - 47000))
    Rho[mask] = 0.0014831 * np.exp(-g / (T[mask]*R) * (1000*h[mask] - 47000))

    # --- Layer 5: 53–79 km ---
    mask = (h > 53) & (h <= 79)
    a = -0.0045
    T[mask] = 282.66 + a * (1000*h[mask] - 53000)
    P[mask] = 0.058261 * (T[mask] / 282.66) ** (-g / (a * R))
    Rho[mask] = 0.000718077 * (T[mask] / 282.66) ** (-(g / (a * R) + 1))

    # --- Layer 6: 79–90 km ---
    mask = (h > 79) & (h <= 90)
    T[mask] = 165.66
    P[mask] = 0.0010078 * np.exp(-g / (T[mask]*R) * (1000*h[mask] - 79000))
    Rho[mask] = 0.0000211951 * np.exp(-g / (T[mask]*R) * (1000*h[mask] - 79000))

    # --- Layer 7: 90–100 km ---
    mask = (h > 90) & (h <= 100)
    a = 0.0040
    T[mask] = 165.66 + a * (1000*h[mask] - 90000)
    P[mask] = 0.000104233 * (T[mask] / 165.66) ** (-g / (a * R))
    Rho[mask] = 0.00000219214 * (T[mask] / 165.66) ** (-(g / (a * R) + 1))
    
    mask = (h < 0) | (h > 100)
    if np.any(mask):
        raise ValueError("Altitude out of range (0-100 km)")
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
plt.show(block = False)

# FS2

conduitWidth = 2 # m

z = np.linspace(0, 100, 2000) # Altitude from 0 to 100 km
P = np.zeros_like(z) 
T, P, Rho, mu = standardAtmshpereModel(z)
zMeters = z * 1000 # Convert km to m

geometricCentroid = 100 / 2
print(f"Geometric centroid height: {geometricCentroid:.4f} km")

integrateTest = np.trapezoid(P * 1000, zMeters)
integrateTest2 = np.trapezoid(P *1000 * zMeters, zMeters)

zCp = integrateTest2 / integrateTest
zCpKm = zCp / 1000
print(f"Center of pressure height: {zCpKm:.4f} km")

conduitForce = conduitWidth * integrateTest
print(f"Force on one side of the conduit: {conduitForce:.4} N")

centroidMoment = conduitForce * (zCpKm - 50)
print(f"Moment about the center of the conduit: {centroidMoment:.4f} Nm")

# FS4

# FM

"""
-1/g*dU/dh = (W/CdS)^-1 * (rho*U^2/2) 
"""
g = 9.80665 # m/s^2
ballisticCoefficient = 4800 # kg/m^2
initialVelocity = 11200 # m/s
initialAltitude = 100 # km
dropZ = np.linspace(100, 0, 2000)
Rho2 = np.zeros_like(dropZ)
T2, P2, Rho2, mu2 = standardAtmshpereModel(dropZ)
dropZMeters = dropZ * 1000


g = 9.80665  # m/s^2
ballisticCoefficient = 4800  # kg/m^2

initialVelocity = 11200  # m/s (Corrected from 112000)
initialAltitude = 100000  # meters (100 km)

# Prepare density vs altitude arrays (already calculated above)
# dropZ is in km, so dropZMeters is already defined
# Rho2 is density from Standard Atmosphere model

# Define time-domain ODE system: y = [h(t), U(t)]
def rhs(t, y):
    h, U = y
    
    # Clamp altitude if it goes below ground
    if h <= 0:
        return [0, 0]

    # Interpolate density at current altitude
    rho = np.interp(h, dropZMeters[::-1], Rho2[::-1])

    dhdt = -U  # descending
    dUdt = - (g / ballisticCoefficient) * (rho * U**2 / 2)

    return [dhdt, dUdt]

# Stop when altitude reaches 0
def hit_ground(t, y):
    return y[0]  # stop when h = 0
hit_ground.terminal = True
hit_ground.direction = -1  # only trigger when descending

# Solve
sol = sp.integrate.solve_ivp(
    rhs,
    (0, 20),  # time span guess
    [initialAltitude, initialVelocity],
    
    max_step=1.0,
    rtol=1e-6,
    atol=1e-8
)

t = sol.t
h = sol.y[0]
U = sol.y[1]

# Plot Velocity vs Altitude
plt.figure(4)
plt.plot(U / 1000, h / 1000)
plt.xlabel('Velocity, km/s')
plt.ylabel('Altitude, km')
plt.title('Velocity vs Altitude during Descent (solve_ivp)')
plt.gca().invert_yaxis()  # ground at bottom
plt.grid()
plt.show()

plt.figure(5)
plt.plot(t, U / 1000)
plt.xlabel('Time, s')
plt.ylabel('Velocity, km/s')
plt.title('Velocity vs Time during Descent (solve_ivp)')
plt.grid()
plt.show()


print(f"Final Velocity at impact = {U[-1]:.2f} m/s")
