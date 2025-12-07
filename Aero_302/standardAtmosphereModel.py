import numpy as np


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