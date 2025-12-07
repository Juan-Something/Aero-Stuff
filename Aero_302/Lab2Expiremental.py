import os, re
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

folder = r"Z:\VS Code\Test Environment\.venv\Aero_302\Lab 2"
files = sorted(f for f in os.listdir(folder) if f.endswith(".mat"))

# constants (your values)
pAtmo = 1005.7 * 100
R = 287.05
T = 18.6 + 273.15
rho = pAtmo / (R * T)
mu = 1.87e-5
d = 0.15875
r = d / 2
dtheta = np.deg2rad(15)
theta = np.deg2rad(15 + np.arange(24) * 15)

Cd_vals, Re_vals, names = [], [], []

for f in files:
    path = os.path.join(folder, f)
    data = sio.loadmat(path)
    pData = data["P"]

    pTotal = np.mean(pData[:, 0])
    pInf   = np.mean(pData[:, 1])
    q      = pTotal - pInf

    # mean pressure at taps 3..26
    Pbar   = pData.mean(axis=0)[3:27]

    # force per unit span for each panel
    dF = (Pbar - pInf) * r * dtheta
    D  = dF * np.cos(theta)

    # outputs you asked for
    cD = -np.sum(D) / (q * d)
    V  = np.sqrt(2 * q / rho)
    Re = rho * V * d / mu

    Cd_vals.append(cD)
    Re_vals.append(Re)
    names.append(f)

    print(f"{f}:  Cd={cD:.3f},  Re={Re:.2e}")

# optional: plot Cd vs Re
plt.figure()
plt.scatter(Re_vals, Cd_vals)
plt.xlabel("Re")
plt.ylabel("Cd")
plt.title("Cd vs Re")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
