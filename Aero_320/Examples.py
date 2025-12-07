import scipy as sciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpL
import sympy as sp
from scipy import integrate 


# 1
J = np.array([[1/2, 0, 0], [0, 1/3, 0], [0, 0, 1/3]])
t = sp.symbols('t', real = True)
cosT = sp.cos(t)
sinT = sp.sin(t)

omega = np.array([0,1,cosT])

torque = J * omega + np.cross(omega, (J * omega))
sp.pprint(torque)

f = lambda r, theta, z: (r*np.sin(theta)**2 + z**2)*r  

# 2
h = 71 # m
r = 9/2 # m
m = 180e+3
#pX = integrate.trapezoid((r*cosT)**2 + (r*sinT)**2)
# pY = integrate.tplquad(f,0,r, 0, 2*np.pi, 0, h)

J2 = np.array([[(m * r**2 / 4) + (m * h**2)/3, 0, 0], [0, (m * r**2 / 4) + (m * h**2)/3, 0], [0, 0, (m*r**2 / 2)]])
print(J2)

# t1,x = integrate.solve_ivp(fun= lambda: t1,x,  )

thrust = 9.81e+3
dryMass = 160e+3
fuelMass = 20e+3
mass = dryMass + fuelMass




def eulerAnglePropagation (t, x, thrust, J):

    dx = np.zeros(len(x))
    w = x[0:2,0]
    phi = x[3,0]
    theta = x[4,0]
    psi = x[5,0]
    T = x[6:8,0]
    dx[0:2, 0] = np.linalg.inv(J) * (T - np.linalg.skew(w) * J * w)
    matrix = np.array([[np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)*np.sin(theta)], [0, np.cos(phi)*np.cos(theta)], [0, np.sin(phi), np.cos(phi)]])
    dx[3:5, 0] = (1/np.cos(theta)) * matrix

    dx[6:8,0] = np.array([0,0,np.cos(t)*thrust])
    return dx

# Visualize

plt.figure()
plt.plot(t, x[:, 0:2])

# 3

