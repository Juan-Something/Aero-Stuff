import matplotlib as mpL
import scipy as sciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rVect = np.array([3450,   -1700,  7750])
vVect = np.array([5.4, -5.4, 1])
state1 = np.concatenate([rVect, vVect])
rTol = 1e-8
aTol = 1e-8

muEarth = 398600
tSpan = [0, 24*60*60]

def twoBodyODE(t, state, muEarth): 
    x, y, z, dx, dy, dz =state
    rMag = np.linalg.norm([x, y, z])

    ddx = -muEarth * x / rMag**3
    ddy = -muEarth * y/ rMag**3
    ddz = -muEarth * z / rMag**3

    return [dx, dy, dz, ddx, ddy, ddz]

solution = sciPy.integrate.solve_ivp(twoBodyODE, tSpan, state1, args=(muEarth,), rtol=rTol, atol=aTol)

x, y, z = solution.y[0], solution.y[1], solution.y[2]

# Plot
fig =   plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x, y, z)
ax.set_xlabel('x, km')
ax.set_ylabel('y, km')
ax.set_zlabel('z, km')
plt.show()
