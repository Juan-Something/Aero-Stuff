import scipy as sciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpL
import sympy as sp
from scipy import integrate 

angularVelo = np.array([1, -2, 0.5])
tSpan = (0,10)
tSpanEvaluate = np.linspace(tSpan[0], tSpan[1], 1000)
"""
phiPrime = np.array([1, np.sin(psi)*np.tan(theta), np.cos(phi), np.tan(theta)])
thetaPrime = np.array([0, np.cos(psi), -np.sin(psi)])
psiPrime = np.array([0, np.sin(psi) / np.cos(theta), np.cos(psi) / np.cos(theta)])
"""

# Assuming 3-2-1 rotation, only theta and phi are used 
def eulerRate (t, y):
    phi, theta, psi = y
    cosPhi = np.cos(phi)
    sinPhi = np.sin(phi)
    cosTheta = np.cos(theta)

    if abs(cosTheta) < 1e-8:
        cosTheta = np.sign(cosTheta) * 1e-8

    a, b, c = angularVelo
    dPhi = a + sinPhi*np.tan(theta)*b + cosPhi * np.tan(theta)*c
    dTheta = cosPhi*b - sinPhi*c
    dPsi = (sinPhi*c + cosPhi*b) / cosTheta

    return [dPhi, dTheta, dPsi]

def quaternionRate (t,q):
    e = q[:3]
    eta = q[3]
    dE = .5*(eta*angularVelo + np.cross(e, angularVelo))
    dEta = -.5*np.dot(e, angularVelo)
    return np.hstack((dE, dEta))


initialVelo = [0, 0, 0]
initialQuaternion = [0, 0, 0, 1]



eulerSolution = integrate.solve_ivp(eulerRate, tSpan, initialVelo, t_eval= tSpanEvaluate, rtol = 1e-8, atol = 1e-8)
phi, theta, psi = eulerSolution.y
t = eulerSolution.t

quaternionSolution = integrate.solve_ivp(quaternionRate, tSpan, initialQuaternion, t_eval= tSpanEvaluate, rtol = 1e-8, atol = 1e-8)
quaternionT = quaternionSolution.t
q = quaternionSolution.y.T
q = q / np.linalg.norm(q, axis=1, keepdims=True)  # normalize


plt.figure()
plt.plot(t, phi, label='phi')
plt.plot(t, theta, label='theta')
plt.plot(t, psi, label='psi')
plt.xlabel('time [s]'); plt.ylabel('angle [rad]'); plt.legend(); plt.title('Euler 3-2-1')

plt.figure()
plt.plot(quaternionT, q[:,0], label='e1')
plt.plot(quaternionT, q[:,1], label='e2')
plt.plot(quaternionT, q[:,2], label='e3')
plt.plot(quaternionT, q[:,3], label='eta')
plt.xlabel('time [s]'); plt.ylabel('quaternion'); plt.legend(); plt.title('Quaternion components')

plt.tight_layout()
plt.show()      