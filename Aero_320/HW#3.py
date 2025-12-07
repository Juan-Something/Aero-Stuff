import scipy as sciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpL
import sympy as sp
from Aero_320 import symbolicRotation

rVector = np.array([6783, 3391, 1953]) # km
vVector = np.array([-3.50, 4.39, 4.44]) # km/s

rMag = np.linalg.norm(rVector)

rvCrossProd = np.cross(rVector, vVector)
rvMag = np.linalg.norm(rvCrossProd)

zLVLH = -rVector / rMag
yLVLH = -rvCrossProd / rvMag
xLVLH = np.cross(yLVLH, zLVLH)

lvlhMatrix = np.array([xLVLH, yLVLH, zLVLH])

transLvlh = np.transpose(lvlhMatrix)

print("Rotation Matrix from ECI to LVLH Frame:")
print(lvlhMatrix)
# b
principalAngle = np.arccos((np.trace(lvlhMatrix) - 1)/2)

a = (1.0 / (2.0*np.sin(principalAngle))) * np.array([
    lvlhMatrix[1,2] - lvlhMatrix[2,1],
    lvlhMatrix[2,0] - lvlhMatrix[0,2],
    lvlhMatrix[0,1] - lvlhMatrix[1,0]
])

print(f"principal: {a}")
# c
theta = np.atan2((lvlhMatrix[1,2]),(lvlhMatrix[2,2]))
thetaDeg = np.rad2deg(theta)
phi = -np.arcsin(lvlhMatrix[0,2])
phiDeg = np.rad2deg(phi)
psi = np.atan2(lvlhMatrix[0,1],lvlhMatrix[0,0])
psiDeg = np.rad2deg(psi)


print(f"theta: {thetaDeg: .4f} degrees")
print(f"phi: {phiDeg: .4f} degrees")
print(f"psi: {psiDeg: .4f} degrees")


# d

eta = ((np.trace(lvlhMatrix) + 1)**(1/2)) / 2
epsilon1 = np.absolute((lvlhMatrix[1,2] - lvlhMatrix[2,1]) / (4*eta))
epsilon2 = np.sign(lvlhMatrix[0,1])*np.absolute((lvlhMatrix[2,0] - lvlhMatrix[0,2]) / (4*eta))
epsilon3 = np.sign(lvlhMatrix[0,2])*np.absolute((lvlhMatrix[0,1] - lvlhMatrix[1,0]) / (4*eta))

quaternion = np.array([epsilon1, epsilon2, epsilon3, eta])

print(f"Quaternion: {quaternion}")