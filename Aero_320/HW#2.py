import scipy as sciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpL
import sympy as sp
from Aero_320 import symbolicRotation


# Problem 1
phi, theta, psi = sp.symbols('phi theta psi')
order = 'xzy'
angles = (phi, theta, psi)
R = symbolicRotation.symbolic_rotation_matrix(order, angles)
print("Rotation matrix for order 'yzx':")
sp.pprint(R)




## Problem 2

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
print(transLvlh)

## Symbolic Rotation Calculator
"""
import sympy as sp

def rotation_matrix(axis, angle):
    c = sp.cos(angle)
    s = sp.sin(angle)
    if axis == 'x':
        return sp.Matrix([[1, 0, 0],
                          [0, c, s],
                          [0, -s, c]])
    elif axis == 'y':
        return sp.Matrix([[c, 0, -s],
                          [0, 1, 0],
                          [s, 0, c]])
    elif axis == 'z':
        return sp.Matrix([[c, s, 0],
                          [-s, c, 0],
                          [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

def symbolic_rotation_matrix(order, angles):
"""
"""
    order: string of axes, e.g. 'zyx'
    angles: list/tuple of sympy symbols, e.g. (phi, theta, psi)
    """
"""
    if len(order) != 3 or len(angles) != 3:
        raise ValueError("Order and angles must have length 3.")
    R = sp.eye(3)
    for axis, angle in zip(order, angles):
        R = R * rotation_matrix(axis, angle)
    return sp.simplify(R)

"""