import scipy as sciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpL
import sympy as sp
from Aero_320 import symbolicRotation

phi = np.radians(10)
theta = np.radians(-5)
epsilon = np.radians(20)

def rotationMatrix(phi, theta, epsilon):
    rPhi = np.array([[1, 0, 0],
                     [0, np.cos(phi), np.sin(phi)],    
                     [0, -np.sin(phi), np.cos(phi)]])


    rTheta = np.array([[np.cos(theta), 0, -np.sin(theta)],
                       [0, 1, 0],
                       [np.sin(theta), 0, np.cos(theta)]])
    
    rEpsilon = np.array([[np.cos(epsilon), np.sin(epsilon), 0],
                         [-np.sin(epsilon), np.cos(epsilon), 0],
                         [0, 0, 1]])
    
    rMatrix = rPhi @ rTheta @ rEpsilon
    return rMatrix
phi2, theta2, psi2 = sp.symbols('phi theta psi')
order = 'xyz'
angles = (phi2, theta2, psi2)
symbolicRot = symbolicRotation.symbolic_rotation_matrix(order,angles)
sp.pprint(symbolicRot)




rotMatrix = rotationMatrix(phi, theta, epsilon)
rotMatrixInverse = np.transpose(rotMatrix)
print("Rotation Matrix from tf to t0:")
print(rotMatrixInverse)



rotENUtoACF = np.array([[0.9361, -.3407, -0.0872], [-0.3510, -0.9202, -0.1730], [-0.0213, 0.1925, -0.9811]])

eta = ((np.trace(rotENUtoACF)+1)**(1/2))/2

print(eta)

vecOne = (rotENUtoACF[1,2] - rotENUtoACF[2,1])/(4*eta)
vecTwo = (rotENUtoACF[0,2] - rotENUtoACF[2,0])/(4*eta)
vecThree = (rotENUtoACF[0,1] - rotENUtoACF[1,0])/(4*eta)

quaternMatrix = np.array([vecOne, vecTwo, vecThree, eta])



print(quaternMatrix)


magTest = np.linalg.norm(quaternMatrix)
print(magTest)

phiQuaternRad = 2*np.cosh(eta)
phiQuaternDeg = np.rad2deg(phiQuaternRad)

a = quaternMatrix * np.sin(phiQuaternDeg/2)
#print(a)









