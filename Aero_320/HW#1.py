import scipy as sciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpL


## Question 1

objectDistance = np.array([.1, -.2, .95]) # km
phi = np.radians(2)
theta = np.radians(-1)
epsilon = np.radians(40)

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

lvlH = rotationMatrix(phi, theta, epsilon)
print("Rotation Matrix from Level to Horizon Frame:")
print(lvlH)

translvlH = np.transpose(lvlH)
horizonObjectDistance = np.dot(translvlH, objectDistance)
print("Object Distance in Horizon Frame (km):")
print(horizonObjectDistance)
