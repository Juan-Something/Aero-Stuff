import matplotlib as mpL
import scipy as sp
import scipy.integrate as spi
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


S5G1Data = sio.loadmat("Z:\VS Code\Test Environment\.venv\Aero_302\Lab 2\S5G1RPM500STRIP.mat")
pData = S5G1Data["P"]




pTotal = np.mean(pData[:,0])
pInf = np.mean(pData[:,1])
q = pTotal - pInf
d = .15875 / 24

pAtmo = 1005.7 *100
gasR = 287.05
T = 18.6 + 273.15
atmoRho = pAtmo / (gasR*T)
viscocity = 1.87*(10**-5)


pCoeff_list, F_list, D_list, L_list,  = [], [], [], []
arcLengthArray = np.zeros(24)
delTheta = 15 * np.pi/180

for i in range(24):

    P = np.mean(pData[:, i+3])
    pCoeff = (P - pInf) / (pTotal - pInf)
    arcLength= np.deg2rad(15*i)
    theta = i * delTheta
    F = P * delTheta * (d/2)
    D = F*np.cos(theta)
    L = F*np.sin(theta)

    pCoeff_list.append(pCoeff)
    F_list.append(F)
    D_list.append(D)
    L_list.append(L)
    pCoeff_array = np.array(pCoeff_list)
    F_array = np.array(F_list)
    D_array = np.array(D_list)
    L_array = np.array(L_list)
    arcLengthArray[i] = arcLength

V = np.sqrt((2*q) / atmoRho)
print(V)
sumD = -np.sum(D_array)
sumL = np.sum(L_array)
cD = sumD / (q*d) 
arcLengthDegree = np.rad2deg(arcLengthArray)
Re = (atmoRho * V * .15875) / viscocity
print(Re)
plt.figure()
plt.plot(arcLengthDegree, pCoeff_array)
plt.xlabel("Degree")
plt.ylabel("pressure Coefficient")
plt.show()








