import matplotlib as mpL
import scipy as sp
import scipy.integrate as spi
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

surfaceArea = 0.0804

tareData = sio.loadmat("Z:/VS Code/Test Environment/.venv/Aero_302/Lab 3/Lab_3_AoA/2025/TARE_AVG_LiftDrag_neg30_pos20.mat", squeeze_me=True, struct_as_record=False)
tare = tareData['TARE_AVG_LiftDrag_neg30_pos20']  
testData1 = sio.loadmat("Z:/VS Code/Test Environment/.venv/Aero_302/Lab 3/Lab_3_AoA/2025/S5G1RPM300AOAN10.mat")   
testData = sio.loadmat("Z:/VS Code/Test Environment/.venv/Aero_302/Lab 3/Lab_3_AoA/2025/S2G2RPM300AOAN10.mat")     

tarePTot = tare.Ptot[10]
tareP    = tare.Ps[10]         
tareD    = tare.D[10]
tareL    = tare.L[10]
tareAoA  = tare.AoA

testP = testData["P"]
testF = testData["F"]


lbfToNewton = 4.44822
testX = lbfToNewton*np.mean(testF[:,0])
testy = lbfToNewton*np.mean(testF[:,1])
testZ = lbfToNewton*np.mean(testF[:,2])
tareDNewton = tareD * lbfToNewton
tareLNewton = tareL * lbfToNewton

tareDyn = tarePTot - tareP
stingDyn = np.mean(testP[:,0]) - np.mean(testP[:,1])
# testDyn = 

stingL = (tareLNewton / tareDyn) * stingDyn
stingD = (tareDNewton / tareDyn) * stingDyn
testD = testX - stingD
testL = testZ - stingL

print(f"Drag Force: {testD: .4f}")
print(f"Lift Force: {testL: .4f}")

cL = testL / (surfaceArea * stingL)
cD = testD / (surfaceArea * stingD)

print(f"Drag Coefficient: {cL: .4f}")
print(f"Lift Coefficient: {cD: .4f}")