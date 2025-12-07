import matplotlib as mpL
import scipy as sp
import scipy.integrate as spi
import numpy as np
import pandas as pd
from Aero_302.standardAtmosphereModel import standardAtmshpereModel
import matplotlib.pyplot as plt

initJ = np.array([45, 0, 0], [0, 45, 0], [0, 0, 70])
initAngular = np.array([0.05, 0.01, 3.5])
thrustF = np.array([0, 10*np.sin(0.5), 0])
