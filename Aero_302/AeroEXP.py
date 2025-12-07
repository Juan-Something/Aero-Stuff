import numpy as np
import scipy.io as sio

lbf_to_N = 4.44822
idx = 9  # 10th entry in MATLAB

# Load EXACT same files the MATLAB script uses
tareData = sio.loadmat("Z:/VS Code/Test Environment/.venv/Aero_302/Lab 3/Lab_3_AoA/2025/TARE_AVG_LiftDrag_neg30_pos20.mat", squeeze_me=True, struct_as_record=False)
TESTdata300 = sio.loadmat("Z:/VS Code/Test Environment/.venv/Aero_302/Lab 3/Lab_3_AoA/2025/S2G2RPM300AOAN10.mat", squeeze_me=True, struct_as_record=False)

tare = tareData['TARE_AVG_LiftDrag_neg30_pos20']

# Helper to coerce to array
def arr(x): return np.asarray(x).squeeze()

# ---- TARE (AoA = -10°) ----
Ptoti = arr(tare.Ptot)[idx]   # could be scalar or small vector
Psi   = arr(tare.Ps)[idx]
Ltare = arr(tare.L)[idx] * lbf_to_N
Dtare = arr(tare.D)[idx] * lbf_to_N

qtare = float(np.mean(Ptoti)) - float(np.mean(Psi))  # scalar

# ---- TEST (300 RPM) ----
P = arr(TESTdata300["P"])
F = arr(TESTdata300["F"])

# Forces: handle both 3x1 and Nx3 shapes
if F.ndim == 1 and F.size == 3:
    Dtotal = float(F[0]) * lbf_to_N
    Ltotal = float(F[2]) * lbf_to_N
elif F.ndim == 2 and F.shape[1] >= 3:
    Dtotal = float(np.mean(F[:, 0])) * lbf_to_N
    Ltotal = float(np.mean(F[:, 2])) * lbf_to_N
else:
    raise ValueError("Unexpected F shape")

# Dynamic pressure from sting ports: columns 0 and 1
if P.ndim == 2 and P.shape[1] >= 2:
    qtest = float(np.mean(P[:, 0])) - float(np.mean(P[:, 1]))
else:
    raise ValueError("Unexpected P shape")

# Sting forces scaled by dynamic pressure ratio
Dsting = Dtare * (qtest / qtare)
Lsting = Ltare * (qtest / qtare)

# Wing-only forces
Dwing = Dtotal - Dsting
Lwing = Ltotal - Lsting

print(Dtare)

print("D_wing_300 [N]:", Dwing)
print("L_wing_300 [N]:", Lwing)
