import numpy as np
from numpy.linalg import inv, norm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Aero_320.RBSim.sim import rb_motion_sim


# ---------- Inertia ----------
J = np.array([[17., -3.,  2.],
              [-3., 20., -4.],
              [ 2., -4., 15.]])
Jinv = inv(J)

# ---------- Torques ----------
def torqueZero(t):  return np.zeros(3)
def torqueConst(t): return np.array([1.0, -1.0, 0.0])

# ---------- ZYX Euler kinematics ----------
def eulerRatesZYX(phi, theta, psi, w):
    wx, wy, wz = w
    sPh, cPh = np.sin(phi),  np.cos(phi)
    sTh, cTh = np.sin(theta), np.cos(theta)
    eps = 1e-9
    if abs(cTh) < eps:
        cTh = eps * np.sign(cTh if cTh != 0 else 1.0)
    tanTh = sTh / cTh
    secTh = 1.0 / cTh
    Phdot = wx + sPh * tanTh * wy + cPh * tanTh * wz
    Thdot = cPh * wy - sPh * wz
    Psdot = sPh * secTh * wy + cPh * secTh * wz
    return np.array([Phdot, Thdot, Psdot])

# ---------- Quaternion kinematics ----------
# q = [q0,q1,q2,q3] scalar-first
def qdotFrom_w(q, w):
    wx, wy, wz = w
    Omega = np.array([[0.0, -wx, -wy, -wz],
                      [wx,  0.0,  wz, -wy],
                      [wy, -wz,  0.0,  wx],
                      [wz,  wy, -wx,  0.0]])
    return 0.5 * (Omega @ q)

def qnormalize(q):
    n = norm(q)
    return q if n == 0 else q / n

# ---------- Dynamics ----------
def wdot_of(w, T):  # shared rigid-body dynamics
    return Jinv @ (T - np.cross(w, J @ w))

# State y_e = [wx, wy, wz, φ, θ, ψ]
def rhsEuler(t, y):
    w = y[0:3]
    phi, theta, psi = y[3], y[4], y[5]
    T = torqueFun(t)
    return np.hstack((wdot_of(w, T), eulerRatesZYX(phi, theta, psi, w)))

# State y_q = [wx, wy, wz, q0, q1, q2, q3]
def rhsQuat(t, y):
    w = y[0:3]
    q = qnormalize(y[3:7])
    T = torqueFun(t)
    return np.hstack((wdot_of(w, T), qdotFrom_w(q, w)))

def integrateEuler(torque, t_end=20.0,
                    w0=np.array([0.01, -0.10, 0.05]),
                    e0=(0.0, 0.0, 0.0)):
    global torqueFun
    torqueFun = torque
    y0 = np.hstack((w0, np.array(e0)))
    sol = solve_ivp(rhsEuler, [0.0, t_end], y0, method="RK45",
                    rtol=1e-8, atol=1e-12)
    t = sol.t
    w = sol.y[0:3, :].T
    e = sol.y[3:6, :].T
    return t, w, e

def integrateQuat(torque, t_end=20.0,
                   w0=np.array([0.01, -0.10, 0.05]),
                   q0=np.array([1.0, 0.0, 0.0, 0.0])):
    global torqueFun
    torqueFun = torque
    y0 = np.hstack((w0, qnormalize(q0)))
    sol = solve_ivp(rhsQuat, [0.0, t_end], y0, method="RK45",
                    rtol=1e-8, atol=1e-12)
    t = sol.t
    w = sol.y[0:3, :].T
    q = sol.y[3:7, :].T
    q = np.array([qnormalize(qi) for qi in q])  # control drift
    return t, w, q

# ---------- Run: zero and constant torque ----------
t0_e, w0_e, e0 = integrateEuler(torqueZero)
t0_q, w0_q, q0 = integrateQuat(torqueZero)

t1_e, w1_e, e1 = integrateEuler(torqueConst)
t1_q, w1_q, q1 = integrateQuat(torqueConst)

# ---------- Plots ----------
# Angular velocity (Euler)
for label, t, w in [("Zero torque (Euler)", t0_e, w0_e),
                    ("Const torque (Euler)", t1_e, w1_e)]:
    plt.figure()
    plt.plot(t, w[:,0], label="ωx")
    plt.plot(t, w[:,1], label="ωy")
    plt.plot(t, w[:,2], label="ωz")
    plt.title(f"Angular velocity — {label}")
    plt.xlabel("Time [s]"); plt.ylabel("ω [rad/s]"); plt.legend()

# Euler angles 
for label, t, e in [("Zero torque", t0_e, e0), ("Const torque", t1_e, e1)]:
    plt.figure()
    plt.plot(t, e[:,0], label="φ")
    plt.plot(t, e[:,1], label="θ")
    plt.plot(t, e[:,2], label="ψ")
    plt.title(f"Euler angles (ZYX) — {label}")
    plt.xlabel("Time [s]"); plt.ylabel("Angle [rad]"); plt.legend()

# Quaternion
for label, t, q in [("Zero torque", t0_q, q0), ("Const torque", t1_q, q1)]:
    plt.figure()
    plt.plot(t, q[:,0], label="q0")
    plt.plot(t, q[:,1], label="q1")
    plt.plot(t, q[:,2], label="q2")
    plt.plot(t, q[:,3], label="q3")
    plt.title(f"Quaternion — {label}")
    plt.xlabel("Time [s]"); plt.ylabel("component"); plt.legend()
plt.show()

fig = rb_motion_sim(e[:,2], e[:,1], e[:,0], save_path=None)
plt.show()

# -------------Q2 -------------------

def kineticEnergy(mass, v_vec, omega_b, J_b):
    """Return total kinetic energy [J].
    mass: kg
    v_vec: 3-vector linear velocity in any inertial frame [m/s]
    omega_b: 3-vector angular rate in body frame [rad/s]
    J_b: 3x3 inertia in body frame [kg·m^2]
    """
    v = np.asarray(v_vec).reshape(3)
    w = np.asarray(omega_b).reshape(3)
    J = np.asarray(J_b).reshape(3,3)
    T_trans = 0.5 * mass * float(v @ v)
    T_rot   = 0.5 * float(w @ (J @ w))
    return T_trans + T_rot, T_trans, T_rot

m = 17474.0  # kg
v_ECI = np.array([20.0, 105.0, -10.0])  # m/s
omega_b = np.array([0.5, -0.1, 0.1])    # rad/s
J_b = 1e6 * np.array([[2.44, 0.0, -1.2],
                      [0.0, 27.0, 0.0],
                      [-1.2, 0.0, 30.0]])  # kg·m^2

T_total, T_trans, T_rot = kineticEnergy(m, v_ECI, omega_b, J_b)
print(f"T_total = {T_total:.0f} J")
print(f"  translational = {T_trans:.0f} J")
print(f"  rotational    = {T_rot:.0f} J")
