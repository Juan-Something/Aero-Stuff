import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation

# -----------------------------
# Rotation utilities
# -----------------------------
def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def Ry(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rx(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def euler_zyx_to_R(psi, theta, phi):
    return Rz(psi) @ Ry(theta) @ Rx(phi)

# -----------------------------
# Rigid-body dynamics
# -----------------------------
J = np.array([[17., -3.,  2.],
              [-3., 20., -4.],
              [ 2., -4., 15.]])
Jinv = np.linalg.inv(J)
w0 = np.array([0.01, -0.1, 0.05])
q0 = np.array([1., 0., 0., 0.])

def qdot(q, w):
    q0s, q1, q2, q3 = q
    wx, wy, wz = w
    Omega = np.array([[0.,-wx,-wy,-wz],
                      [wx, 0., wz,-wy],
                      [wy,-wz, 0., wx],
                      [wz, wy,-wx, 0.]])
    return 0.5 * Omega @ q

def euler_from_quat(q):
    q0s, q1, q2, q3 = q
    R = np.array([[1-2*(q2*q2+q3*q3), 2*(q1*q2 - q0s*q3), 2*(q1*q3 + q0s*q2)],
                  [2*(q1*q2 + q0s*q3), 1-2*(q1*q1+q3*q3), 2*(q2*q3 - q0s*q1)],
                  [2*(q1*q3 - q0s*q2), 2*(q2*q3 + q0s*q1), 1-2*(q1*q1+q2*q2)]])
    theta = np.arcsin(np.clip(-R[2,0], -1.0, 1.0))
    phi = np.arctan2(R[2,1], R[2,2])
    psi = np.arctan2(R[1,0], R[0,0])
    return np.array([psi, theta, phi])

def rhs(t, x, torque_func):
    w, q = x[:3], x[3:]
    tau = torque_func(t)
    h = J @ w
    wdot = Jinv @ (tau - np.cross(w, h))
    return np.hstack((wdot, qdot(q, w)))

def normalize_quat(q):
    n = np.linalg.norm(q)
    return q / n if n != 0 else np.array([1.,0.,0.,0.])

def integrate_case(torque_func, T=20.0, dt=0.01):
    t_eval = np.linspace(0, T, int(T/dt)+1)
    x0 = np.hstack((w0, q0))
    sol = solve_ivp(lambda t, x: rhs(t, x, torque_func),
                    [0, T], x0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    w = sol.y[0:3].T
    q = np.array([normalize_quat(qi) for qi in sol.y[3:7].T])
    eul = np.array([euler_from_quat(qi) for qi in q])
    return sol.t, w, q, eul

# -----------------------------
# Cases A–C
# -----------------------------
vals, _ = np.linalg.eigh(J)
print("A. Principal moments (kg·m²):", np.sort(vals))

# Case B: no torque
zero_torque = lambda t: np.zeros(3)
tB, wB, qB, eulB = integrate_case(zero_torque)

# Case C: constant torque [1,-1, 0]
const_torque = lambda t: np.array([1., -1., 0.])
tC, wC, qC, eulC = integrate_case(const_torque)

# -----------------------------
# Optional: visualize Case B or C motion
# -----------------------------
def rb_motion_sim(psi, theta, phi, r=0.5, l=1.0, out_path="RBMotionSim.gif"):
    psi, theta, phi = map(np.asarray, (psi, theta, phi))
    N = len(psi)
    z = np.linspace(-l/2, l/2, 20)
    ang = np.linspace(0, 2*np.pi, 24)
    Z, ANG = np.meshgrid(z, ang)
    X, Y = r*np.cos(ANG), r*np.sin(ANG)

    b1, b2, b3 = np.eye(3)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.view_init(elev=50, azim=125)
    ax.grid(True)
    ax.set_facecolor("white")
    ax.set_autoscale_on(False)

    # inertial axes
    ax.quiver(0,0,0, 0.8,0,0, color='blue')
    ax.quiver(0,0,0, 0,0.8,0, color='blue')
    ax.quiver(0,0,0, 0,0,0.8, color='blue')

    # initial surface and body axes
    R0 = euler_zyx_to_R(psi[0], theta[0], phi[0])
    Xb = R0[0,0]*X + R0[0,1]*Y + R0[0,2]*Z
    Yb = R0[1,0]*X + R0[1,1]*Y + R0[1,2]*Z
    Zb = R0[2,0]*X + R0[2,1]*Y + R0[2,2]*Z
    surf = [ax.plot_surface(Xb, Yb, Zb, color='gray', alpha=0.6, edgecolor='none')]
    bq = [ax.quiver(0,0,0,*R0[:,i],color='red') for i in range(3)]

    def set_body(R):
        # update body axes
        for q in bq: q.remove()
        for i in range(3):
            bq[i] = ax.quiver(0,0,0, *R[:,i], color='red')
        # update surface
        surf[0].remove()
        Xb = R[0,0]*X + R[0,1]*Y + R[0,2]*Z
        Yb = R[1,0]*X + R[1,1]*Y + R[1,2]*Z
        Zb = R[2,0]*X + R[2,1]*Y + R[2,2]*Z
        surf[0] = ax.plot_surface(Xb, Yb, Zb, color='gray', alpha=0.6, edgecolor='none')
        return bq[0], bq[1], bq[2], surf[0]

    def animate(k):
        R = euler_zyx_to_R(psi[k], theta[k], phi[k])
        return set_body(R)

    anim = animation.FuncAnimation(fig, animate, frames=N, interval=20, blit=False)
    anim.save(out_path, writer=animation.PillowWriter(fps=30))
    plt.close(fig)
    return out_path

# Example: animate Case B (set to eulC for Case C)

gif_path = rb_motion_sim(eulC[:,0], eulC[:,1], eulC[:,2], out_path="RBMotionSim_caseC.gif")
print("Animation saved to:", gif_path)
