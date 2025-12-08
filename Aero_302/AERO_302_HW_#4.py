import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# time grid
t = np.linspace(0, 10, 2000)

# velocity u(t)
u = lambda t: 10*np.cos(10*t)*np.exp(-t)
u_vals = u(t)

# integrate to get x(t)
x_vals = spi.cumulative_trapezoid(u_vals, t, initial=0.0)

y0 = 10.0  # cm (or m, just be consistent)

# --- streamline in x–y plane (y = const) ---
x_stream = np.linspace(-1, 1, 200)
y_stream = y0 * np.ones_like(x_stream)

plt.figure()
plt.title("Streamline in x–y plane")
plt.plot(x_stream, y_stream)
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

# --- pathline in x–y plane ---
plt.figure()
plt.title("Pathline in x–y plane")
plt.plot(x_vals, y0*np.ones_like(x_vals))
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.grid()

# --- pathline in x–t plane ---
plt.figure()
plt.title("Pathline in x–t plane")
plt.plot(t, x_vals)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid()

plt.show()
