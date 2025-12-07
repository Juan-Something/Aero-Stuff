import numpy as np
import matplotlib.pyplot as plt
import os, re
import scipy.io as sio

#CVA
CT = np.linspace(0.0, 1.2, 601)
aThrust  = (-1.0 + np.sqrt(1.0 + CT)) / 2.0           
eta = 1.0 / (1.0 + aThrust)                           
plt.figure()
plt.plot(CT, eta)
plt.xlabel("Thrust coefficient $C_T$")
plt.ylabel("Ideal propulsive efficiency $\\eta$")
plt.title("Actuator-disc: $\\eta$ vs $C_T$")
plt.grid(True)
plt.tight_layout()
plt.show()


# ISF1

"""
Assumptions: perfect gas, 1-D flow, adiabatic and reversible
"""
def isentropicRatios(mach, gammaVal=1.4):
    mach = np.asarray(mach, dtype=float)
    tau = 1.0 + 0.5 * (gammaVal - 1.0) * mach**2
    t0OverT = tau
    p0OverP = tau ** (gammaVal / (gammaVal - 1.0))
    rho0OverRho = tau ** (1.0 / (gammaVal - 1.0))
    return t0OverT, p0OverP, rho0OverRho


machRange = np.linspace(0.0, 3.0, 100)
gammaVal = 1.4

tOverT0, pOverP0, rhoOverRho0 = isentropicRatios(machRange, gammaVal)

plt.figure()
ax1 = plt.subplot(311)
ax1.plot(machRange, 1.0 / tOverT0, label='T/T0')
ax1.set_title(f'Isentropic ratios vs Mach (γ={gammaVal})')
ax1.set_ylabel('T/T0')
ax1.grid(True)
ax1.legend(loc='best')

ax2 = plt.subplot(312)
ax2.plot(machRange, 1.0 / pOverP0, label='p/p0')
ax2.set_ylabel('p/p0')
ax2.grid(True)
ax2.legend(loc='best')

ax3 = plt.subplot(313)
ax3.plot(machRange, 1.0 / rhoOverRho0, label=r'$\rho/\rho_0$')
ax3.set_xlabel('Mach number M')
ax3.set_ylabel(r'$\rho/\rho_0$')
ax3.grid(True)
ax3.legend(loc='best')

plt.tight_layout()
plt.show()

# Part 4
"""
data = r"Z:\VS Code\Test Environment\.venv\Aero_302\HW2_Data\S05G1D4RPM500.mat"
rho = 1.2250
P = data["P"]
gamma = 1.4
R = 287
a = np.sqrt(gamma * R * 273.15)
pTot = np.mean(P[:,0])
"""

# ISF2
"""
Assumptions: standard atmosphere, incompressible, isentropic 
"""
gammaConst = 1.4
rGas = 287.05       # J/(kg·K)
gAccel = 9.80665    # m/s^2
seaLevelT = 288.15  # K
seaLevelP = 101325.0  # Pa
lapseRate = 0.0065  # K/m, troposphere

def standardAtmosphereModel(hKm):
    """ISA 0–11 km. Returns scalars: T [K], P [Pa], rho [kg/m^3], aSpeed [m/s]."""
    hM = float(hKm) * 1000.0
    tKelvin = seaLevelT - lapseRate * hM
    pPa = seaLevelP * (tKelvin / seaLevelT) ** (gAccel / (-lapseRate * rGas))
    rhoKgM3 = pPa / (rGas * tKelvin)
    aSpeed = np.sqrt(gammaConst * rGas * tKelvin)
    return tKelvin, pPa, rhoKgM3, aSpeed


# Atmosphere at 10 km
tKelvin, pPa, rhoKgM3, aSpeed = standardAtmosphereModel(10.0)

# Mach and velocity
mach = np.linspace(0.0, 1.0, 100)
velocity = mach * aSpeed

# Bernoulli (incompressible)
qBernoulli = 0.5 * rhoKgM3 * velocity**2
p0Bernoulli = pPa + qBernoulli
t0Bernoulli = np.full_like(mach, tKelvin)

# Isentropic (compressible)
t0OverT, p0OverP, _ = isentropicRatios(mach, gammaConst)
p0Isentropic = pPa * p0OverP
t0Isentropic = tKelvin * t0OverT
qIsentropic = p0Isentropic - pPa

p_static = np.full_like(mach, pPa)          # Pa
T_static = np.full_like(mach, tKelvin)      # K
rho_static = np.full_like(mach, rhoKgM3)    # kg/m^3

t0OverT, p0OverP, rho0OverRho = isentropicRatios(mach, gammaConst)
rho0Isentropic = rhoKgM3 * rho0OverRho

rho0Bernoulli = np.full_like(mach, rhoKgM3)

# Plot densities
plt.figure()
plt.plot(mach, rho_static, label=r'$\rho$ static')
plt.plot(mach, rho0Bernoulli, label=r'$\rho_0$ Bernoulli')
plt.plot(mach, rho0Isentropic, '--', label=r'$\rho_0$ Isentropic')
plt.xlabel('Mach')
plt.ylabel(r'Density [kg/m$^3$]')
plt.title('Static and total density vs Mach at 10 km')
plt.grid(True)
plt.legend()
plt.show()

# Pressure: static and totals vs Mach
plt.figure()
plt.plot(mach, p_static/1000.0, label="p static")
plt.plot(mach, p0Bernoulli/1000.0, label="p0 Bernoulli")
plt.plot(mach, p0Isentropic/1000.0, "--", label="p0 Isentropic")
plt.xlabel("Mach")
plt.ylabel("Pressure [kPa]")
plt.title("Static and total pressure vs Mach at 10 km")
plt.grid(True); plt.legend(); plt.show()

# Temperature: static and totals vs Mach
plt.figure()
plt.plot(mach, T_static, label="T static")
plt.plot(mach, t0Bernoulli, label="T0 Bernoulli")
plt.plot(mach, t0Isentropic, "--", label="T0 Isentropic")
plt.xlabel("Mach")
plt.ylabel("Temperature [K]")
plt.title("Static and total temperature vs Mach at 10 km")
plt.grid(True); plt.legend(); plt.show()



# CM1
"""
Assumptions: Incompressible, linear velocity
"""
bulbDiameter = 0.15  # m
conduitDiameter = 0.10  # m
bulbVolume = np.pi * bulbDiameter**3 / 6.0

conduitLength = 10.0  # m
conduitArea = np.pi * conduitDiameter**2 / 4.0
initialTime = 0.75  # s

waterRho = 1000.0  # kg/m^3
gLocal = 9.80665   # m/s^2
pAtm = 101325.0    # Pa
waterGasConstant = 461.5  # J/(kg·K)  (for water vapor)

hydroP = pAtm + waterRho * gLocal * conduitLength

# From steam table, at hydroP ≈ 2e5 Pa, saturated temp of steam ≈ 120 C
boilTempC = 120.2
steamTempK = (boilTempC + 80.0) + 273.15
rho0Steam = hydroP / (waterGasConstant * steamTempK)

print(f"P at 10 m: {hydroP/1e3:.1f} kPa")
print(f"T_boil around {boilTempC:.1f} °C, T_steam = {steamTempK:.2f} K")
print(f"rho_s0 = {rho0Steam:.4f} kg/m^3")

def flowVolume(velocity):
    vDot = conduitArea * velocity
    mDot = waterRho * vDot
    return vDot, mDot

inletVelocity = 15.0  # m/s
vDot, mDot = flowVolume(inletVelocity)

print(f"u_i={inletVelocity:4.1f} m/s -> Vdot={vDot:9.6f} m^3/s, mdot={mDot:7.2f} kg/s")

# Part 4
xReq = (1.0 / 0.003 - 1.0) * bulbVolume / conduitArea
exitVelocity = 15.0  # m/s
x0 = 0.01            # m, small nonzero interface height at t0

deltaT = 2 * xReq / exitVelocity #(conduitLength / exitVelocity) * np.log(xReq / x0)

tVec = np.linspace(initialTime, initialTime + deltaT, 400)
xInterface = x0 * np.exp((exitVelocity / conduitLength) * (tVec - initialTime))
rhoS = rho0Steam * bulbVolume / (bulbVolume + conduitArea * xInterface)

print(f"x_req = {xReq:.3f} m, Δt={deltaT:.1f} s")
print(f"rho_s(t_end)/rho_s0 = {rhoS[-1]/rho0Steam:.5f}")

plt.figure()
plt.plot(tVec, rhoS)
plt.axhline(0.003 * rho0Steam, linestyle="--")
plt.xlabel("time [s]")
plt.ylabel("steam density [kg/m^3]")
plt.title("Steam density vs time (exponential interface motion)")
plt.grid(True)
plt.show()

#CM2
"""
Assumptions: steady flow, constant density, 
"""
dNozzle, UNozzle = 0.05, 100.0
xNozzle = 10*dNozzle
uMaxNozzle = 5*dNozzle*UNozzle/xNozzle
rNozzle = np.linspace(0, 4*dNozzle, 400)
uNozzle = uMaxNozzle*np.exp(-50*(rNozzle**2)/(xNozzle**2))
plt.plot(rNozzle/dNozzle, uNozzle)
plt.xlabel('r/d')
plt.ylabel('u [m/s]')
plt.title('x=10d')
plt.grid(True)
plt.show()