import numpy as np

import mpr_visc_theory as mpr

# length of cell
Lx = mpr.sqrt(40)
# breadth of cell
Ly = Lx / 2


# declearing fluid properties
cryo = mpr.electrolyte(
    name="cryolite", density=2130, kinematic_viscosity=8.8e-7, electric_conductivity=210
)

alum = mpr.metal(
    name="aluminium",
    density=2330,
    kinematic_viscosity=4.4e-7,
    electric_conductivity=3.33e6,
)

# surface tension
gamma_int = mpr.surface_tension(0)

# variable to store dimensions
dim = mpr.geometry(length=Lx, width=Ly, electrolyte_height=0.05, metal_height=0.25)

# external current and magnetic field strength
power = mpr.power(const_current=300e3, const_mag=0.001)


mpr.parameter_check(cryo, alum, dim)


# find critical wave number and critical onset value
crit_waveNo, beta_crit = mpr.calculate_instability_onset(
    cryo, alum, gamma_int, dim, wavenumber_limit=6
)

print(f"Critical wave number: {crit_waveNo}")
print(f"Instability onset: {beta_crit:.3f}")

# calculated wave mode
wave1 = mpr.wavemode1(crit_waveNo[0], crit_waveNo[1])
wave2 = mpr.wavemode1(crit_waveNo[2], crit_waveNo[3])


# wave-number dependent viscous damping rates
damp1, damp2 = mpr.calculate_viscous_damping(
    electrolyte=cryo,
    metal=alum,
    surface_tension=gamma_int,
    geometry=dim,
    wavemode1=wave1,
    wavemode2=wave2,
)

print(
    f"Viscous damping rate of wave mode {wave1} and {wave2} is:",
    f"{damp1:.4f}, {damp2:.4f}",
)

# stability criterion
beta = mpr.calculate_SeleParameter(cryo, alum, gamma_int, dim, power, wave1, wave2)

# viscous growth rate
gr = mpr.calculate_GrowthRate(cryo, alum, gamma_int, dim, power, wave1, wave2)

print(
    f"For Sele parameter = {beta:.3f}, the growth rate of {wave1} and {wave2} is:",
    f"{gr:.4f}",
)
