import os
import numpy as np
from numpy import pi, sin, cos, tan, sinh, cosh, tanh, sqrt
import cmath as cmth
from dataclasses import dataclass
from itertools import product

np.set_printoptions(legacy="1.25")


##### PHYSICAL PARAMETERS ##############
g = 9.81


##### MATH FUNCTION ##############
def coth(x):
    return 1 / np.tanh(x)


def csqrt(x):
    return cmth.sqrt(x)


###################################


## DEFINING CLASSES ############
@dataclass
class electrolyte:
    name: str
    density: float  # kg/m^3
    kinematic_viscosity: float  # m2/s (Kinematic Viscosity)
    electric_conductivity: float  # S/m electrical conductivity


@dataclass
class metal:
    name: str
    density: float  # kg/m^3
    kinematic_viscosity: float  # m2/s (Kinematic Viscosity)
    electric_conductivity: float  # S/m electrical conductivity


@dataclass
class surface_tension:
    gamma: float  # N/m, value of surface tension


@dataclass
class geometry:
    # all values in m
    length: float  # length of geometry
    width: float  # width of geometry
    electrolyte_height: float  # layer height of electrolyte
    metal_height: float  # layer height of metal


@dataclass
class power:
    const_current: float  # A, constant vertical current
    const_mag: float  # T, constant vertical magnetic field


@dataclass
class wavemode1:
    # wave mode (m1, n1) as in the manuscript
    m: int
    n: int


@dataclass
class wavemode2:
    # wave mode (m2, n2) as in the manuscript
    m: int
    n: int


#########################################################
# %%
####### RAW FUNCTIONS ######################


def deltaFunc(x):
    val = 1 if x == 0 else 2
    return val


# wave number function
def kmn(m, n, Lx, Ly):
    """
    Wave number function. Equation (12).
    """
    
    if m <= 0 and n <= 0:
        raise ValueError("Incorrect wave modes")
    else:
        return pi * sqrt(pow(m / Lx, 2) + pow(n / Ly, 2))


# wave frequency
def wmn(rho1, rho2, gamma, k, h1, h2):
    """
    Gravity-capillary wave frequency. Equation (13).
    """
    num = (rho2 - rho1) * g * k + gamma * k**3
    den = rho1 * coth(k * h1) + rho2 * coth(k * h2)
    return sqrt(num / den)


# conductivity jump parameter
def Lambda(rho1, rho2, sig1, sig2, h1, h2, k):
    """
    conductivity jump parameter. Equation (15).
    """
    num = pow(sig1, -1) + pow(sig2, -1)
    den = pow(sig1, -1) * tanh(k * h1) + pow(sig2, -1) * coth(k * h2)
    return num / den


# viscous dissipation due to the bounding walls
def visc_damp_wall(rho1, rho2, nu1, nu2, gamma, m, n, Lx, Ly, h1, h2):
    """
    damping rate due to viscous dissipation due to the bounding walls. Equation (36).
    """
    k = kmn(m, n, Lx, Ly)
    w = wmn(rho1, rho2, gamma, k, h1, h2)

    summation = 0  # summing up two liquid layers
    for i in [1, 2]:
        rho = locals()["rho" + str(i)]
        nu = locals()["nu" + str(i)]
        h = locals()["h" + str(i)]

        common_factor = (
            pow(2, -0.5)
            * (rho * sqrt(w * nu) / (k * Lx * Ly))
            / ((rho1 * coth(k * h1) + rho2 * coth(k * h2)))
        )

        base_faces = Lx * Ly * pow(k, 2) / (2 * pow(sinh(k * h), 2))
        sidewall = (
            deltaFunc(m)
            * h
            * (pow(n * pi, 2) - pow(k * Ly, 2))
            / (2 * Ly * pow(sinh(k * h), 2))
            + deltaFunc(m)
            * (pow(n * pi, 2) + pow(k * Ly, 2))
            / (2 * Ly * k * tanh(k * h))
            + deltaFunc(n)
            * h
            * (pow(m * pi, 2) - pow(k * Lx, 2))
            / (2 * Lx * pow(sinh(k * h), 2))
            + deltaFunc(n)
            * (pow(m * pi, 2) + pow(k * Lx, 2))
            / (2 * Lx * k * tanh(k * h))
        )
        summation += common_factor * (base_faces + sidewall)
    return summation


# viscous dissipation due to the liquid-liquid interface
def visc_damp_interface(rho1, rho2, nu1, nu2, gamma, m, n, Lx, Ly, h1, h2):
    """
    damping rate due to viscous dissipation due to the liquid-liquid interface. Equation (37).
    """
    
    k = kmn(m, n, Lx, Ly)
    w = wmn(rho1, rho2, gamma, k, h1, h2)

    fac = k * sqrt(w) / (sqrt(8))
    num = pow(coth(k * h1) + coth(k * h2), 2)
    den = (pow(rho1 * sqrt(nu1), -1) + pow(rho2 * sqrt(nu2), -1)) * (
        rho1 * coth(k * h1) + rho2 * coth(k * h2)
    )
    return fac * num / den


# viscous dissipation due to bulk irrotational stresses
def visc_damp_irrotational(rho1, rho2, nu1, nu2, gamma, m, n, Lx, Ly, h1, h2):
    """
    damping rate due to viscous dissipation due to irrotational stresses. Equation (38).
    """
    
    k = kmn(m, n, Lx, Ly)

    fac = (
        2
        * pow(k, 2)
        / (
            (pow(rho1 * sqrt(nu1), -1) + pow(rho2 * sqrt(nu2), -1))
            * (rho1 * coth(k * h1) + rho2 * coth(k * h2))
        )
    )
    term1 = (rho2 * sqrt(nu1 * nu2) + rho1 * nu1) / (rho2 * sqrt(nu2) * tanh(k * h1))
    term2 = (rho1 * sqrt(nu1 * nu2) + rho2 * nu2) / (rho1 * sqrt(nu1) * tanh(k * h2))
    term3 = (coth(k * h1) + coth(k * h2)) * (sqrt(nu1) + sqrt(nu2))
    return fac * (term1 + term2 - term3)


def visc_damp_total(rho1, rho2, nu1, nu2, gamma, m, n, Lx, Ly, h1, h2):
    """
    total damping rate. Equation (35).
    """
    
    wall = visc_damp_wall(rho1, rho2, nu1, nu2, gamma, m, n, Lx, Ly, h1, h2)
    sidewall = visc_damp_interface(rho1, rho2, nu1, nu2, gamma, m, n, Lx, Ly, h1, h2)
    bulk = visc_damp_irrotational(rho1, rho2, nu1, nu2, gamma, m, n, Lx, Ly, h1, h2)
    return wall + sidewall + bulk


# form function for the wave modes
def ThetaFunc(m1, n1, m2, n2):
    """
    Theta = f(m,n, m',n'). Equation (23a).
    """
    
    fac = sqrt(deltaFunc(m1 * m2) * deltaFunc(n1 * n2))
    num = (
        (pow(-1, m1 + m2) - 1)
        * (pow(-1, n1 + n2) - 1)
        * (pow(m1 * n2, 2) - pow(m2 * n1, 2))
    )
    den = (pow(m1, 2) - pow(m2, 2)) * (pow(n1, 2) - pow(n2, 2))
    return fac * num / den


# Terms which arise when we remove the shallow water approximation
def T_func(rho1, rho2, sig1, sig2, h1, h2, k1, k2):
    """
    non-shallow layer height dependencies. Equation (23d, 23e).
    """
    
    Lambda1 = Lambda(rho1, rho2, sig1, sig2, h1, h2, k1)
    Lambda2 = Lambda(rho1, rho2, sig1, sig2, h1, h2, k2)

    if k1 != k2:

        fac1 = Lambda1 / (k1 * (k1**2 - k2**2))
        term1 = (
            k1 / tanh(k2 * h2)
            - k2 / tanh(k1 * h2)
            + k2 / (sinh(k1 * h1) * cosh(k2 * h1))
            - k2 / tanh(k1 * h1)
            + k1 * tanh(k2 * h1)
        )
        T_norm = fac1 * term1

        fac2 = Lambda2 / (k2 * (k2**2 - k1**2))
        term2 = (
            k2 / tanh(k1 * h2)
            - k1 / tanh(k2 * h2)
            + k1 / (sinh(k2 * h1) * cosh(k1 * h1))
            - k1 / tanh(k2 * h1)
            + k2 * tanh(k1 * h1)
        )
        T_dash = fac2 * term2

    else:
        T_norm = (Lambda1 / (4 * pow(k1, 2) * pow(np.sinh(k1 * h2), 2))) * (
            np.sinh(k1 * h1 + 2 * k1 * h2) / np.cosh(k1 * h1)
            + 2 * k1 * h2
            - np.tanh(k1 * h1)
        )
        T_dash = T_norm

    return T_norm, T_dash


# formula for growth rate
def growth_rate(rho1, rho2, nu1, nu2, sig1, sig2, gamma,
    m1, n1, m2, n2,
    Lx, Ly, h1, h2,
    I0, Bz, damp1, damp2,
    ):
    """
    viscous growth rate. Equation (41).
    """
    
    k1 = kmn(m1, n1, Lx, Ly)
    w1 = wmn(rho1, rho2, gamma, k1, h1, h2)
    k2 = kmn(m2, n2, Lx, Ly)
    w2 = wmn(rho1, rho2, gamma, k2, h1, h2)

    w_avg = np.mean([w1, w2])
    w_del = (w1 - w2) / 2

    J0 = I0 / (Lx * Ly)

    Theta = ThetaFunc(m1, n1, m2, n2)

    T1, T2 = T_func(rho1, rho2, sig1, sig2, h1, h2, k1, k2)

    U1 = Lx * Ly * (gamma * k1**2 + (rho2 - rho1) * g)
    U2 = Lx * Ly * (gamma * k2**2 + (rho2 - rho1) * g)

    damp_avg = np.mean([damp1, damp2])
    damp_del = (damp1 - damp2) / 2

    term1 = (
        pow(J0 * Bz * Theta, 2) * (pow(w_avg, 2) - pow(w_del, 2)) * T1 * T2 / (U1 * U2)
    )
    term2 = pow(w_del * 1j - damp_del, 2)

    return csqrt(term1 + term2) - damp_avg


# formula to calculate the instability onset
def betaCrit_func(rho1, rho2, nu1, nu2, sig1, sig2, gamma,
    m1, n1, m2, n2,
    Lx, Ly, h1, h2,
    damp1, damp2,
    ):
    """
    stability criterion for the MPR instability. Equation (42).
    """
    
    k1 = kmn(m1, n1, Lx, Ly)
    w1 = wmn(rho1, rho2, gamma, k1, h1, h2)
    k2 = kmn(m2, n2, Lx, Ly)
    w2 = wmn(rho1, rho2, gamma, k2, h1, h2)

    w_avg = np.mean([w1, w2])
    w_del = (w1 - w2) / 2

    damp_avg = np.mean([damp1, damp2])
    damp_del = (damp1 - damp2) / 2

    Theta = ThetaFunc(m1, n1, m2, n2)
    T1, T2 = T_func(rho1, rho2, sig1, sig2, h1, h2, k1, k2)

    fac = pow(Lx * Ly, 2) / (h1 * h2 * np.abs(Theta))
    num = damp_avg**2 - pow(w_del * 1j - damp_del, 2)
    den = (pow(w_avg, 2) - pow(w_del, 2)) * T1 * T2

    return np.real(fac * csqrt(num / den))


# %%


# parameter check
def parameter_check(electrolyte, metal, geometry):
    """
    Validate physical and geometrical parameters of a two-layer system.
    """

    # initiate variables
    rho1 = electrolyte.density
    rho2 = metal.density
    nu1 = electrolyte.kinematic_viscosity
    nu2 = metal.kinematic_viscosity
    sig1 = electrolyte.electric_conductivity
    sig2 = metal.electric_conductivity

    gamma = surface_tension.gamma

    m1, n1 = wavemode1.m, wavemode1.n
    m2, n2 = wavemode2.m, wavemode2.n
    Lx, Ly = geometry.length, geometry.width
    h1, h2 = geometry.electrolyte_height, geometry.metal_height
    
    k1, k2 = kmn(m1, n1, Lx, Ly), kmn(m2, n2, Lx, Ly)
    w1 = wmn(rho1, rho2, gamma, k1, h1, h2)
    w2 = wmn(rho1, rho2, gamma, k2, h1, h2)
    
    #calculating the thickness of the viscous sub layer
    vsl1 = np.sqrt(nu1/w1)
    vsl2 = np.sqrt(nu2/w2)
    
        
    if rho1 <= 0 or rho2 <= 0:
        raise ValueError("Incorrect density values")
    elif nu1 <= 0 or nu2 <= 0:
        raise ValueError("Incorrect kinematic viscosity values")
    elif sig1 <= 0 or sig2 <= 0:
        raise ValueError("Non physical electric conductivity")
    elif Lx <= 0 or Ly <= 0:
        raise ValueError("Incorrect geometry")
    elif h1 <= 0 or h2 <= 0:
        raise ValueError("Incorrect layer heights")
    if max(Lx, Ly) < min(h1, h2):
        print("WARING: Lx and Ly should be layer than the layer height to account.")
    if vsl1/h1 > 0.1:
        print("Warning: viscous sub-layer too thick compared to the layer height. Adjust h1")
    elif vsl2/h2 > 0.1:
        print("Warning: viscous sub-layer too thick compared to the layer height. Adjust h2")
    
    print("parameter check: OK \n")


# %%
####### USER FUNCTIONS ######################


def calculate_viscous_damping(
    electrolyte, metal, surface_tension, geometry, wavemode1, wavemode2
):

    # initiate variables
    rho1 = electrolyte.density
    rho2 = metal.density
    nu1 = electrolyte.kinematic_viscosity
    nu2 = metal.kinematic_viscosity

    gamma = surface_tension.gamma

    m1, n1 = wavemode1.m, wavemode1.n
    m2, n2 = wavemode2.m, wavemode2.n
    Lx, Ly = geometry.length, geometry.width
    h1, h2 = geometry.electrolyte_height, geometry.metal_height

    for i in [1, 2]:
        m, n = locals()["m" + str(i)], locals()["n" + str(i)]
        locals()["mode" + str(i) + "_damp"] = (
            visc_damp_wall(rho1, rho2, nu1, nu2, gamma, m, n, Lx, Ly, h1, h2)
            + visc_damp_interface(rho1, rho2, nu1, nu2, gamma, m, n, Lx, Ly, h1, h2)
            + visc_damp_irrotational(rho1, rho2, nu1, nu2, gamma, m, n, Lx, Ly, h1, h2)
        )

    return locals()["mode" + str(1) + "_damp"], locals()["mode" + str(2) + "_damp"]


# Sele parameter
def calculate_SeleParameter(
    electrolyte, metal, surface_tension, geometry, power, wavemode1, wavemode2
):

    # initiate variables
    rho1 = electrolyte.density
    rho2 = metal.density
    gamma = surface_tension.gamma
    m1, n1 = wavemode1.m, wavemode1.n
    m2, n2 = wavemode2.m, wavemode2.n
    I0, Bz = power.const_current, power.const_mag
    Lx, Ly = geometry.length, geometry.width
    h1, h2 = geometry.electrolyte_height, geometry.metal_height

    k1, k2 = kmn(m1, n1, Lx, Ly), kmn(m2, n2, Lx, Ly)

    den = ((rho2 - rho1) * g + gamma * k1 * k2) * h1 * h2

    return I0 * Bz / den


def calculate_GrowthRate(
    electrolyte, metal, surface_tension, geometry, power, wavemode1, wavemode2
):

    # initiate variables
    rho1 = electrolyte.density
    rho2 = metal.density
    nu1 = electrolyte.kinematic_viscosity
    nu2 = metal.kinematic_viscosity
    sig1 = electrolyte.electric_conductivity
    sig2 = metal.electric_conductivity

    gamma = surface_tension.gamma

    m1, n1 = wavemode1.m, wavemode1.n
    m2, n2 = wavemode2.m, wavemode2.n

    I0, Bz = power.const_current, power.const_mag

    Lx, Ly = geometry.length, geometry.width
    h1, h2 = geometry.electrolyte_height, geometry.metal_height

    damp1 = visc_damp_total(rho1, rho2, nu1, nu2, gamma, m1, n1, Lx, Ly, h1, h2)
    damp2 = visc_damp_total(rho1, rho2, nu1, nu2, gamma, m2, n2, Lx, Ly, h1, h2)

    gr = growth_rate(rho1, rho2, nu1, nu2, sig1, sig2, gamma,
        m1, n1, m2, n2,
        Lx, Ly, h1, h2,
        I0, Bz, damp1, damp2,
    )

    return gr


def calculate_instability_onset(
    electrolyte, metal, surface_tension, geometry, wavenumber_limit
):

    rho1 = electrolyte.density
    rho2 = metal.density
    nu1 = electrolyte.kinematic_viscosity
    nu2 = metal.kinematic_viscosity
    sig1 = electrolyte.electric_conductivity
    sig2 = metal.electric_conductivity

    gamma = surface_tension.gamma

    Lx, Ly = geometry.length, geometry.width
    h1, h2 = geometry.electrolyte_height, geometry.metal_height

    AR = Lx / Ly

    def find_nearest(line_array, alpha):
        idx = (np.abs(line_array - alpha)).argmin()
        return idx

    # function to generate list of all possible wave mode combinations
    def genWaveNo(wavenumber_limit):
        num = np.arange(0, wavenumber_limit, 1)
        comb_set = [p for p in product(num, repeat=4)]

        # cleansing of list to remove unrealistic wave modes
        i = 0
        while i < len(comb_set):
            pi = np.pi
            m1, n1, m2, n2 = comb_set[i]
            cond1 = m1**2 - m2**2
            cond2 = n2**2 - n1**2
            cond3 = pow(m1 * n2, 2) - pow(m2 * n1, 2)
            cond4 = (np.cos(m1 * pi) * np.cos(m2 * pi) - 1) * (
                np.cos(n1 * pi) * np.cos(n2 * pi) - 1
            )

            cond5 = False

            if AR < 1:
                if n1 > n2:
                    cond5 = True
                else:
                    cond5 = False
            elif AR > 1:
                if m2 > m1:
                    cond5 = True
                else:
                    cond5 = False
            elif AR == 1:
                case1 = n1 > m1 and m2 > n2
                case2 = n1 == m2 and m1 == n2
                if case1 == True and case2 == True:
                    cond5 = True

            if cond1 == 0 or cond2 == 0 or cond3 == 0 or cond4 == 0 or cond5 == True:
                comb_set.pop(i)
                i = i
            else:
                i += 1
        return comb_set

    comb = genWaveNo(wavenumber_limit)

    result_array = []
    for j in range(len(comb)):
        m1, n1, m2, n2 = comb[j][:]

        damp1 = visc_damp_total(rho1, rho2, nu1, nu2, gamma, m1, n1, Lx, Ly, h1, h2)
        damp2 = visc_damp_total(rho1, rho2, nu1, nu2, gamma, m2, n2, Lx, Ly, h1, h2)

        beta = betaCrit_func(
            rho1, rho2, nu1, nu2, sig1, sig2, gamma,
            m1, n1, m2, n2,
            Lx, Ly, h1, h2,
            damp1, damp2,
        )

        tmp_var = comb[j][:], beta
        result_array.append(tmp_var)

    beta_array = np.zeros(len(result_array))
    for i in range(len(result_array)):
        beta_array[i] = result_array[i][1]

    # the beta with the lowest value gives us the wave mode which destablizes before others
    min_value = beta_array[:].min()
    idx = find_nearest(beta_array, min_value)

    return result_array[idx][0], beta_array[idx]
