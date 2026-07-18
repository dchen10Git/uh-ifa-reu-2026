import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import astropy.units as u
from pathlib import Path
import rebound_sims as reb_sims
from scipy.optimize import brentq
from timescales import get_ta_te

# === UNIT CONVERSIONS ===
AU = u.AU.to(u.cm)    
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) 
yr = u.yr.to(u.s)    
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU)

# Laplace coefficient library
with open("fg_library.pkl", "rb") as fpkl:
    fg_lib = pkl.load(fpkl)
    
def get_k_params(k):
    """Return the k-dependent constants (alpha_res, m_order, B, R).
    Should be used for first-order resonances only."""
    alpha_res = ((k - 1) / k) ** (2 / 3)
    m_order = k - 1 # m+1:m resonance = k:k-1 resonance
    B = 0.8 * m_order
    f_coef, g_coef = fg_lib[(k, 1)]  # first-order resonance
    
    # delta_{m,1} triggers at k = 2
    if k != 2:
        R = abs(f_coef) / g_coef
    else:
        R = abs(f_coef) / (g_coef - 2 * alpha_res)
    return alpha_res, m_order, B, R

def eps_p_and_crit(k, parameters, m_star, m1, m2, r1, r2, a1, a2, p_coupling=2, tau1='mig_and_gas', tau2='mig'):
    """
    Calculates the critical planet mass ratio for overstability, eps_p_crit, and the actual planet mass ratio, eps_p.
    Returns eps_p, eps_p_crit, ta (convergent migration timescale). See Deck & Batygin (2015) Eq. 15 for the full formula.
    The dictionary 'parameters' requires the following keys: Sigma_1au, h_1au, alpha, beta, ide_position, ide_width.
    """
    
    alpha_res, m_order, B, R = get_k_params(k)
    ta1, te1 = get_ta_te(parameters, m_star, m1, r1, a1, t=0, ide=True, tau=tau1)
    ta2, te2 = get_ta_te(parameters, m_star, m2, r2, a2, t=0, ide=True, tau=tau2)

    zeta = m1 / m2
    ta = 1 / (1 / ta2 - 1 / ta1)
    te = 1 / (1 / te1 + zeta / te2)
    tae = 1 / (1 / te1 - zeta ** 2 * alpha_res / (R ** 2 * te2))
    
    eps_p = (m1 + m2) / m_star
    C = (
        (3 * p_coupling * m_order) / (B * 2 ** 1.5)
        * (te / tae)
        * (1 + zeta) ** 2
        / (m_order * (zeta + 1) + p_coupling * (te / tae)) ** 1.5
    )
    eps_p_crit = C * (te / ta) ** 1.5  # this might be nan if ta is negative:
                                       # that indicates the inner planet migrates faster, which is possible
    return eps_p, eps_p_crit, ta

def h_overstable_boundary(k, parameters, m_star, m1, m2, r1, r2, a1, a2, hmin=0.01, hmax=0.11):
    """
    Returns the critical h satisfying eps_p = eps_p_crit(h)
    See Deck & Batygin (2015) Eq. 15 for the full formula.
    Returns np.nan if no crossing exists.
    """    
    def f(h):
        try:
            parameters['h_1au'] = h
            eps_p, eps_p_crit, _ = eps_p_and_crit(k, parameters, m_star, m1, m2, r1, r2, a1, a2)
            return eps_p - eps_p_crit
        except Exception:
            return np.nan

    try:
        if np.sign(f(hmin)) == np.sign(f(hmax)):
            return np.nan
        return brentq(f, hmin, hmax)
    except Exception:
        return np.nan

def adiabaticity_crit(k, m_star, m1, m2):
    """Returns C for:
        1 / (tau_a * Omega_1) <= C
        See Batygin & Morbidelli (2026) Eq. 6 for the full formula.
    """
    
    # # Batygin 2015
    # if k == 2:
    #     # first-order resonance coefficient
    #     with open("fg_library.pkl", "rb") as f:
    #         fg_lib = pkl.load(f)

    #     f, g = fg_lib[(2, 1)]
    #     f_res = f

    #     P = 0.25**(3/2)
    #     Omega = 2*np.pi / P
    #     Mstar = 1
    #     C = (
    #         np.pi**2
    #         * Omega**3
    #         * (Mstar/(m1+m2))**(4/3)
    #         * (k-1)**(-2/9)
    #         * np.sqrt(3)
    #         * abs(f_res)**(4/3)
    #     )
    
    # "Compact case" from B&M 2026
    if k > 1:
        C = (
            (5 * np.pi / 8)
            * (5 / (36 * k**5 * (k - 1)**(2/3)))**(1/3)
            * (m_star / (m1 + m2))**(4/3)
        )
    
    return C

def dissipative_crit(k, m_star, m1, m2, h, s=1):
    """
    Assumes compact configuratino with damping via Type I migration. 
    From Batygin & Morbidelli (2026) Eq. 5. Returns the critical
    value for dissipative stability: C_k such that
        1 / (tau_a * Omega_1) <= C_k
    """
    chi_a = 1 / (2.7 + 1.1 * s)
    chi_e = 1 / 0.780
    
    K1 = chi_a/chi_e * h**-2 * (m1 / (m2-m1))
    K2 = chi_a/chi_e * h**-2 * (m2 / (m2-m1))
    
    zeta = m1 / m2
    return (
        (m2 / m_star)
        * np.sqrt(
            32 * k**3 * (1 + zeta) * (zeta * K1 + K2)
            / (25 * K1 * K2)
        )
    )

def overstability_K2_crit(k, m1, m2, s=1):
    '''
    Overstability criterion from Batygin & Morbidelli (2026); not used for our
    purposes as it assumes Type I migration only.
    '''
    
    zeta = m1 / m2
    chi_a = 1 / (2.7 + 1.1 * s)
    chi_e = 1 / 0.780
    
    lhs_base = (
        zeta
        * (k * (1 + zeta) - 2 * zeta)
        / (1 - zeta)
        * (chi_a / chi_e)
    )

    A = lhs_base**(3/2)
    B = (15 / 32) * (1 - zeta**2) / m2 # fixed typo

    h_crit = (A / B)**(1/3) # critical aspect ratio
    K2_crit = chi_a/chi_e / (1-zeta) * h_crit**-2 
    
    return K2_crit
