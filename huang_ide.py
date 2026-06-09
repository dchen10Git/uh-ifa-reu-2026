# === IMPORTS ===
import numpy as np
from astropy import units as u

import re
import os
import pickle as pkl
from dask.distributed import Client, LocalCluster

import warnings
warnings.filterwarnings('ignore')

# === UNIT CONVERSIONS ===
AU = u.AU.to(u.cm) # cm per AU
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) # g per Msun
yr = u.yr.to(u.s) # s per yr
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

# === HUANG IDE MODEL ===

def f_functions(r, r_c, Delta, A_a, A_e):
    # Piecewise functions f_a and f_e
    conditions = [
        r < r_c - Delta,
        (r_c - Delta <= r) & (r < r_c),
        (r_c <= r) & (r < r_c + Delta + 1 / A_a),
        r >= r_c + Delta + 1 / A_a
    ]

    f_a = [
        0,          
        A_a * (r_c - Delta - r) / Delta,
        (r-r_c)* (A_a + 1) / (Delta + 1/A_a) - (A_a), # modified to make it continuous, paper might be wrong
        1
    ]

    f_e = [
        0,          
        A_e * (r - r_c + Delta) / Delta,
        (A_e - 1) * (r_c + Delta + 1 / A_a - r) / (Delta + 1 / A_a) + 1, 
        1
    ]

    f_a_vals = np.select(conditions, f_a, default=np.nan)
    f_e_vals = np.select(conditions, f_e, default=np.nan)
    return f_a_vals, f_e_vals

def get_taus_huang(m_vals, r_vals, m_star, current_a_vals, tau_a_earth, r_earth, q_earth, q_vals, Q_sim, C_e):
    '''Computes damping timescales based on current semimajor axis values.
 
    Parameters:
        current_a_vals: 1D NumPy array of current semimajor axis values.

    Returns:
        tau_a: semimajor axis damping timescale.
        tau_e: eccentricity damping timescale.
    '''
    f_a_vals, f_e_vals = f_functions(current_a_vals)
    tau_a = -tau_a_earth * (q_earth / q_vals) / f_a_vals # negative so damping.
    tau_e_disk = C_e * tau_a * f_a_vals / f_e_vals # I removed a h^2 here
    tau_e_star = 7.63e5 * Q_sim * (m_vals/m_earth) * (1/m_star)**1.5 * (r_earth/r_vals)** 5 * (current_a_vals/0.05)**6.5
    tau_e = (tau_e_disk * tau_e_star) / (tau_e_disk + tau_e_star) # combining the two based on Eqs. 4 and 13
    return tau_a, tau_e

'''
q_vals = m_vals / m_star
q_earth =  3.003e-6 / m_star

Huang & Ormel (2022) used positions r_c in [0.013 - 0.030 au] with width 
Delta = 2hr_c = 0.06 r_c. In particular, r_c = 0.023 worked best.
'''