# === IMPORTS ===
import numpy as np
from scipy import special
from astropy import units as u
import warnings
import rebound
warnings.filterwarnings('ignore')

# === UNIT CONVERSIONS ===
AU = u.AU.to(u.cm) # cm per AU
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) # g per Msun
yr = u.yr.to(u.s) # s per yr
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

def rebx_ide(a, parameters):
    d_edge, h_edge = parameters['ide_position'], parameters['ide_width']

    # Smooth planetary trap, Eq. 3.10 in Picheirri 2018
    if a >= d_edge*(1+h_edge):
        tau_a_red = 1
    elif d_edge*(1-h_edge) <= a <= d_edge*(1+h_edge):
        tau_a_red = 5.5 * np.cos(((d_edge*(1+h_edge)-a)*2*np.pi) / (4*h_edge*d_edge)) - 4.5
    elif 0 <= a <= d_edge*(1-h_edge):
        tau_a_red = -10
    else: 
        tau_a_red = 1e-32 # no damping
    
    # Prevent tau_a_red being too small:
    if tau_a_red >= 0:
        tau_a_red = max(tau_a_red, 1e-9)
    else:
        tau_a_red = min(tau_a_red, -1e-9)
        
    # # New IDE model with Phi
    # arg = (a-d_edge) / h_edge
    # Phi = 0.5*(1 + special.erf(arg))
    # seff = alpha - parameters['zeta'] * a * np.exp(-(arg**2)) / (Phi * np.sqrt(np.pi) * h_edge)
    # TorqC = (2.5 - 0.5 - 0.1*seff) + 1.4 - 1.1*(1.5 - seff)
    # t_wave_eff = 1/(m_p) * (1/(Sigma*Phi*a**2)) * h**4 / np.sqrt(G/a**3) # factor of Phi added
    # t_a = t_wave_eff / TorqC * h**-2 * (P + P/abs(P) * (0.070*inc/h + 0.085*(inc/h)**4 - 0.080*(e/h)*(inc/h)**2))
    
    return tau_a_red

def tau_t1_mig(rock, parameters, t, ide=True):
    """Calculates damping timescales for Type I migration given parameters for one rock.
    Based on formulas from Pichierri 2018, Tanaka & Ward 2004, and Cresswell & Nelson 2008. 
    Should be applied on planets and embryos.
    Note that negative tau indicates damping for modify_orbits_forces.

    Args:
        rock (rebound.particle.Particle): REBOUND particle object to 
        calculate damping timescales for.
        parameters (dict): Dictionary containing planet and star parameters.
        t (float): Current simulation time in years. Determines disk dissipation, if set.
        ide (bool): Whether to apply the IDE correction.

    Returns:
        tuple: Tuple containing tau_a, tau_e, tau_i for the given object,
        in units of years
    """                        
    Sigma_1au, h_1au, alpha, beta = parameters['Sigma_1au'], parameters['h_1au'], parameters['alpha'], parameters['beta']
    
    # If disk dissipation is enabled, reduce Sigma_1au over time
    if 'tau_dissipation' in parameters: 
        if parameters['tau_dissipation'] is not None:
            tau_dissipation = parameters['tau_dissipation']
            Sigma_1au *= np.exp(-t/tau_dissipation)
    
    # Everything should be in simulation units
    m_p = rock.m
    a = rock.a
    r = rock.d
    e = rock.e
    inc = rock.inc
    
    Sigma = (Sigma_1au * AU**2 / Msun) * r**-alpha # Converted from g/cm^2
    h = h_1au * r**beta # scale height; if beta = 0, h = h_1au
    
    # Note the following formulas assume h = h_1au
    P = (1 + (e/(2.25*h))**1.2 + (e/(2.84*h))**6) / (1 - (e/(2.02*h))**4) # = 1 in low-e low-i limit
    t_wave = 1/(m_p) * (1/(Sigma*a**2)) * h**4 / np.sqrt(G/a**3) # Cresswell & Nelson 2008 Eq. 9 / Pichierri 2018 Eq. 3.3
    
    # Old t_a
    t_a = t_wave / (2.7 + 1.1*alpha) * h**-2 * (P + P/abs(P) * (0.070*inc/h + 0.085*(inc/h)**4 - 0.080*(e/h)*(inc/h)**2)) # Eq. 13 / Kajtazi 2023 Eq. 7
    
    if ide:
        tau_a_red = rebx_ide(a, parameters)
        t_a /= tau_a_red
    
    t_e = t_wave / 0.780 * (1 - 0.14*(e/h)**2 + 0.06*(e/h)**3 + 0.18*(e/h)*(inc/h)**2) # Eq. 11
    t_i = t_wave / 0.544 * (1 - 0.3*(inc/h)**2 + 0.24*(inc/h)**3 + 0.14*(e/h)**2*(inc/h)) # Eq. 12
    
    return -t_a, -t_e, -t_i # Negative so damping
  
def tau_gas(rock, parameters, t, ide=True):
    """Calculates damping timescales given parameters for one rock.
    Based on formulas from Adachi 1976. Should be applied on planetesimals.
    Note that negative tau indicates damping for modify_orbits_forces.

    Args:
        rock (rebound.particle.Particle): REBOUND particle object to 
        calculate damping timescales for.
        parameters (dict): Dictionary containing planet and star parameters.
        t (float): Current simulation time in years. Determines disk dissipation, if set.
        ide (bool): Whether to apply the IDE correction.

    Returns:
        tuple: Tuple containing tau_a, tau_e, tau_i for the given object,
        in units of years
    """                                     
    Sigma_1au, h_1au, alpha, beta = parameters['Sigma_1au'], parameters['h_1au'], parameters['alpha'], parameters['beta']
    
    # If disk dissipation is enabled, reduce Sigma_1au over time
    if 'tau_dissipation' in parameters:
        if parameters['tau_dissipation'] is not None:
            tau_dissipation = parameters['tau_dissipation']
            Sigma_1au *= np.exp(-t/tau_dissipation)

    m_p = rock.m * Msun # convert to g
    r = rock.d * AU # distance to the star, converted to cm
    R_p = rock.r * AU # radius, converted to cm
    e = rock.e
    inc = rock.inc
    Omega_k = 2*np.pi/rock.P / yr # Keplerian orbital frequency, converted to 1/s
    
    Sigma = Sigma_1au * (r/AU)**-alpha
    h = h_1au * (r/AU)**beta # scale height (dimensionless
    v_K = r*Omega_k # cm/s
        
    rho_p = m_p / (4/3 * np.pi * R_p**3) # Ptsml density (mass / vol) in g/cm^3
    C_d = 0.44 # for km-sized ptsmls and small Mach number (see Brasser 2007) 
    rho_g = Sigma/(np.sqrt(np.pi)*(h*r)) # g/cm^3
    eta = (11/16) * h**2 
    
    K = 2.157
    E = 1.211
    alpha_rho = 3 # In Adachi, gas density index (not surface density)
    
    t_0 = (8*rho_p*R_p) / (3*C_d*rho_g*v_K) / yr # yr
    t_a = t_0 / (2 * ((2*(2*E+K)/(3*np.pi)*e + 2/np.pi*inc + eta) * eta + 
                        (((2*E+K)/(9*np.pi)) * alpha_rho + (68*E-11*K)/(54*np.pi))* (e**3) +
                        (inc**3)/(2*np.pi))) # Adachi 1976 Eq. 4.11
    t_e = t_0 / ((2*E)/(np.pi)*e + 2/np.pi*inc + eta) # Adachi 1976 Eq. 4.12
    t_i = t_0 / (1/2 * (2*E/np.pi*e + 8/(3*np.pi)*inc + eta)) # Adachi 1976 Eq. 4.13

    if ide:
        tau_a_red = rebx_ide(rock.a, parameters)
        t_a /= tau_a_red
        
    return -t_a, -t_e, -t_i # Negative so damping
 
def tau_mig_and_gas(rock, parameters, t, ide=True):
    """Calculates damping timescales given parameters for one rock.
    Note that negative tau indicates damping for modify_orbits_forces.

    Args:
        rock (rebound.particle.Particle): REBOUND particle object to 
        calculate damping timescales for.
        parameters (dict): Dictionary containing planet and star parameters.
        t (float): Current simulation time in years. Determines disk dissipation, if set.
        ide (bool): Whether to apply the IDE correction.

    Returns:
        tuple: Tuple containing tau_a, tau_e, tau_i for the given object,
        in units of years
    """
    tau_mig = -np.array(tau_t1_mig(rock, parameters, t, ide))
    tau_drag = -np.array(tau_gas(rock, parameters, t, ide)) # prevent name confusion
    tau_both = - (1 / (1 / tau_mig + 1 / tau_drag)) # return as negative for damping timescale
    
    return tau_both

def get_ta_te(parameters, m_star, m1, r1, a1, t=0, ide=True, tau='mig_and_gas'):
    """Given disk parameters and planet parameters (without needing to initialize the simulation), 
    calculate damping timescale of that planet, and return (ta, te) from tau_t1_mig, 
    tau_gas, or tau_mig_and_gas. tau should be 'mig_and_gas', 'mig', or 'gas'.
    Positive timescales indicates damping. 
    One should use the other functions to get negative timescales for use in modify_orbits_forces.
    
    Parameters requires the following keys: Sigma_1au, h_1au, alpha, beta, ide_position, ide_width."""

    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.add(m=m_star)
    sim.add(m=m1, r=r1, a=a1)

    if tau == 'mig_and_gas':        
        ta, te = -np.array(tau_mig_and_gas(sim.particles[1], parameters, t, ide)[0:2])
    elif tau == 'mig':
        ta, te = -np.array(tau_t1_mig(sim.particles[1], parameters, t, ide)[0:2])
    elif tau == 'gas':
        ta, te = -np.array(tau_gas(sim.particles[1], parameters, t, ide)[0:2])
    else:
        raise ValueError("Invalid tau type. Choose from 'mig_and_gas', 'mig', or 'gas'.")
    
    return ta, te
        
# === Old code for Type I Migration ===
'''
# New Type I migration code using Phi
mig = rebx.load_force("IDE_typeI")
rebx.add_force(mig)

mig.params["rx"] = parameters["ide_position"]
mig.params["w"] = parameters['ide_width']
mig.params["tIm_surface_density_1"] = parameters['Sigma_1au'] * AU**2 / Msun
mig.params["tIm_sd0_exponent"] = parameters['alpha']
mig.params["tIm_scale_height_1"] = parameters['h_1au']
mig.params["tIm_flaring_index"] = parameters['beta']
mig.params["zeta"] = parameters['zeta']
mig.params["min_mass"] = parameters['m_em'] * m_earth

# REBOUNDx implementation of Type I migration (for debugging)
mig = rebx.load_force("type_I_migration")
rebx.add_force(mig)

# REBOUNDx Type I migration implementation
mig.params["tIm_surface_density_1"] = parameters['Sigma_1au'] * AU**2 / Msun # Converted to Msun/AU^2 from g/cm^2
mig.params["tIm_surface_density_exponent"] = parameters['alpha']
mig.params["tIm_scale_height_1"] = parameters['h_1au']
mig.params["tIm_flaring_index"] = parameters['beta']
mig.params["ide_position"] = parameters["ide_position"]
mig.params["ide_width"] = parameters['ide_width']
'''

# === HUANG IDE MODEL (No longer used) ===

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