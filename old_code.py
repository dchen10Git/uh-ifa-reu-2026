# === IMPORTS ===
import numpy as np
import pandas as pd
from astropy import units as u
import re
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

# From Izidoro 2014
def get_tau(rock, rock_name, parameters):
    """Calculates damping timescales given parameters for one rock.
    Based on formulas from Izidoro 2014, Brasser 2007, and Adachi 1976.
    Planets and embryos are affected by migration while planetesimals
    are affected by gas drag.
    Note that negative tau indicates damping for modify_orbits_forces.

    Args:
        rock (rebound.particle.Particle): REBOUND particle object to 
        calculate damping timescales for.
        rock_name (str): Name of the particle.
        parameters (dict): Dictionary containing planet and star parameters.

    Returns:
        tuple: Tuple containing tau_a, tau_e, tau_i for the given object,
        in units of years
    """                                     
    Sigma_1au, h_1au, alpha, beta = parameters['Sigma_1au'], parameters['h_1au'], parameters['ide_position'], parameters['ide_width']
    
    m_p = rock.m
    r = rock.d # distance to the star
    a_p = rock.a
    e = rock.e
    inc = rock.inc
    Omega_k = 2*np.pi/rock.P # Keplerian orbital frequency
        
    # Type I migration
    if 'planet' in rock_name:
        Sigma_1au *= AU**2 / Msun # Converted to Msun/AU^2 from g/cm^2
        Sigma_g = Sigma_1au * r**(-alpha)
        h = h_1au * r**beta # scale height
        
        t_a = (2/(2.7+1.1*alpha)) * (1/m_p) * (1/(Sigma_g*a_p**2)) * (h/r)**2 * ((1 + (e*r/(1.3*h))**5) / (1 - (e*r/(1.1*h))**4)) / Omega_k
        t_wave = (1/m_p) * (1/(Sigma_g*a_p**2)) * (h/r)**4 / Omega_k
        t_e = (t_wave/0.780) * (1 - 0.14*(e/(h/r))**2 + 0.06 * (e/(h/r))**3 + 0.18 * (e/(h/r)) * (inc/(h/r))**2)
        t_i = (t_wave/0.544) * (1 - 0.3*(inc/(h/r))**2 + 0.24 * (inc/(h/r))**3 + 0.14 * (e/(h/r))**2 * (inc/(h/r)))
    
    # Type I migration
    elif 'embryo' in rock_name:   
        Sigma_1au *= AU**2 / Msun # Converted to Msun/AU^2 from g/cm^2        
        Sigma_g = Sigma_1au * r**(-alpha)
        h = h_1au * r**beta # scale height
        
        t_a = (2/(2.7+1.1*alpha)) * (1/m_p) * (1/(Sigma_g*a_p**2)) * (h/r)**2 * ((1 + (e*r/(1.3*h))**5) / (1 - (e*r/(1.1*h))**4)) / Omega_k
        t_wave = (1/m_p) * (1/(Sigma_g*a_p**2)) * (h/r)**4 / Omega_k
        t_e = (t_wave/0.780) * (1 - 0.14*(e/(h/r))**2 + 0.06 * (e/(h/r))**3 + 0.18 * (e/(h/r)) * (inc/(h/r))**2)
        t_i = (t_wave/0.544) * (1 - 0.3*(inc/(h/r))**2 + 0.24 * (inc/(h/r))**3 + 0.14 * (e/(h/r))**2 * (inc/(h/r)))
           
    # Aerodynamic gas drag 
    elif 'ptsml' in rock_name:
        Sigma_g = Sigma_1au * (r/AU)**(-alpha) # g/cm^2
        h = h_1au * (r/AU)**beta # scale height in AU
        v_K = r*Omega_k # cm/s
        
        m_p *= u.Msun.to(u.g) # convert to g
        R_p = rock.r * AU # cm
        rho_p = m_p / (4/3 * np.pi * R_p**3) # Ptsml density (mass / vol) in g/cm^3
        C_d = 0.44 # for km-sized ptsmls and small Mach number (see Brasser 2007) 
        rho_g = Sigma_1au/(np.sqrt(np.pi)*(h_1au*AU)) * (r/AU)**(-11/4) # g/cm^3
        eta = (11/16)*(h*AU/r)**2
        
        K = 2.157
        E = 1.211
        alpha = 3 # constant in Adachi model (NOT flaring index)
        
        t_0 = (8*rho_p*R_p) / (3*C_d*rho_g*v_K) / yr # yr
        t_a = t_0 / (2 * ((2*(2*E+K)/(3*np.pi)*e + 2/np.pi*inc + eta) * eta + 
                            (((2*E+K)/(9*np.pi)) * alpha + (68*E-11*K)/(54*np.pi))* (e**3) +
                            (inc**3)/(2*np.pi))) # Adachi 1976 Eq. 4.11
        t_e = t_0 / ((2*E)/(np.pi)*e + 2/np.pi*inc + eta) # Adachi 1976 Eq. 4.12
        t_i = t_0 / (1/2 * (2*E/np.pi*e + 8/(3*np.pi)*inc + eta)) # Adachi 1976 Eq. 4.13
        
    return -t_a, -t_e, -t_i # Negative so damping

# Parameter generation

def parse_entry(entry):
    """Parses a given string with uncertainties. 
    
    Converts asymmetric uncertainties into symmetric uncertainties if needed.
        
    Parameters:
        entry (str): String of the form '1.0±0.01' or '1.04 +0.01 -0.02'
    
    Returns:
        tuple: mu (float), sigma (float)
    """
    
    if pd.isna(entry):
        raise ValueError("Entry is NaN")
    
    # Remove extra whitespace
    entry = entry.strip()
    
    # Remove spaces for easier parsing
    entry_nospace = entry.replace(" ", "")
    
    # Case 1: symmetric uncertainty (±)
    match_pm = re.match(r"^([0-9.+\-eE]+)±([0-9.+\-eE]+)$", entry_nospace)
    if match_pm:
        mu = float(match_pm.group(1))
        sigma = float(match_pm.group(2))
        return mu, sigma
    
    # Case 2: asymmetric uncertainty (+x -y)
    match_asym = re.match(
        r"^([0-9.+\-eE]+)\+([0-9.+\-eE]+)\-([0-9.+\-eE]+)$",
        entry_nospace
    )
    if match_asym:
        mu = float(match_asym.group(1))
        sigma_plus = float(match_asym.group(2))
        sigma_minus = float(match_asym.group(3))
        
        # Convert asymmetric → effective symmetric σ
        sigma = 0.5 * (sigma_plus + sigma_minus)
        
        return mu, sigma
    
    raise ValueError(f"Could not parse entry: {entry}")

# For TRAPPIST-1
def generate_params_from_csv(csv_file, params, random=False):
    """
    Reads parameter CSV downloaded from the NASA Exoplanet archive of
    TRAPPIST-1 parameters and returns randomly drawn params.
    """
    
    df = pd.read_csv(csv_file)
    
    # Set index to Source column for easy lookup
    df = df.set_index("Source")
    
    # Extract Agol et al. 2021 column
    col = "Agol et al. 2021"
    
    params_dict = {}
    
    for param in params:
        # Parse parameter
        mu, sigma = parse_entry(df.loc[param, col])
    
        # Draw Gaussian samples if we want them to be random
        if random:
            samples = np.random.normal(mu, sigma)
        else:
            samples = mu

        # Add to dict
        params_dict[param] = samples
    
    return params_dict

# For TRAPPIST-1
def generate_params(planet_names, rng):
    # Nested dict containing params for each planet in sim
    # Randomly generate mass, radius, & semimajor axis values
    planet_params = {f"{planet_name}": generate_params_from_csv(f'TRAPPIST-1_params/TRAPPIST-1_{planet_name}_planet_params.csv', ('a (au)', 'Rp (R⨁)', 'Mp (M⨁)'), random=False) for planet_name in planet_names}
    stellar_params = generate_params_from_csv('TRAPPIST-1_params/TRAPPIST-1_stellar_params.csv', ('R✶ (R⦿)', 'M✶ (M⦿)'), random=False)
    
    # Define planet masses (m)
    m_vals = np.array([planet_params[planet_name]['Mp (M⨁)'] for planet_name in planet_names])
    m_vals *= m_earth # convert to Msun

    # Define planet radii (r)
    r_vals = np.array([planet_params[planet_name]['Rp (R⨁)'] for planet_name in planet_names])
    r_vals *= r_earth # convert to AU

    # Define stellar parameters
    m_star = stellar_params['M✶ (M⦿)']
    r_star = stellar_params['R✶ (R⦿)'] * r_sun

    # Uniform random initial period ratios just above 2:1
    initial_P_ratios = rng.uniform(1.53, 1.55, size=len(planet_names)-1) 
    initial_P_ratios = [1.53, 1.53, 1.8] # for b/c/d/e in early cavity infall model
                                    
    return m_vals, r_vals, m_star, r_star, initial_P_ratios
