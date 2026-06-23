# === IMPORTS ===
import numpy as np
from astropy import units as u
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

def tau_t1_mig(rock, parameters):
    """Calculates damping timescales for Type I migration given parameters for one rock.
    Based on formulas from Pichierri 2018, Tanaka & Ward 2004, and Cresswell & Nelson 2008. 
    Should be applied on planets and embryos.
    Note that negative tau indicates damping for modify_orbits_forces.

    Args:
        rock (rebound.particle.Particle): REBOUND particle object to 
        calculate damping timescales for.
        parameters (dict): Dictionary containing planet and star parameters.

    Returns:
        tuple: Tuple containing tau_a, tau_e, tau_i for the given object,
        in units of years
    """                        
    Sigma_1au, h_1au, alpha, beta = parameters['Sigma_1au'], parameters['h_1au'], parameters['alpha'], parameters['beta']
    d_edge, h_edge = parameters['ide_position'], parameters['ide_width']
    
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
    t_a = t_wave / (2.7 + 1.1*alpha) * h**-2 * (P + P/abs(P) * (0.070*inc/h + 0.085*(inc/h)**4 - 0.080*(e/h)*(inc/h)**2)) # Eq. 13 / Kajtazi 2023 Eq. 7
    t_e = t_wave / 0.780 * (1 - 0.14*(e/h)**2 + 0.06*(e/h)**3 + 0.18*(e/h)*(inc/h)**2) # Eq. 11
    t_i = t_wave / 0.544 * (1 - 0.3*(inc/h)**2 + 0.24*(inc/h)**3 + 0.14*(e/h)**2*(inc/h)) # Eq. 12
    
    # Smooth planetary trap, Eq. 3.10 in Picheirri 2018
    if a >= d_edge*(1+h_edge):
        tau_a_red = 1
    elif d_edge*(1-h_edge) <= a <= d_edge*(1+h_edge):
        tau_a_red = 5.5 * np.cos(((d_edge*(1+h_edge)-a)*2*np.pi) / (4*h_edge*d_edge)) - 4.5
    elif 0 <= a <= d_edge*(1-h_edge):
        tau_a_red = -10
    else: 
        tau_a_red = 1e-32 # no damping
    
    t_a /= tau_a_red
        
    return -t_a, -t_e, -t_i # Negative so damping
  
def tau_gas(rock, parameters):
    """Calculates damping timescales given parameters for one rock.
    Based on formulas from Adachi 1976. Should be applied on planetesimals.
    Note that negative tau indicates damping for modify_orbits_forces.

    Args:
        rock (rebound.particle.Particle): REBOUND particle object to 
        calculate damping timescales for.
        parameters (dict): Dictionary containing planet and star parameters.

    Returns:
        tuple: Tuple containing tau_a, tau_e, tau_i for the given object,
        in units of years
    """                                     
    Sigma_1au, h_1au, alpha, beta = parameters['Sigma_1au'], parameters['h_1au'], parameters['alpha'], parameters['beta']
    
    m_p = rock.m * Msun # convert to g
    r = rock.d * AU # distance to the star, converted to cm
    R_p = rock.r * AU # radius, converted to cm
    e = rock.e
    inc = rock.inc
    Omega_k = 2*np.pi/rock.P / yr # Keplerian orbital frequency, converted to 1/s
    
    Sigma = Sigma_1au * (r/AU)**-alpha
    h = h_1au * (r/AU)**beta * AU # scale height in cm
    v_K = r*Omega_k # cm/s
        
    rho_p = m_p / (4/3 * np.pi * R_p**3) # Ptsml density (mass / vol) in g/cm^3
    C_d = 0.44 # for km-sized ptsmls and small Mach number (see Brasser 2007) 
    rho_g = Sigma/(np.sqrt(np.pi)*(h)) # g/cm^3
    eta = (11/16)*(h/r)**2
    
    K = 2.157
    E = 1.211
    alpha_rho = 3 # In Adachi, gas density index (not surface density)
    
    t_0 = (8*rho_p*R_p) / (3*C_d*rho_g*v_K) / yr # yr
    t_a = t_0 / (2 * ((2*(2*E+K)/(3*np.pi)*e + 2/np.pi*inc + eta) * eta + 
                        (((2*E+K)/(9*np.pi)) * alpha_rho + (68*E-11*K)/(54*np.pi))* (e**3) +
                        (inc**3)/(2*np.pi))) # Adachi 1976 Eq. 4.11
    t_e = t_0 / ((2*E)/(np.pi)*e + 2/np.pi*inc + eta) # Adachi 1976 Eq. 4.12
    t_i = t_0 / (1/2 * (2*E/np.pi*e + 8/(3*np.pi)*inc + eta)) # Adachi 1976 Eq. 4.13

    return -t_a, -t_e, -t_i # Negative so damping
 
