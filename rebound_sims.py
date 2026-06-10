# === IMPORTS ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rebound
import reboundx

from astropy import units as u
from astropy import constants as const
from pathlib import Path
from time import time

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

# === DATA MANAGEMENT ===
    
def concatenate_data(stages):
    if type(stages) == dict:
        return stages
    
    # Concatenate first two
    all_stage_data = {
            name: pd.concat(
                [stages[0][name], stages[1][name]],
                ignore_index=True
            )
            for name in stages[0]
        }
    
    # Concatenate the rest
    for i in range(2, len(stages)):
        next_stages = stages[i]
        all_stage_data = {
            name: pd.concat(
                [all_stage_data[name], next_stages[name]],
                ignore_index=True
            )
            for name in all_stage_data
        }
    return all_stage_data
    
def save_simulation_run(stage_data, sim_id, file_path, sim_metadata=None):
    """Save all planets from one simulation run into HDF5.
    
    Parameters:
        stage_data (dict): Data in a dict in the form {planet_name: DataFrame}.
        sim_id (int): Simulation ID.
        file_path (str): File path for data storage.
        sim_metadata (dict, optional): e.g. {"m_star": 1.0, "integrator": "whfast"}
    """
    with pd.HDFStore(file_path, mode="a") as store:
        # Save planet list
        planet_list = list(stage_data.keys())
        store.put("planet_list",
                  pd.Series(planet_list))
        
        # Save simulation metadata
        if sim_metadata is not None:
            store.put("metadata",
                      pd.Series(sim_metadata))
        
        # Save each planet
        for planet_name, df in stage_data.items():
            
            key = f"{planet_name}"
            store.put(key, df, format="table")
            
            # Attach attributes
            storer = store.get_storer(key)
            storer.attrs.planet_name = planet_name
            storer.attrs.sim_id = sim_id
         
def load_simulation_run(file_path):
    '''Load a simulation for analysis.
    
    Parameters:
        file_path (str): Location of simulation data (saved as hdf5)
    
    Returns:
        tuple:
            * result (dict[str, pd.DataFrame]): Loaded simulation.
            * metadata (dict): Metadata containing additional information (planet_names, sim_id, IDE params, etc.).
    
    '''
    result = {}
    
    with pd.HDFStore(file_path, mode="r") as store:
        
        planet_list = store["planet_list"].tolist()
        
        for planet_name in planet_list:
            key = f"{planet_name}"
            df = store[key]
            
            # Pull HDF5 attributes
            storer = store.get_storer(key)
            df.attrs["planet_name"] = storer.attrs.planet_name
            df.attrs["sim_id"] = storer.attrs.sim_id
            
            result[planet_name] = df
        
        metadata = store[f"metadata"].to_dict()
    
    return result, metadata

# === PARAMETER GENERATION ===

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

# From Izidoro 2014
def get_tau(rock, rock_name, Sigma_1au=None, h_1au=None):
    """Calculates damping timescales given parameters for one rock.
    Based on formulas from Izidoro 2014, Brasser 2007, and Adachi 1976.
    Planets and embryos are affected by migration while planetesimals
    are affected by gas drag.
    Note that negative tau indicates damping for modify_orbits_forces.

    Args:
        rock (rebound.particle.Particle): REBOUND particle object to 
        calculate damping timescales for.
        rock_name (str): Name of the particle.
        Sigma_1au (float): Disk surface density at 1 AU, given in units
        of g/cm^2.
        h_1au (float): Aspect ratio (scale height at 1 AU).

    Returns:
        tuple: Tuple containing tau_a, tau_e, tau_i for the given object,
        in units of years
    """    
    # Type I migration
    if 'planet' in rock_name:
        m_p = rock.m
        r = rock.d # distance to the star
        a_p = rock.a
        e = rock.e
        inc = rock.inc
        Omega_k = 2*np.pi/rock.P # Keplerian orbital frequency
        
        Sigma_1au *= AU**2 / Msun # Converted to Msun/AU^2 from g/cm^2
        Sigma_g = Sigma_1au * r**(-3/2)
        alpha = 1.5 # flaring index
        h = h_1au * r**1.25 # scale height
        
        t_a = (2/(2.7+1.1*alpha)) * (1/m_p) * (1/(Sigma_g*a_p**2)) * (h/r)**2 * ((1 + (e*r/(1.3*h))**5) / (1 - (e*r/(1.1*h))**4)) / Omega_k
        t_wave = (1/m_p) * (1/(Sigma_g*a_p**2)) * (h/r)**4 / Omega_k
        t_e = (t_wave/0.780) * (1 - 0.14*(e/(h/r))**2 + 0.06 * (e/(h/r))**3 + 0.18 * (e/(h/r)) * (inc/(h/r))**2)
        t_i = (t_wave/0.544) * (1 - 0.3*(inc/(h/r))**2 + 0.24 * (inc/(h/r))**3 + 0.14 * (e/(h/r))**2 * (inc/(h/r)))
        return -t_a, -t_e, -t_i # Negative so damping
    
    elif 'embryo' in rock_name:
        m_p = rock.m
        r = rock.d # distance to the star
        a_p = rock.a
        e = rock.e
        inc = rock.inc
        Omega_k = 2*np.pi/rock.P # Keplerian orbital frequency
            
        Sigma_1au *= AU**2 / Msun # Converted to Msun/AU^2 from g/cm^2        
        Sigma_g = Sigma_1au * r**(-3/2)
        alpha = 1.5 # flaring index
        h = h_1au * r**1.25 # scale height
        
        t_a = (2/(2.7+1.1*alpha)) * (1/m_p) * (1/(Sigma_g*a_p**2)) * (h/r)**2 * ((1 + (e*r/(1.3*h))**5) / (1 - (e*r/(1.1*h))**4)) / Omega_k
        t_wave = (1/m_p) * (1/(Sigma_g*a_p**2)) * (h/r)**4 / Omega_k
        t_e = (t_wave/0.780) * (1 - 0.14*(e/(h/r))**2 + 0.06 * (e/(h/r))**3 + 0.18 * (e/(h/r)) * (inc/(h/r))**2)
        t_i = (t_wave/0.544) * (1 - 0.3*(inc/(h/r))**2 + 0.24 * (inc/(h/r))**3 + 0.14 * (e/(h/r))**2 * (inc/(h/r)))
        
        return -t_a, -t_e, -t_i # Negative so damping
    
    elif 'ptsml' in rock_name:
        e = rock.e
        inc = rock.inc
        R_p = rock.r * AU # cm
        r = rock.d * AU # cm
        Omega_k = 2*np.pi/(rock.P*yr) # 1/s
        
        Sigma_g = Sigma_1au * (r/AU)**(-3/2) # g/cm^2
            
        h = h_1au * (r/AU)**1.25 # scale height in AU
        v_K = r*Omega_k # cm/s
        
        m_p = 0.0033 * u.Mearth.to(u.g) # g
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
        return -t_a, -t_e, -t_i
    
# === RUNNING THE SIM ===

def get_hill_radius(m1, a1, m2, a2, M_star):
    '''Returns mutual hill radius of two planets.'''
    return ((m1+m2)/(3*M_star))**1/3 * (a1+a2)/2

def integrate_sim(sim, rocks, rock_names, parameters, years, start_time=0):
    '''Integrates a REBOUND simulation over a given number of years and
    saves the new state of the sim.
    
    Parameters:
        sim (rebound.Simulation): Simulation object to integrate.
        rocks (list[rebound.Particles]): List containing rocks (non-stellar particles) in the sim.
        rock_names (list[str]): List containing names of the rocks (non-stellar particles).
        parameters (dict): Dictionary containing planet and star parameters.
        years (float): Number of years to integrate.
        start_time (float, optional): Start time of the integration, defaults to 0.
        
    Returns:
        tuple:
            * stage_data (pd.Dataframe): Simulation data.
            * completed_sim (bool): Whether the integration was fully compelted.
    '''
    m_vals, m_star, r_vals, r_star, Sigma_1au, h_1au = parameters["m_vals"], parameters["m_star"], parameters["r_vals"], parameters["r_star"], parameters["Sigma_1au"], parameters["h_1au"]
    num_pl, num_em, num_ptsml = parameters["num_pl"], parameters["num_em"], parameters["num_ptsml"]
    num_rocks = len(rock_names)
    
    # Set up times for integration & data collection
    n_out = 2000 # number of data points to collect
    stage_times = np.linspace(start_time, years+start_time, n_out, endpoint=False)  # all times to integrate over
        
    hist = {
        "time": stage_times,

        "a": np.full((n_out, num_rocks), np.nan),
        "e": np.full((n_out, num_rocks), np.nan),
        "inc": np.full((n_out, num_rocks), np.nan),
        "P": np.full((n_out, num_rocks), np.nan),
        "P_ratio": np.full((n_out, num_rocks), np.nan),
        "l": np.full((n_out, num_rocks), np.nan),
        "pomega": np.full((n_out, num_rocks), np.nan),
        
        "tau_a": np.full((n_out, num_rocks), np.nan),
        "tau_e": np.full((n_out, num_rocks), np.nan),
        "tau_i": np.full((n_out, num_rocks), np.nan),
    }
        
    sim.random_seed = 16 # for reproducibility

    completed_sim = True

    for i, t in enumerate(stage_times): 
        
        alive_rocks = []
        for j, name in enumerate(rock_names):

            try: # Prevent error for removed particles
                rock = sim.particles[name]
                
                tau_a, tau_e, tau_i = get_tau(rock, name, Sigma_1au, h_1au)
                rock.params["tau_a"] = tau_a
                rock.params["tau_e"] = tau_e
                rock.params["tau_inc"] = tau_i
        
                hist["tau_a"][i, j] = tau_a
                hist["tau_e"][i, j] = tau_e
                hist["tau_i"][i, j] = tau_i
                                
                # Save data
                hist["a"][i, j] = rock.a
                hist["e"][i, j] = rock.e
                hist["inc"][i, j] = rock.inc
                hist["P"][i, j] = rock.P
                hist["l"][i, j] = rock.l
                hist["pomega"][i, j] = rock.pomega

                if j < num_pl-1: # don't record period ratio for last planet
                    hist["P_ratio"][i, j] = rocks[j+1].P / rock.P # index might be wrong if planet is gone
                
                # Remove particle if planet goes into star
                if rock.d < 0.05 or rock.d > 100:
                    sim.remove(hash=name)
                    # print(f"{name} removed")
                    
                alive_rocks.append(rock)
                    
            except rebound.ParticleNotFound as error:
                pass

            # === CHECKS FOR CLOSE ENCOUNTERS ===
            #     # Stop sim if separation within 5*r_hill
            #     r_hill = get_hill_radius(m_vals[j], a_vals[j], m_vals[j+1], a_vals[j+1], m_star)
            #     if np.abs(a_vals[j] - a_vals[j+1]) < 5*r_hill:
            #         completed_sim = False
            #         break
                    
            #     # Also stop sim if planets crossed each other(P_ratio < 1)
            #     if planets[j+1].P / planets[j].P < 1:
            #         completed_sim = False
            #         break
                
        print(f"Step {i} of {len(stage_times)}", end="\r", flush=True)
        sim.dt = np.min([rock.P / 20 for rock in alive_rocks]) # 1/20 of innermost particle 
        sim.integrate(t)    
        
        # Stop sim if planet b is gone       
        if np.isnan(hist["a"][i, 0]):
            completed_sim = False
            break

        # Stop simulation early if failed
        if not completed_sim:
            break
    
    # Convert to df
    stage_data = {}
    for j, name in enumerate(rock_names):
        stage_data[name] = pd.DataFrame({
            "time": hist["time"],
            "a": hist["a"][:, j],
            "e": hist["e"][:, j],
            "inc": hist["inc"][:, j],
            "P": hist["P"][:, j],
            "P_ratio": hist["P_ratio"][:, j],
            "l": hist["l"][:, j],
            "pomega": hist["pomega"][:, j],
            "tau_a": hist["tau_a"][:, j],
            "tau_e": hist["tau_e"][:, j],
            "tau_i": hist["tau_i"][:, j],
        })
    
    return stage_data, completed_sim

def simulate_system(sim_id, file_path, rock_names, parameters, years=None, integrator="whfast"):
    """Creates a REBOUND simulation, runs the simulation, and saves it to disk.

    Args:
        sim_id (int): Simulation ID.
        file_path (str): Name of the file path for data storage.
        rock_names (list[str]): Names of the rocks (non-stellar particles).    
        parameters (dict): Dictionary containing planet and star parameters.
        years (float, optional, defaults to None): Number of years to integrate the simulation.
            If none is given, will use 0.8*tau_a of the last planet and clip within (30kyr, 10Myr).
        integrator (str, optional, defaults to whfast): Name of the REBOUND integrator to use.
    """    
    m_vals, m_star, r_vals, r_star, a_vals = parameters["m_vals"], parameters["m_star"], parameters["r_vals"], parameters["r_star"], parameters["a_vals"]
    ide_position, ide_width, Sigma_1au, h_1au = parameters["ide_position"], parameters["ide_width"], parameters["Sigma_1au"], parameters["h_1au"]
    num_pl, num_em, num_ptsml = parameters["num_pl"], parameters["num_em"], parameters["num_ptsml"]
    
    # Create the simulation
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.integrator = integrator
    
    # Add the star
    sim.add(m=m_star, r=r_star, hash='star')
    num_rocks = len(rock_names)
    
    # Add rocks 
    hash_to_name = {}
    for i in range(num_rocks):
        sim.add(m=m_vals[i], r=r_vals[i], a=a_vals[i], hash=rock_names[i], primary=sim.particles[0])
        
        # Sync hashes to names
        h = int(sim.particles[-1].hash.value)
        hash_to_name[h] = rock_names[i]
    
    sim.N_active = num_pl + num_em # Massive particles which mutually interact gravitationally
    sim.testparticle_type = 1 # Ptsmls will not interact with each other
        
    # === Collision Handling ===
    sim.collision = "direct"
    
    # Tracking stats
    collision_log = []
    particle_fate = {name: "alive" for name in rock_names}
    accretion_stats = {
        name: {
            "embryos": 0,
            "ptsmls": 0,
            "mass": 0.
        }
        for name in rock_names
    }
    
    def collision_resolve(sim_pointer, collided_particles_index):
        sim = sim_pointer.contents
        ps = sim.particles

        i = collided_particles_index.p1
        j = collided_particles_index.p2

        # Determine survivor (bigger particle) and victim (smaller particle)
        if ps[i].m >= ps[j].m:
            survivor_idx = i
            victim_idx = j
            remove_code = 2
        else:
            survivor_idx = j
            victim_idx = i
            remove_code = 1

        survivor_name = hash_to_name[int(ps[survivor_idx].hash.value)]
        victim_name = hash_to_name[int(ps[victim_idx].hash.value)]

        # Log collision
        collision_log.append({
            "time": sim.t,
            "survivor": survivor_name,
            "victim": victim_name,
            "victim mass": ps[victim_idx].m,
        })

        # Track fate
        particle_fate[victim_name] = f"accreted by {survivor_name}"

        # Track accretion
        if "embryo" in victim_name:
            accretion_stats[survivor_name]["embryos"] += 1
            sim.N_active -= 1
        elif "ptsml" in victim_name:
            accretion_stats[survivor_name]["ptsmls"] += 1

        accretion_stats[survivor_name]["mass"] += ps[victim_idx].m

        # Merge
        ps[survivor_idx] = (ps[survivor_idx] * ps[survivor_idx].m + ps[victim_idx] * ps[victim_idx].m) / (ps[survivor_idx].m + ps[victim_idx].m)
        ps[survivor_idx].m = ps[survivor_idx].m + ps[victim_idx].m
        ps[survivor_idx].r = (ps[survivor_idx].r**3 + ps[victim_idx].r**3)**(1/3)

        return remove_code
    
    sim.collision_resolve = collision_resolve

    # Move to center of momentum
    sim.move_to_com()
    ps = sim.particles
    rocks = ps[1:] # for easier indexing; ps[0] = planet b

    rebx = reboundx.Extras(sim)
    mof = rebx.load_force("modify_orbits_forces")
    rebx.add_force(mof)
    
    mof.params["ide_position"] = ide_position
    mof.params["ide_width"] = ide_width

    if years is None: # Set to 0.8*tau_a of the last planet
        years = -0.8*get_tau(rocks[num_pl-1], rock_names[num_pl-1], Sigma_1au, h_1au)[0]

    # Clip
    if years < 1e3 or years > 1e6:
        years = np.clip(years, 1e3, 1e6)
        print(f"Years clipped to {years:.3e}")
        
    print(f"Sim {sim_id:<2d} | {years:.3g} years", flush=True)
    data, complete_sim = integrate_sim(sim, rocks, rock_names, parameters, years, start_time=0)
    print(f"Sim complete? {complete_sim}")
    
    # Save data
    save_simulation_run(data, sim_id, file_path, 
                        sim_metadata={"num_rocks": num_rocks, 
                                      "rock_names": rock_names, 
                                      "collision_log": collision_log,
                                      "particle_fate": particle_fate,
                                      "accretion_stats": accretion_stats} | parameters)
 
# === SIM SETUP ===       

num_pl = 3
num_em = 2
num_ptsml = 2

rock_names = ['planet b', 'planet c', 'planet d'] + [f"embryo {i}" for i in range(num_em)] + [f"ptsml {i}" for i in range(num_ptsml)]
dataset_id = 2
n_sims = 1

def run_sim(sim_id):
    # Different rng for each sim
    rng = np.random.default_rng(seed=sim_id + os.getpid())
    
    # Set where to save the data
    base_dir = Path.cwd()
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"
    
    # TRAPPIST-1 params: m_vals, r_vals, m_star, r_star, initial_P_ratios = generate_params(planet_names, rng)
    # ALTERNATIVELY: Specify values below: 
    
    # === PARAMETERS ===
    # Planets take Kepler-223 b/c/d values
    m_vals = np.array([6.6, 4.5, 7.1] + [0.03]*num_em + [0.0033]*num_ptsml) * m_earth
                    # Younger planet is puffier
    r_vals = np.array(2*[3.0, 3.4, 5.2] + [0.3]*(num_em) + [(100*1e5/AU)/r_earth]*num_ptsml) * r_earth
    m_star = 1.04 # Msun
    r_star = 1.52 * r_sun
    a_vals = np.concatenate(([2, 3.2, 4.85], np.linspace(0.55, 1.45, num_em), np.linspace(0.5, 1.5, num_ptsml))) # Initial a_vals
    ide_position = 0.2 # a bit above where K-223b sits
    ide_width = 0.02
    
    Sigma_1au = 1700 * np.tile(np.logspace(-1, 1, num=5), 5)[sim_id] # Each row is the same
    
    # K_factor = np.repeat(np.logspace(1.5, 2.5, num=5), 5)[sim_id] # Each column is the same
    # K_factor = 100
    # alpha = 1.5
    # h_1au = ((2.7+1.1*alpha) / 0.780 * K_factor)**(-1/2)
    
    h_1au = np.repeat(np.logspace(-2, -1, num=5), 5)[sim_id] # Each column is the same
    
    Sigma_1au = 1000
    h_1au = 0.03
        
    parameters = {"m_vals": m_vals,
                  "m_star": m_star,
                  "r_vals": r_vals,
                  "r_star": r_star,
                  "a_vals": a_vals,
                  "num_pl": num_pl,
                  "num_em": num_em,
                  "num_ptsml": num_ptsml,
                  "ide_position": ide_position,
                  "ide_width": ide_width,
                  "Sigma_1au": Sigma_1au,
                  "h_1au": h_1au
                }
    
    # Sim integration!
    outcome = simulate_system(sim_id, file_path, rock_names, parameters, integrator="trace")
    return (sim_id, m_vals, r_vals, m_star, r_star)
    
if __name__ == "__main__":
    dataset_dir = Path.cwd().parent / "sim_results" / f"dataset{dataset_id}"
    
    # Create the folder
    dataset_dir.mkdir(parents=True, exist_ok=True) # change to False to be safe
    print(f"Directory: {dataset_dir}")

    print(f"Dataset: {dataset_id}")
    tstart = time()

    # === MULTIPROCESSING ===    
    multiprocess = False
    
    if multiprocess:
        # Start a local Dask cluster
        n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        print(f"CPUs: {n_cpus}")
        cluster = LocalCluster()    
        client = Client(cluster)
        
        print(f"Running sims on {len(client.scheduler_info()['workers'])} workers")
        print(f"Dask dashboard: {client.dashboard_link}")

        try:
            # Submit all simulations as Dask futures
            futures = [client.submit(run_sim, sim_id) for sim_id in range(n_sims)]
            
            # Gather results (blocks until all futures are complete)
            outcomes = client.gather(futures)
        finally:
            client.close()
            cluster.close()
    else: # Don't use Dask, do one sim
        sim_id = 0
        outcomes = [run_sim(sim_id)]
    
    # Save the outcomes
    outcome_file = f"../sim_results/dataset{dataset_id}/outcomes.pkl"
    with open(outcome_file, "wb") as f:
        pkl.dump(outcomes, f)
        print(f"Saved to {outcome_file}")
    
    # Load to verify
    with open(outcome_file, "rb") as f:
        sim_outcomes = pkl.load(f)
    
    print(f'Time elapsed: {np.round(time()-tstart)} sec')
    
# To run, use python3 -W ignore rebound_sims.py