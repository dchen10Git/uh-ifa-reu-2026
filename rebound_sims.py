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
def get_tau(rock, rock_type):
    '''Calculates damping timescales given parameters for one rock.
    Based on Izidoro 2014.
    Note that only planets and planetesimals should feel migration
    Negative tau indicates damping for modify_orbits_forces.
    '''
    if rock_type == 'planet':
        tau_a_earth = 2e5
        tau_a = tau_a_earth * (m_earth / rock.m)
        return -tau_a, -10000, -10000
    
    elif rock_type == 'embryo':
        m_p = rock.m
        r = rock.d # distance to the star
        a_p = rock.a
        e = rock.e
        inc = rock.inc
        Omega_k = 1/rock.P # Keplerian orbital frequency
            
        alpha = 1.5
        
        Sigma_1g = 3400 * AU**2 / Msun # Converted to Msun/AU^2 from g/cm^2
        
        Sigma_g = Sigma_1g * r**(-3/2)
        
        h1 = 0.047
        h = h1 * r**1.25
        
        t_m = (2/(2.7+1.1*alpha)) * (1/m_p) * (1/(Sigma_g*a_p**2)) * (h/r)**2 * ((1+(e*r/(1.3*h)**5)) / (1-(e*r/(1.1*h)**4))) / Omega_k
        t_wave = (1/m_p) * (1/(Sigma_g*a_p**2)) * (h/r)**4 / Omega_k
        t_e = (t_wave/0.780) * (1 - 0.14*(e/(h/r)**2 + 0.06*(e/(h/r))**3) + 0.18*(e/(h/r))*(inc/(h/r))**2)
        t_i = (t_wave/0.544) * (1 - 0.3*(inc/(h/r)**2 + 0.24*(inc/(h/r))**3) + 0.14*(e/(h/r))**2*(inc/(h/r)))
            
        return -t_m, -t_e, -t_i # Negative so damping

# === HUANG IDE MODEL ===

# def f_functions(r, r_c, Delta, A_a, A_e):
#     # Piecewise functions f_a and f_e
#     conditions = [
#         r < r_c - Delta,
#         (r_c - Delta <= r) & (r < r_c),
#         (r_c <= r) & (r < r_c + Delta + 1 / A_a),
#         r >= r_c + Delta + 1 / A_a
#     ]

#     f_a = [
#         0,          
#         A_a * (r_c - Delta - r) / Delta,
#         (r-r_c)* (A_a + 1) / (Delta + 1/A_a) - (A_a), # modified to make it continuous, paper might be wrong
#         1
#     ]

#     f_e = [
#         0,          
#         A_e * (r - r_c + Delta) / Delta,
#         (A_e - 1) * (r_c + Delta + 1 / A_a - r) / (Delta + 1 / A_a) + 1, 
#         1
#     ]

#     f_a_vals = np.select(conditions, f_a, default=np.nan)
#     f_e_vals = np.select(conditions, f_e, default=np.nan)
#     return f_a_vals, f_e_vals

# def get_taus_huang(m_vals, r_vals, m_star, current_a_vals, tau_a_earth, r_earth, q_earth, q_vals, Q_sim, C_e):
#     '''
#     Computes damping timescales based on current semimajor axis values.
    
#     Parameters:
#         current_a_vals: 1D NumPy array of current semimajor axis values.
    
#     Returns:
#         tau_a: semimajor axis damping timescale.
#         tau_e: eccentricity damping timescale.
#     '''
#     f_a_vals, f_e_vals = f_functions(current_a_vals)
#     tau_a = -tau_a_earth * (q_earth / q_vals) / f_a_vals # negative so damping.
#     tau_e_disk = C_e * tau_a * f_a_vals / f_e_vals # I removed a h^2 here
#     tau_e_star = 7.63e5 * Q_sim * (m_vals/m_earth) * (1/m_star)**1.5 * (r_earth/r_vals)** 5 * (current_a_vals/0.05)**6.5
#     tau_e = (tau_e_disk * tau_e_star) / (tau_e_disk + tau_e_star) # combining the two based on Eqs. 4 and 13
#     return tau_a, tau_e

# def simulate_trappist1_huang(m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor, planet_names, sim_id, file_path, integrator="whfast", test=False):
#     '''
#     Given initial parameters. planet_names, sim_id, file_path, and integrator,
#     simulates the TRAPPIST-1 system, saves the data, and returns list depending 
#     on the outcome. 
#     '''
#     # Create the simulation
#     sim = rebound.Simulation()
#     sim.units = ('AU', 'yr', 'Msun')
#     sim.integrator = integrator

#     # Add the star
#     sim.add(m=m_star, r=r_star)
    
#     q_vals = m_vals / m_star
#     q_earth =  3.003e-6 / m_star

#     num_planets = len(m_vals)
    
#     # Define initial eccentricities (e)
#     e_vals = np.zeros_like(m_vals)

#     # Draw initial mean anomalies (M)
#     M_vals = np.zeros_like(m_vals)

#     # Initial semimajor axis of b
#     a_b = 0.05

#     # Define initial periods (P) and semimajor axes (a)
#     P_vals = [(a_b**3 / m_star)**(1/2)]
#     for i in range(num_planets-1):
#         P_vals = np.append(P_vals, P_vals[i] * initial_P_ratios[i])
        
#     a_vals = (P_vals**2 * m_star)**(1/3)

#     # Add planets 
#     for i in range(num_planets):
#         sim.add(m=m_vals[i], r=r_vals[i], a=a_vals[i], e=e_vals[i], M=M_vals[i])

#     # Move to center of momentum
#     sim.move_to_com()
#     ps = sim.particles
#     planets = ps[1:] # for easier indexing
    
#     initial_tau_a_vals = get_taus(a_vals, m_vals, m_star)[0]
#     print(f"tau_a values: {np.round(initial_tau_a_vals)} yr \n")
    
#     years = np.clip(2*initial_tau_a_vals[-1], 30000, 10000000) # Integrate for tau_a of the last planet (Keller does 3*tau_a), with 
#                                                                # lower limit 30 kyr and upper limit 10 Myr
#     # print(f"Integrating {years/1000:.4} kyrs \n")
    
#     if test: # Short simulation for testing purposes
#         years = 100
    
#     # Code using modify_orbits_forces
#     rebx = reboundx.Extras(sim)

#     # Planet-disk interaction
#     mof = rebx.load_force("modify_orbits_forces")
#     rebx.add_force(mof)

    
#     # Huang & Ormel (2022) used positions r_c in [0.013 - 0.030 au] with width 
#     # Delta = 2hr_c = 0.06 r_c. In particular, r_c = 0.023 worked best.
    
#     data, complete_sim = integrate_sim(sim, num_planets, planets, planet_names, m_vals, m_star, years, start_time=0)
    
#     # Save data
#     save_simulation_run(data, sim_id, file_path, sim_metadata={
#                         "num_planets": num_planets, 
#                         "planet_names": planet_names,
#                         "m_vals": m_vals,
#                         "r_vals": r_vals,
#                         "m_star": m_star,
#                         "r_star": r_star,
#                         "initial_P_ratios": initial_P_ratios,
#                         "Sigma_1au": Sigma_1au,
#                         "K_factor": K_factor,
#                         "integrator": integrator
#                         })

#     saved_sim = load_simulation_run(file_path)
#     outcome = mmr_id.res_chain_outcome(saved_sim)
#     if complete_sim:
#         return outcome
#     else:
#         return np.full_like(outcome, -1)

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
            * stage_data_df (pd.Dataframe): Simulation data.
            * completed_sim (bool): Whether the integration was fully compelted.
    '''
    m_vals, m_star, r_vals, r_star = parameters["m_vals"], parameters["m_star"], parameters["r_vals"], parameters["r_star"]
    num_pl, num_em, num_ptsml = parameters["num_pl"], parameters["num_em"], parameters["num_ptsml"]
    num_rocks = len(rock_names)
    
    # Set up times for integration & data collection
    n_out = 2000 # number of data points to collect
    stage_times = np.linspace(start_time, years+start_time, n_out, endpoint=False)  # all times to integrate over
    stage_data = {}

    for name in rock_names[:num_rocks]:
        stage_data[name] = {
            "time": stage_times.copy(),
            "a": np.full(n_out, np.nan),
            "e": np.full(n_out, np.nan),
            "P": np.full(n_out, np.nan),
            "P_ratio": np.full(n_out, np.nan),
            "l": np.full(n_out, np.nan),
            "pomega": np.full(n_out, np.nan),
            "tau_a": np.full(n_out, np.nan),
            "tau_e": np.full(n_out, np.nan),
            "tau_i": np.full(n_out, np.nan),
        }
        
    sim.random_seed = 16 # for reproducibility

    completed_sim = True

    for i, t in enumerate(stage_times): 
        
        for j, name in enumerate(rock_names):

            # Update damping timescales for planets and embryos only
            try:
                rock = sim.particles[name]
                
                if 'planet' in name:
                    tau_a, tau_e, tau_i = get_tau(rock, "planet")
                    rock.params["tau_a"] = tau_a
                    rock.params["tau_e"] = tau_e
                    rock.params["tau_inc"] = tau_i
            
                    stage_data[name]["tau_a"][i] = tau_a
                    stage_data[name]["tau_e"][i] = tau_e
                    stage_data[name]["tau_i"][i] = tau_i
                elif 'embryo' in name:
                    tau_a, tau_e, tau_i = get_tau(rock, "embryo")
                    rock.params["tau_a"] = tau_a
                    rock.params["tau_e"] = tau_e
                    rock.params["tau_inc"] = tau_i
            
                    stage_data[name]["tau_a"][i] = tau_a
                    stage_data[name]["tau_e"][i] = tau_e
                    stage_data[name]["tau_i"][i] = tau_i                    
            
                # Save data
                stage_data[name]["a"][i] = rocks[j].a
                stage_data[name]["e"][i] = rocks[j].e
                stage_data[name]["l"][i] = rocks[j].l
                stage_data[name]["pomega"][i] = rocks[j].pomega
                stage_data[name]["P"][i] = rocks[j].P  

                if j < num_pl-1: # don't record period ratio for last planet
                    stage_data[name]["P_ratio"][i] = rocks[j+1].P / rock.P # index might be wrong if planet is gone
                
                # Remove particle if planet goes into star
                if rock.a < 0.1 or rock.a > 100:
                    sim.remove(hash=name)
                    print(f"{name} removed")
            
            except:
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
                
        print(f"\rStep {i} of {len(stage_times)}")
        sim.dt = rocks[0].P / 200 # 1/200 of planet b
        sim.integrate(t)    
        
        # Prevent stop in data collection       
        if np.isnan(stage_data['planet_b']["a"][i]):
            completed_sim = False
            break

        # Stop simulation early if failed
        if not completed_sim:
            break
    
    # Convert to df
    stage_data_df = {
        name: pd.DataFrame(data)
        for name, data in stage_data.items()
    }
    
    return stage_data_df, completed_sim

def simulate_system(sim_id, file_path, rock_names, parameters, years=1000, integrator="whfast"):
    '''Creates a REBOUND simulation, runs the simulation, and saves it to disk.
    
    Parameters:
        sim_id (int): Simulation ID.
        file_path (str): Name of the file path for data storage.
        rock_names (list[str]): Names of the rocks (non-stellar particles).    
        parameters (dict): Dictionary containing planet and star parameters.
        years (float, optional, defaults to 1000): Number of years to integrate the simulation.
        integrator (str, optional, defaults to whfast): Name of the REBOUND integrator to use.
    '''
    m_vals, m_star, r_vals, r_star, a_vals = parameters["m_vals"], parameters["m_star"], parameters["r_vals"], parameters["r_star"], parameters["a_vals"]
    
    # Create the simulation
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.integrator = integrator
    
    # Add the star
    sim.add(m=m_star, r=r_star, hash='star')
    num_rocks = len(rock_names)
        
    # Add planets 
    for i in range(num_rocks):
        sim.add(m=m_vals[i], r=r_vals[i], a=a_vals[i], hash=rock_names[i])

    # Move to center of momentum
    sim.move_to_com()
    ps = sim.particles
    rocks = ps[1:] # for easier indexing; ps[0] = planet b

    rebx = reboundx.Extras(sim)
    mof = rebx.load_force("modify_orbits_forces")
    rebx.add_force(mof)

    # Failsafe
    if years < 1000:
        years = 1000
        print("Years clipped to 1000")
        
    print(f"Sim {sim_id:<2d} | {years:.3g} years", flush=True)
    data, complete_sim = integrate_sim(sim, rocks, rock_names, parameters, years, start_time=0)
    print(f"Sim complete? {complete_sim}")
    
    # Save data
    save_simulation_run(data, sim_id, file_path, sim_metadata={"num_rocks": num_rocks, "rock_names": rock_names} | parameters)
 
# === SIM SETUP ===       

num_pl = 3
num_em = 10
num_ptsml = 20

rock_names = ['planet_b', 'planet_c', 'planet_d'] + [f"embryo_{i}" for i in range(num_em)] + [f"ptsml_{i}" for i in range(num_ptsml)]

dataset_id = 0
n_sims = 1

def run_sim(sim_id):
    # Different rng for each sim
    rng = np.random.default_rng(seed=sim_id + os.getpid())
    
    # Set where to save the data
    base_dir = Path.cwd()
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"
    
    # Get random param values
    # TRAPPIST-1: m_vals, r_vals, m_star, r_star, initial_P_ratios = generate_params(planet_names, rng)
    # ALTERNATIVELY: Specify values below: 
    
    m_vals = np.array([4, 5, 6] + [0.01]*num_em + [0]*num_ptsml) * m_earth
    r_vals = np.array([3, 3, 3] + [0.1]*(num_em+num_ptsml)) * r_earth
    m_star = 1.
    r_star = 0.1
    a_vals = np.concatenate(([4, 4.94, 6.11], np.arange(1, 3, 0.2), np.arange(1.05, 3.05, 0.1))) # Initial a_vals
        
    parameters = {"m_vals": m_vals,
                  "m_star": m_star,
                  "r_vals": r_vals,
                  "r_star": r_star,
                  "a_vals": a_vals,
                  "num_pl": num_pl,
                  "num_em": num_em,
                  "num_ptsml": num_ptsml
                }
    
    # Sim integration!
    years = 80000
    outcome = simulate_system(sim_id, file_path, rock_names, parameters, years=years, integrator="trace")
    return (sim_id, m_vals, r_vals, m_star, r_star)
    
# === MULTIPROCESSING ===    

if __name__ == "__main__":
    dataset_dir = Path.cwd().parent / "sim_results" / f"dataset{dataset_id}"
    
    # Create the folder
    dataset_dir.mkdir(parents=True, exist_ok=True) # change to False to be safe
    print(f"Created directory: {dataset_dir}")

    print(f"Dataset: {dataset_id}")
    tstart = time()

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