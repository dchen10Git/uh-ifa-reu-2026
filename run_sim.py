# === IMPORTS ===
import numpy as np
from astropy import units as u
from pathlib import Path
from time import time
from dask.distributed import Client, LocalCluster

import os
import warnings
warnings.filterwarnings('ignore')

import rebound_sims as reb_sims

# === UNIT CONVERSIONS ===
AU = u.AU.to(u.cm) # cm per AU
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) # g per Msun
yr = u.yr.to(u.s) # s per yr
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

# === RUNNING THE SIM ===       
dataset_id = 0
n_sims = 900

def run_sim(sim_id):    
    # # Skip sims if needed
    if sim_id < 500:
        return
    
    # Set where to save the data
    base_dir = Path.cwd()
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"
    
    # === PARAMETERS ===
    planets = {
        "name": ['planet b', 'planet c', 'planet d'],
        "m_vals": [5, 5, 5], # [m_earth]
        "a_vals": [0.1, 0.2, 0.3], # [AU]
        "r_vals": [10, 10, 10] # [r_earth]; twice the current values
    }

    num_pl = 2
    num_em = 0
    num_ptsml = 0

    rock_names = planets['name'][:num_pl] + [f"embryo {i}" for i in range(num_em)] + [f"ptsml {i}" for i in range(num_ptsml)]
    
    # Setting up em/ptsml disk
    m_em = 0.03 # [m_earth]
    r_em = 0.3 # [r_earth]
    m_ptsml = 0.0033 # [m_earth]
    r_ptsml = (100*1e5/AU)/r_earth # 100 km in [r_earth]
    small_body_a_vals = np.linspace(0.4, 0.9, num_em+num_ptsml) # equally spaced locations in disk range
    em_indices = np.round(np.linspace(0, len(small_body_a_vals) - 1, num=num_em)).astype(int) # picks num_em equally spaced indices
    em_a_vals = small_body_a_vals[em_indices]
    ptsml_a_vals = np.delete(small_body_a_vals, em_indices)
    
    # Combine values for planets, ems, ptsmls
    m_vals = np.array(planets['m_vals'][:num_pl] + [m_em]*num_em + [m_ptsml]*num_ptsml) * m_earth
    r_vals = np.array(planets['r_vals'][:num_pl] + [r_em]*(num_em) + [r_ptsml]*num_ptsml) * r_earth
    m_star = 1. # Msun
    r_star = 1.5 * r_sun
    a_vals = np.concatenate((planets['a_vals'][:num_pl], em_a_vals, ptsml_a_vals)) # Initial a_vals
    
    # Gas disk parameters
    ide_position = 0.1 
    Sigma_1au = np.tile(np.logspace(2.6, 4, num=30), 30)[sim_id] # Each row is the same
    h_1au = np.repeat(np.logspace(-1.7, -1, num=30), 30)[sim_id] # Each column is the same
    alpha = 1
    beta = 0
    ide_width = ide_position * h_1au**beta # scale height at ide position
    
    pebble_flux = 0/1000 # number per year
                                              # Converted to Msun/AU^2 from g/cm^2
    tau_a = (2/(2.7+1.1*alpha)) / (5*m_earth) / (Sigma_1au*AU**2 / Msun) * h_1au**2 / (2*np.pi) # for a = 1
    tau_pl = tau_a/20 # planet formation timescale (set to tau_a/2, or tau_a/5 for planets only)
    years = tau_a # Set to 2.5*tau_a of the first planet (or tau_a for planets only)
    
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
                  "h_1au": h_1au,
                  "alpha": alpha,
                  "beta": beta,
                  "pebble_flux": pebble_flux,
                  "m_ptsml": m_ptsml,
                  "r_ptsml": r_ptsml,
                  "tau_pl": tau_pl,
                  "years": years
                } 
    
    # Sim integration!
    try:
        reb_sims.simulate_system(sim_id, file_path, rock_names, parameters, years=years, integrator="trace")
    except Exception as e:
        print(f"Sim {sim_id} | Unexpected error: {e}")
        # raise # Use this for debug traceback
        return # Allow continuation of other sims
    
if __name__ == "__main__":
    dataset_dir = Path.cwd().parent / "sim_results" / f"dataset{dataset_id}"
    
    # Create the folder
    dataset_dir.mkdir(parents=True, exist_ok=True) # change to False to be safe
    print(f"Directory: {dataset_dir}")

    print(f"Dataset: {dataset_id}")
    tstart = time()

    # === MULTIPROCESSING ===    
    multiprocess = True
    
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
            results = client.gather(futures)
        finally:
            client.close()
            cluster.close()
    else: # Don't use Dask, do one sim
        assert n_sims == 1
        sim_id = 18
        
        run_sim(sim_id)
    
    print(f'Time elapsed: {int(time()-tstart)} sec')
    
# To run, use python3 -W ignore run_sim.py