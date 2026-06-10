# === IMPORTS ===
import numpy as np
import pandas as pd

from astropy import units as u
from astropy import constants as const
from pathlib import Path
from time import time
from dask.distributed import Client, LocalCluster

import re
import os
import pickle as pkl

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

# === SIM SETUP ===       

num_pl = 3
num_em = 1
num_ptsml = 1

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
    
    Sigma_1au = 10000
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
    outcome = reb_sims.simulate_system(sim_id, file_path, rock_names, parameters, integrator="trace")
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
    
# To run, use python3 -W ignore run_sim.py