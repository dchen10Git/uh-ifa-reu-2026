# === IMPORTS ===
import numpy as np
import pandas as pd
from astropy import units as u
from astropy import constants as const
from pathlib import Path
from time import time
from dask.distributed import Client, LocalCluster

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

# === RUNNING THE SIM ===       
dataset_id = 3
n_sims = 1

def run_sim(sim_id):
    # Different rng for each sim
    rng = np.random.default_rng(seed=sim_id + os.getpid())
    
    # Set where to save the data
    base_dir = Path.cwd()
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"
    
    # === PARAMETERS ===
    # Planets take Kepler-223 b/c/d values
    planets = {
        "name": ['planet b', 'planet c', 'planet d'],
        "m_vals": [6.6, 4.5, 7.1], # [m_earth]
        "a_vals": [1, 80, 90], # [AU]
        "r_vals": [6, 6.8, 10.4] # [r_earth]; twice the current values
    }

    num_pl = 3
    num_em = 30
    num_ptsml = 200

    rock_names = planets['name'] + [f"embryo {i}" for i in range(num_em)] + [f"ptsml {i}" for i in range(num_ptsml)]
    
    # Setting up em/ptsml disk
    m_em = 0.03 # [m_earth]
    r_em = 0.3 # [r_earth]
    m_ptsml = 0.0033 # [m_earth]
    r_ptsml = (100*1e5/AU)/r_earth # 100 km in [r_earth]
    small_body_a_vals = np.linspace(0.3, 0.9, num_em+num_ptsml) # equally spaced locations in disk range
    em_indices = np.round(np.linspace(0, len(small_body_a_vals) - 1, num=num_em)).astype(int) # picks num_em equally spaced indices
    em_a_vals = small_body_a_vals[em_indices]
    ptsml_a_vals = np.delete(small_body_a_vals, em_indices)
    
    # Combine values for planets, ems, ptsmls
    m_vals = np.array(planets['m_vals'] + [m_em]*num_em + [m_ptsml]*num_ptsml) * m_earth
                    # Younger planet is puffier
    r_vals = np.array(planets['r_vals'] + [r_em]*(num_em) + [r_ptsml]*num_ptsml) * r_earth
    m_star = 1.04 # Msun
    r_star = 1.52 * r_sun
    a_vals = np.concatenate((planets['a_vals'], em_a_vals, ptsml_a_vals)) # Initial a_vals
    
    # Gas disk parameters
    ide_position = 0.2 # a bit above where K-223b sits
    ide_width = 0.1
    
    Sigma_1au = 1700 * np.tile(np.logspace(-1, 1, num=5), 5)[sim_id] # Each row is the same
    h_1au = np.repeat(np.logspace(-2, -1, num=5), 5)[sim_id] # Each column is the same
    
    pebble_flux = 0
        
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
                  "pebble_flux": pebble_flux,
                  "m_ptsml": m_ptsml,
                  "r_ptsml": r_ptsml
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
        assert n_sims == 1
        sim_id = 17
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