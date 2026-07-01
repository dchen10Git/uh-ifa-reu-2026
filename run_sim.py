# === IMPORTS ===
import numpy as np
from astropy import units as u
from pathlib import Path
from time import time
from dask.distributed import Client, LocalCluster
from scipy.stats import loguniform

import sys
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

def run_sim(dataset_id, sim_id):         
    # Set where to save the data
    base_dir = Path.cwd()
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"
    
    # === PARAMETERS ===
    planets = {
        "name": ['planet b', 'planet c', 'planet d'],
        "m_vals": [5, 5, 5], # [m_earth]
        "a_vals": [1, 1, 1], # [AU]
        "r_vals": [10, 10, 10] # [r_earth]; twice the current values
    }

    num_pl = 3
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
    
    # Full parameter space
    Sigma_1au = np.tile(np.logspace(2.6, 4, num=30), 30)[sim_id] # Each row is the same
    h_1au = np.repeat(np.logspace(-1.7, -1, num=30), 30)[sim_id] # Each column is the same
    
    # # Zoomed-in, random
    # rng = np.random.default_rng(sim_id)
    # Sigma_1au = loguniform.rvs(10**(454/145), 10**(517/145), random_state=rng)
    # h_1au = loguniform.rvs(10**(-437/290), 10**(-187/145), random_state=rng)
    
    alpha = 1
    beta = 0
    ide_width = ide_position * h_1au**beta # scale height at ide position
    
    pebble_flux = 0/1000 # number per year
                                              # Converted to Msun/AU^2 from g/cm^2
    tau_a = 1/(2.7+1.1*alpha) / m_vals[0] / (Sigma_1au*AU**2 / Msun) * h_1au**2 / (2*np.pi) # for a = 1
    tau_pl = tau_a # planet formation timescale (set to tau_a, or tau_a/1000 for planets only)
    years = 6*tau_a # Set to 6*tau_a of the first planet (or tau_a for planets only)
    n_out = 100
    
    zeta = 1

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
                  "zeta": zeta,
                  "pebble_flux": pebble_flux,
                  "m_em": m_em,
                  "r_em": r_em,
                  "m_ptsml": m_ptsml,
                  "r_ptsml": r_ptsml,
                  "tau_pl": tau_pl,
                  "years": years
                } 
    
    # Sim integration!
    try:
        reb_sims.simulate_system(sim_id, file_path, rock_names, parameters, years, n_out, print_step=True, integrator="trace")
    except Exception as e:
        print(f"Sim {sim_id} | Unexpected error: {e}")
        raise # Use this for debug traceback
        return # Allow continuation of other sims
    
if __name__ == "__main__":
    dataset_id = 8
    
    # Job number passed from terminal line (or sbatch)
    job_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    sims_per_job = 1
    start_sim = job_id * sims_per_job
    end_sim = start_sim + sims_per_job
    
    dataset_dir = Path.cwd().parent / "sim_results" / f"dataset{dataset_id}"
    
    # Create the folder
    dataset_dir.mkdir(parents=True, exist_ok=True) # change to False to be safe
    print(f"Directory: {dataset_dir}")

    print(f"Dataset: {dataset_id}")
    tstart = time()

    # === MULTIPROCESSING ===    
    # Start a local Dask cluster
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"CPUs: {n_cpus}")
    cluster = LocalCluster()    
    client = Client(cluster)
    
    futures = [client.submit(run_sim, dataset_id, sim_id) for sim_id in range(start_sim, end_sim)]
    client.gather(futures)

    client.close()
    cluster.close()
    
    print(f'Time elapsed: {int(time()-tstart)} sec')
    
# To run, use python3 -W ignore run_sim.py