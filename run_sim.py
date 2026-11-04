# To run: python3 -W ignore run_sim.py <start_sim_id; default 0>

# === IMPORTS ===
import numpy as np
from astropy import units as u
from pathlib import Path
from dask.distributed import Client, LocalCluster
from scipy.stats import loguniform

import sys
import os
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("distributed.comm.tcp").setLevel(logging.CRITICAL)

import rebound_sims as reb_sims

# === UNIT CONVERSIONS ===
AU = u.AU.to(u.cm) # cm per AU
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) # g per Msun
yr = u.yr.to(u.s) # s per yr
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

def get_params(method, sim_id):
    if method == 'grid':
        n_sigma, n_h, n_m = 5, 6, 2  # product equals total sim count

        m_em_vals  = np.array([1e-6, 1e-5]) # [m_earth]
        h_vals     = np.logspace(np.log10(0.02), np.log10(0.10), n_h)
        sigma_vals = np.logspace(np.log10(1000), np.log10(8000), n_sigma)

        M_grid, H_grid, Sigma_grid = np.meshgrid(m_em_vals, h_vals, sigma_vals, indexing='ij')

        m_em      = M_grid.ravel()[sim_id]
        h_1au     = H_grid.ravel()[sim_id]
        Sigma_1au = Sigma_grid.ravel()[sim_id]
    
    elif method == 'manual':
        Sigma_1au, h_1au, m_em = 8000, 0.03, 1e-5
        
    elif method == 'random':
        rng = np.random.default_rng(sim_id)
        Sigma_1au = loguniform.rvs(10**(454/145), 10**(517/145), random_state=rng)
        h_1au = loguniform.rvs(10**(-437/290), 10**(-187/145), random_state=rng) 
    
    return Sigma_1au, h_1au, m_em

def check_sim_exists(dataset_id, sim_id):
    base_dir = Path.cwd()
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"
    return file_path.exists()

def run_sim(dataset_id, sim_id):         
    # Set where to save the data
    base_dir = Path.cwd()
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"
    
    # === PARAMETERS ===
    planets = {
        "name": ['planet b', 'planet c', 'planet d'],
        "m_vals": [5, 5, 5], # [m_earth]
        "a_vals": [0.11, 1, 1], # [AU]; 0.11 for IDE
        "r_vals": [10, 10, 10] # [r_earth]; younger planets are puffier
    }

    num_pl, num_em, num_ptsml = 2, 50, 0
    rock_names = planets['name'][:num_pl] + [f"embryo {i}" for i in range(num_em)] + [f"ptsml {i}" for i in range(num_ptsml)]
    
    method = 'grid' # set to grid, manual, or random
    Sigma_1au, h_1au, m_em = get_params(method, sim_id)
    
    # Setting up em/ptsml disk
    r_em = (m_em)**(1/3) # [r_earth]; assuming Earth density
    m_ptsml = 0.0033 # [m_earth]; Izidoro14 value
    r_ptsml = (m_ptsml)**(1/3) # [r_earth]; assuming Earth density
    
    small_body_a_vals = np.linspace(0.3, 0.8, num_em+num_ptsml) # equally spaced locations in disk range
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
    alpha = 1 # Surface density profile index (Sigma ~ r^-alpha)
    beta = 0 # Flaring index (h/r ~ r^beta)
    
    # tau_a for first planet               # Converted to Msun/AU^2 from g/cm^2
    tau_a = 1/(2.7+1.1*alpha) / m_vals[0] / (Sigma_1au*AU**2 / Msun) * h_1au**2 / (2*np.pi) # for a = 1
    tau_pl = 0 # planet formation timescale (set to tau_a, or 0 for planets only)
    years = 2*tau_a # Set to 3*tau_a for 1 migrating planet, 6*tau_pl for 3 migrating planets
    n_out = 500 # Minimum of 50 would be fine, increase to a few thousand for better resolution

    # Set parameters
    parameters = {
                  # Particle parameters
                  "m_vals": m_vals,
                  "m_star": m_star,
                  "r_vals": r_vals,
                  "r_star": r_star,
                  "a_vals": a_vals,
                  "num_pl": num_pl,
                  "num_em": num_em,
                  "num_ptsml": num_ptsml,
                  "m_em": m_em,
                  "r_em": r_em,
                  "m_ptsml": m_ptsml,
                  "r_ptsml": r_ptsml,
                  
                  # Gas disk parameters
                  "Sigma_1au": Sigma_1au,                       
                  "h_1au": h_1au,
                  "alpha": alpha,                               # Surface density profile index (Sigma ~ r^-alpha)
                  "beta": beta,                                 # Flaring index (h/r ~ r^beta)
                  "ide_position": 0.1,
                  "ide_width": h_1au * 0.1**beta,               # Scale height at IDE
                  
                  # Replenishment parameters
                  "pebble_flux": 0,                             # number per year; num_em/tau_a -> replenish original amount after tau_a passed
                  "stop_replenishment": None,                   # set to None for never, 3*tau_a for 3 planet case
                  "add_embryos": True,                          # set to False if we want to add planetesimals instead of embryos
                  
                  # Simulation parameters
                  "tau_pl": tau_pl,
                  "years": years,
                  "integrator": 'trace',
                  "embryos_active": False,                      # if False, will still interact with planets (but not with other embryos)
                  "end_when_no_ems": False,                     # if True, will end the sim when all embryos are accreted or ejected
                  "tau_dissipation": None,                      # set to None for no disk dissipation, or a number for disk dissipation timescale [yr] (e.g. tau_a)
                  "collisions": True,
                } 
    
    # if check_sim_exists(dataset_id, sim_id):
    #     print(f"Sim {sim_id} already exists. Skipping...")
    #     return
        
    # if m_em > 1e-5:
    #     return
    
    # Sim integration!
    try:
        reb_sims.simulate_system(sim_id, file_path, rock_names, parameters, years, n_out, print_step=True)
    except Exception as e:
        print(f"Sim {sim_id} | Unexpected error: {e}")
        # raise # Use this for debug traceback, otherwise turn off when multiprocessing
        return # Allow continuation of other sims
    
if __name__ == "__main__":
    dataset_id = "2pl_AdaptiveDrag"
    
    # Job number passed from terminal line (or sbatch)
    job_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    sims_per_job = 60
    start_sim = job_id * sims_per_job
    end_sim = start_sim + sims_per_job
    
    dataset_dir = Path.cwd().parent / "sim_results" / f"dataset{dataset_id}"
    
    # Create the folder
    dataset_dir.mkdir(parents=True, exist_ok=True) # change to False to be safe
    print(f"Directory: {dataset_dir}")

    print(f"Dataset: {dataset_id}")
    
    # === MULTIPROCESSING ===    
    # Start a local Dask cluster
    if "SLURM_CPUS_PER_TASK" in os.environ:
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        n_cpus = os.cpu_count() - 2 or 1  # PC fallback; save 2 for other processes

    print(f"CPUs in use: {n_cpus}")
    
    cluster = LocalCluster(n_workers=n_cpus, threads_per_worker=1, processes=True)
    # print("See dashboard: http://localhost:8787/status")    
    
    client = Client(cluster)
    
    futures = [client.submit(run_sim, dataset_id, sim_id) for sim_id in range(start_sim, end_sim)]
    client.gather(futures)

    client.close()
    cluster.close()
    
