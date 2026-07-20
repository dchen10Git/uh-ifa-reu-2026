# To run: python3 -W ignore run_sim.py <start_sim_id>

# === IMPORTS ===
import numpy as np
from astropy import units as u
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def get_params(method, sim_id):
    if method == 'grid':
        n_sigma, n_h, n_m = 30, 30, 16  # product equals total sim count

        m_em_vals  = np.logspace(np.log10(1e-8), np.log10(1e-1), n_m)
        h_vals     = np.logspace(np.log10(0.01), np.log10(0.10), n_h)
        sigma_vals = np.logspace(np.log10(170), np.log10(17000), n_sigma)

        M_grid, H_grid, Sigma_grid = np.meshgrid(m_em_vals, h_vals, sigma_vals, indexing='ij')

        m_em      = M_grid.ravel()[sim_id]
        h_1au     = H_grid.ravel()[sim_id]
        Sigma_1au = Sigma_grid.ravel()[sim_id]
    
    elif method == 'manual':
        Sigma_1au, h_1au, m_em = 170, 0.10, 1e-4
        
    elif method == 'random':
        rng = np.random.default_rng(sim_id)
        Sigma_1au = loguniform.rvs(10**(454/145), 10**(517/145), random_state=rng)
        h_1au = loguniform.rvs(10**(-437/290), 10**(-187/145), random_state=rng) 
    
    return Sigma_1au, h_1au, m_em

def run_sim(dataset_id, sim_id):         
    # Set where to save the data
    base_dir = Path.cwd()
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"
    
    # === PARAMETERS ===
    planets = {
        "name": ['planet b', 'planet c', 'planet d'],
        "m_vals": [5, 5, 5], # [m_earth]
        "a_vals": [0.11, 1, 1], # [AU]
        "r_vals": [10, 10, 10] # [r_earth]; younger planets are puffier
    }

    num_pl, num_em, num_ptsml = 2, 20, 0
    rock_names = planets['name'][:num_pl] + [f"embryo {i}" for i in range(num_em)] + [f"ptsml {i}" for i in range(num_ptsml)]
    
    method = 'grid' # set to grid, manual, or random
    Sigma_1au, h_1au, m_em = get_params(method, sim_id)
    
    # Setting up em/ptsml disk
    r_em = (m_em)**(1/3) # [r_earth]; assuming density same as Earth
    m_ptsml = 0.0033 # [m_earth]
    r_ptsml = (100*1e5/AU)/r_earth # 100 km in [r_earth]
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
    beta = 0  # Flaring index (h/r ~ r^beta)
    ide_position = 0.1
    ide_width = h_1au * ide_position**beta # scale height at ide position
    
    pebble_flux = 0/1000 # number per year
    
    # tau_a for first planet               # Converted to Msun/AU^2 from g/cm^2
    tau_a = 1/(2.7+1.1*alpha) / m_vals[0] / (Sigma_1au*AU**2 / Msun) * h_1au**2 / (2*np.pi) # for a = 1
    tau_pl = 0 # planet formation timescale (set to tau_a, or 0 for planets only)
    years = 2*tau_a # Set to 2*tau_a for 1 migrating planet, 6*tau_pl for 3 migrating planets
    n_out = 50

    if years > 2500000: # 2500 kyr (cutoff for ~35 hours of runtime on Midway)
        print(f"Skipping sim {sim_id} due to long runtime: years = {years:.2e})")
        return # Skip this sim as it takes too long to run
    
    integrator = 'trace'
    embryos_active = False # if False, will still interact with planets (but not with other embryos)
    end_when_no_ems = True
    tau_dissipation = None # set to None for no disk dissipation, or a number for disk dissipation timescale [yr] (e.g. tau_a)

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
                  "m_em": m_em,
                  "r_em": r_em,
                  "m_ptsml": m_ptsml,
                  "r_ptsml": r_ptsml,
                  "tau_pl": tau_pl,
                  "years": years,
                  "integrator": integrator,
                  "embryos_active": embryos_active,
                  "end_when_no_ems": end_when_no_ems,
                  "tau_dissipation": tau_dissipation,
                } 
    
    # Sim integration!
    try:
        reb_sims.simulate_system(sim_id, file_path, rock_names, parameters, years, n_out, print_step=False)
    except Exception as e:
        print(f"Sim {sim_id} | Unexpected error: {e}")
        # raise # Use this for debug traceback, otherwise turn off when multiprocessing
        return # Allow continuation of other sims
    
if __name__ == "__main__":
    dataset_id = 2
    
    # Job number passed from terminal line (or sbatch)
    job_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    sims_per_job = 144
    start_sim = job_id * sims_per_job
    end_sim = start_sim + sims_per_job
    
    dataset_dir = Path.cwd().parent / "sim_results" / f"dataset{dataset_id}"
    
    # Create the folder
    dataset_dir.mkdir(parents=True, exist_ok=True) # change to False to be safe
    print(f"Directory: {dataset_dir}")

    print(f"Dataset: {dataset_id}")

    # === MULTIPROCESSING ===    
    # Start a local Dask cluster
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"CPUs: {n_cpus}")
    
    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
        futures = [executor.submit(run_sim, dataset_id, sim_id) for sim_id in range(start_sim, end_sim)]
        for f in as_completed(futures):
            f.result()  # re-raises exceptions instead of silently dropping them
    
