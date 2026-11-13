# On Midway: sbatch midway_script.sh   (sims from sim_ids_to_run.txt, split across the job array)
# Locally (sims too long for Midway): python3 -W ignore run_sim.py --long
# Flags: --ids-file FILE (sim_id list; default sim_ids_to_run.txt) | --long (run only sims > MAX_YEARS instead of <=)
#        --job-id I --n-jobs N (run slice I of N; default from SLURM array env, else 0 of 1 = everything)

# === IMPORTS ===
import argparse
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

MAX_YEARS = 9e5 # sims longer than this are run locally, not on Midway

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
        n_sigma, n_h, n_m = 40, 40, 6  # product equals total sim count

        m_em_vals  = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]) # [m_earth]
        h_vals     = np.logspace(np.log10(0.01), np.log10(0.10), n_h)
        sigma_vals = np.logspace(np.log10(170), np.log10(17000), n_sigma)

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

def do_not_run_sim(dataset_id, sim_id, parameters):
    '''Returns True if we don't want the sim to be run.'''
   
    # if sim_id < 1400: # Skip first 1400 sims (already done)
    #     return True

    # if parameters['m_em'] != 1e-1:
    #     print(f"Sim {sim_id} has m_em = {parameters['m_em']}. Skipping...")
    #     return

    if parameters['years'] > MAX_YEARS: # Skip sims that take too long
        print(f"Sim {sim_id} will take longer than {MAX_YEARS/1000:.0f} kyr. Skipping...")
        return True
    
    # # Skip sims that already exist
    # if check_sim_exists(dataset_id, sim_id):
    #     print(f"Sim {sim_id} already exists. Skipping...")
    #     return True
    
    return False # we want it to run

def build_setup(sim_id):
    '''Builds everything needed to integrate a sim. Cheap (no integration), so it is
    also used to look up the sim length before deciding where to run a sim.'''
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
    years = 2*tau_a # Set to 2*tau_a for 1 migrating planet, 6*tau_pl for 3 migrating planets
    n_out = 250 # Minimum of 50 would be fine, increase to a few thousand for better resolution

    embryos_active = True if m_em == 1e-1 else False # if False, will still interact with planets (but not with other embryos)

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
                  "embryos_active": embryos_active,
                  "end_when_no_ems": True,                     # if True, will end the sim when all embryos are accreted or ejected
                  "tau_dissipation": None,                      # set to None for no disk dissipation, or a number for disk dissipation timescale [yr] (e.g. tau_a)
                  "collisions": True,
                }

    return rock_names, parameters, years, n_out

def sim_years(sim_id):
    return build_setup(sim_id)[2]

def run_sim(dataset_id, sim_id, allow_long=False):
    # Set where to save the data
    base_dir = Path.cwd()
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"

    rock_names, parameters, years, n_out = build_setup(sim_id)

    if not allow_long and do_not_run_sim(dataset_id, sim_id, parameters):
        return

    # Sim integration!
    try:
        reb_sims.simulate_system(sim_id, file_path, rock_names, parameters, years, n_out, print_step=True)
    except Exception as e:
        print(f"Sim {sim_id} | Unexpected error: {e}")
        # raise # Use this for debug traceback, otherwise turn off when multiprocessing
        return # Allow continuation of other sims
    
if __name__ == "__main__":
    dataset_id = "2pl_AdaptiveDrag"
    
    # Under sbatch, job_id/n_jobs come from the array (SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT),
    # so changing `#SBATCH --array=` is all that is needed to change the split.
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-file", default="sim_ids_to_run.txt", help="text file, one sim_id per line")
    parser.add_argument("--job-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    parser.add_argument("--long", action="store_true",
                        help="run ONLY the sims longer than MAX_YEARS (for local runs); default runs only the shorter ones")
    args = parser.parse_args()

    # Select this job's sims: filter by length, sort longest-first, deal out round-robin
    # so every job gets a similar mix of long and short sims (sim length varies by ~1000x).
    sim_ids = np.unique(np.loadtxt(args.ids_file, dtype=int, ndmin=1))
    lengths = np.array([sim_years(s) for s in sim_ids])
    keep = lengths > MAX_YEARS if args.long else lengths <= MAX_YEARS
    sim_ids, lengths = sim_ids[keep], lengths[keep]
    order = np.argsort(-lengths, kind="stable")
    my_sims = [int(s) for s in sim_ids[order][args.job_id::args.n_jobs]]
    print(f"Job {args.job_id}/{args.n_jobs}: {len(my_sims)} of {len(sim_ids)} sims "
          f"({'>' if args.long else '<='} {MAX_YEARS:.0e} yr)")
    if not my_sims:
        sys.exit(0)

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
        n_cpus = os.cpu_count() - 1 or 1  # PC fallback; save 1 for other processes

    print(f"CPUs in use: {n_cpus}")
    
    cluster = LocalCluster(n_workers=n_cpus, threads_per_worker=1, processes=True)
    # print("See dashboard: http://localhost:8787/status")    
    
    client = Client(cluster)
    
    # Priority keeps the longest sims starting first so they don't straggle at the end
    futures = [client.submit(run_sim, dataset_id, sim_id, args.long, priority=len(my_sims) - i)
               for i, sim_id in enumerate(my_sims)]
    client.gather(futures)

    client.close()
    cluster.close()
    
