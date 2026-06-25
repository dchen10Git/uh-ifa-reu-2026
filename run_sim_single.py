# === IMPORTS ===
import numpy as np
from astropy import units as u
from pathlib import Path
from time import time

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import rebound_sims as reb_sims
from run_sim import run_sim

# === UNIT CONVERSIONS ===
AU = u.AU.to(u.cm) # cm per AU
G = 4*np.pi**2 # in yr, AU, Msun
Msun = u.Msun.to(u.g) # g per Msun
yr = u.yr.to(u.s) # s per yr
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

# === RUNNING THE SIM ===       
dataset_id = 'fix'
sim_id = 404

dataset_dir = Path.cwd().parent / "sim_results" / f"dataset{dataset_id}"
    
# Create the folder
dataset_dir.mkdir(parents=True, exist_ok=True) # change to False to be safe
print(f"Directory: {dataset_dir}")

print(f"Dataset: {dataset_id}")
tstart = time()    

run_sim(sim_id)

print(f'Time elapsed: {int(time()-tstart)} sec')

# To run, use python3 -W ignore run_sim_single.py