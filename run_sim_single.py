# === IMPORTS ===
from pathlib import Path
from time import time
import sys
import warnings
warnings.filterwarnings('ignore')
from run_sim import run_sim

# === RUNNING THE SIM ===
assert len(sys.argv) == 3
dataset_id = sys.argv[1]
sim_id = int(sys.argv[2])

dataset_dir = Path.cwd().parent / "sim_results" / f"dataset{dataset_id}"
    
# Create the folder
dataset_dir.mkdir(parents=True, exist_ok=True) # change to False to be safe
print(f"Directory: {dataset_dir}")

print(f"Dataset: {dataset_id}")
tstart = time()    

run_sim(dataset_id, sim_id)

print(f'Time elapsed: {int(time()-tstart)} sec')

# To run, use python3 -W ignore run_sim_single.py <dataset_id> <sim_id>