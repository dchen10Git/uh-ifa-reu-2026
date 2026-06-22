#!/bin/bash
#SBATCH --job-name=ptsmls
#SBATCH --output=sbatch91.out
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=12
#SBATCH --account=pi-fabrycky

echo Script started.

module avail Anaconda3

echo Loaded Anaconda3.

source activate myenv

echo Environment activated.

/home/dchen10/.conda/envs/myenv/bin/python3 run_sim.py

echo Finished running.



