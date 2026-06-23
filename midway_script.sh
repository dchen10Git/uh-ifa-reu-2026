#!/bin/bash
#SBATCH --job-name=ptsmls
#SBATCH --output=sbatch0.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=25
#SBATCH --account=pi-fabrycky

echo Script started.

module avail Anaconda3

echo Loaded Anaconda3.

source activate myenv

echo Environment activated.

/home/dchen10/.conda/envs/myenv/bin/python3 run_sim.py

echo Finished running.



