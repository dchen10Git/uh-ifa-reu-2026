#!/bin/bash
#SBATCH --job-name=ptsmls
#SBATCH --output=sbatch.out
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=10
#SBATCH --account=pi-fabrycky
#SBATCH --mem-per-cpu=4G

echo Script started.

module avail Anaconda3

echo Loaded Anaconda3.

source activate myenv

echo Environment activated.

/home/dchen10/.conda/envs/myenv/bin/python3 run_sim.py

echo Finished running.
