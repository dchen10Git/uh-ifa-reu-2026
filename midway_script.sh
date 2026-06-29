#!/bin/bash
#SBATCH --job-name=ptsmls8
#SBATCH --output=sbatch8_%a.out
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=1
#SBATCH --account=pi-fabrycky
#SBATCH --array=70,71,218,219,222,226,228,230

echo "Script started."
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

module load Anaconda3

source activate myenv

echo "Environment activated."

python3 run_sim.py ${SLURM_ARRAY_TASK_ID}

echo "Finished running."