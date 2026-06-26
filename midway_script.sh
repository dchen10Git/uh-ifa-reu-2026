#!/bin/bash
#SBATCH --job-name=ptsmls
#SBATCH --output=sbatch9_%a.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --account=pi-fabrycky
#SBATCH --array=0-17

echo "Script started."
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

module load Anaconda3

source activate myenv

echo "Environment activated."

python3 run_sim.py ${SLURM_ARRAY_TASK_ID}

echo "Finished running."