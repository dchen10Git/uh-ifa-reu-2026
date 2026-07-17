#!/bin/bash
#SBATCH --job-name=ptsmls4
#SBATCH --output=sbatch4_%a.out
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --account=pi-fabrycky
#SBATCH --array=0-19

echo "Script started."
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

module load Anaconda3

source activate myenv

echo "Environment activated."

python3 run_sim.py ${SLURM_ARRAY_TASK_ID}

echo "Finished running."