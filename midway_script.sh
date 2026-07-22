#!/bin/bash
#SBATCH --job-name=ptsmls2
#SBATCH --output=sbatch2_%a.out
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --cpus-per-task=25
#SBATCH --account=pi-fabrycky
#SBATCH --array=1,5,9,13,17,21
echo "Script started."
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

module load Anaconda3

source activate myenv

echo "Environment activated."

python3 run_sim.py ${SLURM_ARRAY_TASK_ID}

echo "Finished running."