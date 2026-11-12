#!/bin/bash
#SBATCH --job-name=newCd
#SBATCH --output=sbatch.out
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --cpus-per-task=25
#SBATCH --account=pi-fabrycky
#SBATCH --array=5
echo "Script started."
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

module load python/anaconda-2025.12

source activate rebound-env

echo "Environment for rebound sims activated."

python3 run_sim.py ${SLURM_ARRAY_TASK_ID}

echo "Finished running."