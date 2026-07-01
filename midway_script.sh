#!/bin/bash
#SBATCH --job-name=ptsmls7
#SBATCH --output=sbatch7_%a.out
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=1
#SBATCH --account=pi-fabrycky
#SBATCH --array=690-692,694-695,698-699,701-719,750-757,759-779,810-839,850-851,866-868

echo "Script started."
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

module load Anaconda3

source activate myenv

echo "Environment activated."

python3 run_sim.py ${SLURM_ARRAY_TASK_ID}

echo "Finished running."