#!/bin/bash
#SBATCH --job-name=ptsmls1
#SBATCH --output=sbatch1_%a.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --account=pi-fabrycky
#SBATCH --array=379-383,471-473,475,477,479,952

echo "Script started."
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

module load Anaconda3

source activate myenv

echo "Environment activated."

python3 run_sim.py ${SLURM_ARRAY_TASK_ID}

echo "Finished running."