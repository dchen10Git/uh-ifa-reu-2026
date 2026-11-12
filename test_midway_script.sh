#!/bin/bash
#SBATCH --job-name=test%a
#SBATCH --output=sbatch_test%a.out
#SBATCH --time=0:10:00
#SBATCH --mem-per-cpu=2048
#SBATCH --cpus-per-task=1
#SBATCH --account=pi-fabrycky
#SBATCH --array=0
echo "Script started."
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

module load python
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rebound-env

echo "Environment for rebound sims activated."

python3 test_run_sim.py ${SLURM_ARRAY_TASK_ID}

echo "Finished running."