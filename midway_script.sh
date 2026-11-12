#!/bin/bash
#SBATCH --job-name=newCd
#SBATCH --output=sbatch_%a.out
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --cpus-per-task=30
#SBATCH --account=pi-fabrycky
#SBATCH --array=0
echo "Script started."
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

module load python 2>/dev/null # Ignore warnings with module loading
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rebound-env

echo "Environment for rebound sims activated."
echo "Python path: $(which python)"

python3 run_sim.py ${SLURM_ARRAY_TASK_ID}

echo "Finished running."