#!/bin/bash
#SBATCH --job-name=newCd
#SBATCH --output=sbatch_%a.out
#SBATCH --error=sbatch_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --cpus-per-task=32
#SBATCH --account=pi-fabrycky
#SBATCH --array=0-99
echo "Script started."
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

module load python 2>/dev/null # Ignore warnings with module loading
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rebound-env

echo "Environment for rebound sims activated."
echo "Python path: $(which python)"

N_TASKS=${SLURM_ARRAY_TASK_COUNT:-$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))} # fallback if the count var is unset
echo "Task ${SLURM_ARRAY_TASK_ID} of ${N_TASKS}"

python3 -W ignore run_sim.py --ids-file sim_ids_to_run.txt --job-id ${SLURM_ARRAY_TASK_ID} --n-jobs ${N_TASKS}

echo "Finished running."