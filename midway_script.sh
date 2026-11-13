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

# Use the env's python directly. `module load python` + `conda activate` in a batch job picked up the
# older cluster-wide anaconda (2022.05) instead of rebound-env, so don't rely on activation.
ENV_DIR=$HOME/.conda/envs/rebound-env
PYTHON=$ENV_DIR/bin/python
export PATH=$ENV_DIR/bin:$PATH

echo "Python path: $PYTHON"
$PYTHON -c "import dask, distributed, rebound; print('dask', dask.__version__, '| distributed', distributed.__version__, '| rebound', rebound.__version__)" || exit 1

N_TASKS=${SLURM_ARRAY_TASK_COUNT:-$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))} # fallback if the count var is unset
echo "Task ${SLURM_ARRAY_TASK_ID} of ${N_TASKS}"

$PYTHON -W ignore run_sim.py --ids-file sim_ids_to_run.txt --job-id ${SLURM_ARRAY_TASK_ID} --n-jobs ${N_TASKS} || exit 1 # nonzero exit so Slurm marks the task FAILED

echo "Finished running."