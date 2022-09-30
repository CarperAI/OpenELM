#!/bin/bash
#SBATCH --partition=lotus_gpu
#SBATCH --account=lotus_gpu
#SBATCH --gres=gpu:1 # Request a number of GPUs
#SBATCH --time=16:00:00 # Set a runtime for the job in HH:MM:SS
#SBATCH --mem=64000 # Set the amount of memory for the job in MB.

set -e # fail fully on first line failure

# Customize this line to point to conda installation
path_to_conda="/home/users/hyper1on/miniconda3"

echo "Running on $(hostname)"

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode

    JOB_ID=$SLURM_JOB_ID

    # Just read in what was passed over cmdline
    JOB_CMD="${@}"
else
    # In array

    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    # Get the line corresponding to the task id
    JOB_CMD=$(head -n ${SLURM_ARRAY_TASK_ID} "$1" | tail -1)
fi

# Find what was passed to run_name
regexp="run_name=(\S+)"
if [[ $JOB_CMD =~ $regexp ]]
then
    JOB_OUTPUT=${BASH_REMATCH[1]}
else
    echo "Error: did not find a run_name argument"
    exit 1
fi

# Check if results exists, if so remove slurm log and skip
if [ -f  "~/elm/logs/$JOB_OUTPUT/results.json" ]
then
    echo "Results already done - exiting"
    rm "slurm-${JOB_ID}.out"
    exit 0
fi

# Check if the output folder exists at all. We could remove the folder in that case.
if [ -d  "~/elm/logs/$JOB_OUTPUT" ]
then
    echo "Folder exists, but was unfinished or is ongoing (no results.json)."
    echo "Starting job as usual"
    # It might be worth removing the folder at this point:
    # echo "Removing current output before continuing"
    # rm -r "$JOB_OUTPUT"
    # Since this is a destructive action it is not on by default
fi

# Use this line if you need to create the environment first on a machine
# ./run_locked.sh ${path_to_conda}/bin/conda-env update -f environment.yml

# Activate the environment
source ${path_to_conda}/bin/activate elm

# Train the model
srun python $JOB_CMD

# Move the log file to the job folder
mv "slurm-${JOB_ID}.out" "/home/users/hyper1on/elm/logs/${JOB_OUTPUT}/"
