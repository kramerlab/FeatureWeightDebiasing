#!/bin/bash

#SBATCH -A m2_datamining
#SBATCH -p longtime 
#SBATCH -J "Decomposition" # gives SLURM_JOB_NAME
#SBATCH -n 1 # gives SLURM_NTASKS
#SBATCH -t 10-00 
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --array=1-65

source ~/.bashrc
conda_initialize
micromamba activate feature_weighted_mrs

N_CV_REPEATS=50

CONFIG=downstream_task_experiment.config
BIAS_FRACTION=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $CONFIG)
BIAS_TYPE=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $CONFIG)
SAMPLE_WEIGHTING_METHOD=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $CONFIG)
DATASET=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $CONFIG)
DROP=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $CONFIG)

srun python ../src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
--bias_type $BIAS_TYPE --experiment_name decomposition --bias_fraction $BIAS_FRACTION \
--drop $DROP --n_cv_repeats $N_CV_REPEATS --load_previous_results 