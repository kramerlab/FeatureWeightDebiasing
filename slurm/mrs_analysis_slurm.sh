#!/bin/bash

#SBATCH -A m2_datamining
#SBATCH -p longtime 
#SBATCH -J "MRS Analysis" # gives SLURM_JOB_NAME
#SBATCH -n 1 # gives SLURM_NTASKS
#SBATCH -t 10-00 
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --array=1-4

N_CV_REPEATS=10
N_CV_SPLITS=5

source ~/.bashrc
conda_initialize
micromamba activate feature_weighted_mrs

CONFIG=mrs_analysis.config
BIAS_FRACTION=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $CONFIG)
BIAS_TYPE=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $CONFIG)
MRS_FUNCTION=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $CONFIG)
DATASET=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $CONFIG)
DROP=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $CONFIG)

srun python ../src/mrs_analysis.py --data_set_name $DATASET --bias_type $BIAS_TYPE --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS \
            --drop $DROP --bias_fraction $BIAS_FRACTION --mrs_function $MRS_FUNCTION --load_previous_results