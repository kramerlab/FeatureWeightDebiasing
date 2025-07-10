#!/bin/bash

#SBATCH -A m2_datamining
#SBATCH -p longtime 
#SBATCH -J "gbs_experiments" # gives SLURM_JOB_NAME
#SBATCH -n 1 # gives SLURM_NTASKS
#SBATCH -t 10-00 
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --array=1-6

source ~/.bashrc
conda_initialize
micromamba activate feature_weighted_mrs

N_CV_REPEATS=10
N_CV_SPLITS=5

CONFIG=gbs_experiments.config
SAMPLE_WEIGHTING_METHOD=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $CONFIG)
DATASET=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $CONFIG)
EXPERIMENT_TYPE=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $CONFIG)

srun python ../src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
--bias_type none --experiment_name $EXPERIMENT_TYPE --bias_fraction 0.0 \
--drop 1 --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS  --load_previous_results