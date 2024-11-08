#!/bin/bash

#SBATCH -A m2_datamining
#SBATCH -p parallel 
#SBATCH -J "feature_weighting_cross_validation" # gives SLURM_JOB_NAME
#SBATCH -n 1 # gives SLURM_NTASKS
#SBATCH -t 5-00 # <time in minutes>
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --mem=16G 
#SBATCH --array=1-18

source ~/.bashrc
conda_initialize
micromamba activate feature_weighted_mrs

N_CV_REPEATS=10
N_CV_SPLITS=5
DROP=5
CONFIG=temperature_comparison.config
BIAS_FRACTION=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $CONFIG)
SAMPLE_WEIGHTING_METHOD=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $CONFIG)
DATASET=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $CONFIG)

srun python ../src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
--bias_type less_positive_class --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS \
--experiment_name temperature_comparison --drop $DROP --bias_fraction $BIAS_FRACTION & 