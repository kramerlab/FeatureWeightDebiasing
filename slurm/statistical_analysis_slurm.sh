#!/bin/bash

#SBATCH -A m2_datamining
#SBATCH -p parallel 
#SBATCH -J "feature_weighting_cross_validation" # gives SLURM_JOB_NAME
#SBATCH -n 4 # gives SLURM_NTASKS
#SBATCH -t 7200 # <time in minutes>
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --mem=200000M 

DROP=1
N_CV_REPEATS=50
DATASET=gbs_allensbach

for SAMPLE_WEIGHTING_METHOD in uniform fw-mrs-temperature fw-mrs-temperature-mean fw-mrs-temperature-svm
do
    srun python ../src/weighting_experiment.py --drop $DROP --n_cv_repeats $N_CV_REPEATS \
    --dataset $DATASET --experiment_name=statistical_analysis_fw --sample_weighting_method $SAMPLE_WEIGHTING_METHOD &
done
wait