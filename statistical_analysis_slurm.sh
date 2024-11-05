#!/bin/bash

#SBATCH -A m2_datamining
#SBATCH -p parallel 
#SBATCH -J "feature_weighting_cross_validation" # gives SLURM_JOB_NAME
#SBATCH -n 4 # gives SLURM_NTASKS
#SBATCH -t 7200 # <time in minutes>
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --mem=200000M 

# Store working directory to be safe
SAVEDPWD=$(pwd)
# We define a bash function to do the cleaning when the signal is caught
cleanup(){
    # Note: The following only works on single with output on the node,
    #       where the jobscript is running.
    #       For multinode output, you can use the 'sgather' command or
    #       get in touch with us, if the case is more complex.
    cp -r /localscratch/${SLURM_JOB_ID}/results ${SAVEDPWD}/results &
    wait
    exit 0
}

# Copy input file
cp -r ${SAVEDPWD}/data /localscratch/${SLURM_JOB_ID}/data
cd /localscratch/${SLURM_JOB_ID}

DROP=1
N_CV_REPEATS=50
DATASET=gbs_allensbach

for SAMPLE_WEIGHTING_METHOD in uniform fw-mrs-temperature fw-mrs-temperature-mean fw-mrs-temperature-svm
do
    python ${SAVEDPWD}/src/weighting_experiment.py --drop $DROP --n_cv_repeats $N_CV_REPEATS \
    --dataset $DATASET --experiment_name=statistical_analysis_fw --sample_weighting_method $SAMPLE_WEIGHTING_METHOD &
done

wait
# Call the cleanup function when everything went fine
cleanup