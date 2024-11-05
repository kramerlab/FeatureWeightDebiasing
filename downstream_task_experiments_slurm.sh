#!/bin/bash

#SBATCH -A m2_datamining
#SBATCH -p parallel 
#SBATCH -J "feature_weighting_cross_validation" # gives SLURM_JOB_NAME
#SBATCH -n 10 # gives SLURM_NTASKS
#SBATCH -t 7200 # <time in minutes>
#SBATCH --cpus-per-task=4
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

N_CV_REPEATS=10
N_CV_SPLITS=5

DROP=1
for BIAS_FRACTION in 0.1 0.2 0.3
do
    for BIAS_TYPE in less_positive_class less_negative_class 
    do
        for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform fw-mrs-temperature-mean fw-mrs-temperature-svm \
            soft-mrs-linear soft-mrs-exponential
        do
            for DATASET in breast_cancer loan_prediction
            do
                python ${SAVEDPWD}/src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name downstream_task --bias_fraction $BIAS_FRACTION \
                 --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS  --load_previous_results &
            done
        done
    done
done

DROP=3
for BIAS_FRACTION in 0.1 0.2 0.3
do
    for BIAS_TYPE in less_positive_class less_negative_class 
    do
        for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform fw-mrs-temperature-mean fw-mrs-temperature-svm \
            soft-mrs-linear soft-mrs-exponential
        do
            for DATASET in folktables_income hr_analytics folktables_employment
            do
                python ${SAVEDPWD}/src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name downstream_task --bias_fraction $BIAS_FRACTION \
              --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS --load_previous_results &
            done
        done
    done
done

DROP=1
for BIAS_FRACTION in 0.8 0.9
do
    for BIAS_TYPE in mean_difference 
    do
        for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform fw-mrs-temperature-mean \
            soft-mrs-linear soft-mrs-exponential
        do
            for DATASET in breast_cancer loan_prediction
            do
                python ${SAVEDPWD}/src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name downstream_task --bias_fraction $BIAS_FRACTION \
                 --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS  --load_previous_results &
            done
        done
    done
done

DROP=3
for BIAS_FRACTION in 0.8 0.9
do
    for BIAS_TYPE in mean_difference
    do
        for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform fw-mrs-temperature-mean fw-mrs-temperature-svm \
            soft-mrs-linear soft-mrs-exponential
        do
            for DATASET in folktables_income hr_analytics folktables_employment
            do
                python ${SAVEDPWD}/src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name downstream_task --bias_fraction $BIAS_FRACTION \
              --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS --load_previous_results &
            done
        done
    done
done

wait
# Call the cleanup function when everything went fine
cleanup