#!/bin/bash

#SBATCH -A m2_datamining
#SBATCH -p parallel 
#SBATCH -J "feature_weighting_cross_validation" # gives SLURM_JOB_NAME
#SBATCH -n 10 # gives SLURM_NTASKS
#SBATCH -t 7200 # <time in minutes>
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=200000M 

N_CV_REPEATS=50

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
                srun python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name decomposition --bias_fraction $BIAS_FRACTION \
                 --drop $DROP --n_cv_repeats $N_CV_REPEATS --load_previous_results &
            done
        done
    done
done

DROP=5
for BIAS_FRACTION in 0.1 0.2 0.3
do
    for BIAS_TYPE in less_positive_class less_negative_class 
    do
        for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform fw-mrs-temperature-mean fw-mrs-temperature-svm \
            soft-mrs-linear soft-mrs-exponential
        do
            for DATASET in folktables_income hr_analytics folktables_employment
            do
                srun python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name decomposition --bias_fraction $BIAS_FRACTION \
              --drop $DROP --n_cv_repeats $N_CV_REPEATS --load_previous_results &
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
                srun python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name decomposition --bias_fraction $BIAS_FRACTION \
                 --drop $DROP --n_cv_repeats $N_CV_REPEATS --load_previous_results &
            done
        done
    done
done

DROP=5
for BIAS_FRACTION in 0.8 0.9
do
    for BIAS_TYPE in mean_difference
    do
        for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform fw-mrs-temperature-mean fw-mrs-temperature-svm \
            soft-mrs-linear soft-mrs-exponential
        do
            for DATASET in folktables_income hr_analytics folktables_employment
            do
                srun python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name decomposition --bias_fraction $BIAS_FRACTION \
              --drop $DROP --n_cv_repeats $N_CV_REPEATS --load_previous_results &
            done
        done
    done
done