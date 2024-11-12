N_CV_REPEATS=10
N_CV_SPLITS=5
DROP=5

for BIAS_FRACTION in 0.1 0.2 0.3
do
    for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature fw-mrs-temperature-svm
    do
            for DATASET in folktables_income folktables_employment  
            do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type less_positive_class --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS \
            --experiment_name temperature_comparison --drop $DROP --bias_fraction $BIAS_FRACTION &
        done
    done
done