N_CV_REPEATS=10
DROP=2

for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature fw-mrs-temperature-mean
do
        for DATASET in folktables_income folktables_employment  
        do
        python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
        --bias_type less_positive_class --n_cv_repeats $N_CV_REPEATS --n_cv_splits 3 \
        --experiment_name temperature_comparison --drop $DROP --bias_fraction 0.1
    done
done