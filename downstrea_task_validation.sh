N_CV_REPEATS=10

DROP=3
for BIAS_TYPE in less_positive_class less_negative_class 
do
    for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature fw-mrs-temperature-mean
    do
        for DATASET in folktables_income folktables_employment
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --experiment_name downstream_task --bias_fraction 0.1 \
            --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits 3 --load_previous_results
        done
    done
done
