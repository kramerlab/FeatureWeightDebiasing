N_CV_REPEATS=50

DROP=1
for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform  fw-mrs-temperature-svm 
do
    for DATASET in breast_cancer loan_prediction
    do
        python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
        --bias_type less_positive_class --experiment_name decomposition --bias_fraction 0.1 \
            --drop $DROP --n_cv_repeats $N_CV_REPEATS --load_previous_results
    done
done

DROP=5
for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform fw-mrs-temperature-svm
do
    for DATASET in folktables_income hr_analytics folktables_employment
    do
        python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
        --bias_type less_positive_class --experiment_name decomposition --bias_fraction 0.1 \
        --drop $DROP --n_cv_repeats $N_CV_REPEATS --load_previous_results
    done
done