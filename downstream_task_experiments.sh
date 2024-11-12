N_CV_REPEATS=10
N_CV_SPLITS=5

DROP=1
for BIAS_FRACTION in 0.1 0.2 0.3
do
    for BIAS_TYPE in less_positive_class less_negative_class 
    do
        for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform fw-mrs-temperature-svm \
            soft-mrs-linear soft-mrs-exponential
        do
            for DATASET in breast_cancer loan_prediction
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name downstream_task --bias_fraction $BIAS_FRACTION \
                 --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS  --load_previous_results
            done
        done
    done
done

DROP=5
for BIAS_FRACTION in 0.1 0.2 0.3
do
    for BIAS_TYPE in less_positive_class less_negative_class 
    do
        for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform fw-mrs-temperature-svm \
            soft-mrs-linear soft-mrs-exponential
        do
            for DATASET in folktables_income hr_analytics folktables_employment
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name downstream_task --bias_fraction $BIAS_FRACTION \
              --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS --load_previous_results
            done
        done
    done
done

DROP=1
for BIAS_FRACTION in 0.8 0.9
do
    for BIAS_TYPE in mean_difference 
    do
        for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform  \
            soft-mrs-linear soft-mrs-exponential fw-mrs-temperature-svm
        do
            for DATASET in breast_cancer loan_prediction
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name downstream_task --bias_fraction $BIAS_FRACTION \
                 --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS  --load_previous_results
            done
        done
    done
done

DROP=5
for BIAS_FRACTION in 0.8 0.9
do
    for BIAS_TYPE in mean_difference
    do
        for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform fw-mrs-temperature-svm \
            soft-mrs-linear soft-mrs-exponential
        do
            for DATASET in folktables_income hr_analytics folktables_employment
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name downstream_task --bias_fraction $BIAS_FRACTION \
              --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS --load_previous_results
            done
        done
    done
done