N_CV_REPEATS=10
N_CV_SPLITS=5
DROP=20


for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature-svm fw-mrs-temperature
do
    for BIAS_TYPE in less_positive_class 
    do
        python src/weighting_experiment.py --dataset breast_cancer  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
        --bias_type $BIAS_TYPE --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS \
        --experiment_name temperature_comparison --drop $DROP --bias_fraction 0.1 
    done
done