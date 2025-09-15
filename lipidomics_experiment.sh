N_CV_REPEATS=1
N_CV_SPLITS=5

DROP=5
for SAMPLE_WEIGHTING_METHOD in uniform psa kmm fw-mrs-temperature mrs-forest fw-mrs-temperature-svm \
    soft-mrs-exponential
do
        python src/weighting_experiment.py --dataset lipidomics --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
        --experiment_name lipidomics --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS  --load_previous_results
done