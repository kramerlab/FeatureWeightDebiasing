N_CV_REPEATS=10
N_CV_SPLITS=5

for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest uniform fw-mrs-temperature-svm
do
    python src/weighting_experiment.py --dataset gbs_gesis  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
    --bias_type none --experiment_name downstream_task --bias_fraction 0.0 \
        --drop 1 --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS  --load_previous_results
done


python src/weighting_experiment.py --dataset gbs_gesis  --sample_weighting_method fw-mrs-temperature  \
    --bias_type none --experiment_name temperature_comparison --bias_fraction 0.0 \
        --drop 1 --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS  --load_previous_results


python src/weighting_experiment.py --dataset gbs_allensbach  --sample_weighting_method fw-mrs-temperature  \
    --bias_type none --experiment_name temperature_comparison --bias_fraction 0.0 \
        --drop 1 --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS  --load_previous_results