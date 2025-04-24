N_CV_REPEATS=10
N_CV_SPLITS=5
DROP=5

for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature-comparison
do
    python src/weighting_experiment.py --dataset folktables_income  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
    --bias_type less_positive_class --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS \
    --experiment_name downstream_task --drop $DROP --bias_fraction 0.1
done