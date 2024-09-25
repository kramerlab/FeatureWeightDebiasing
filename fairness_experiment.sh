DATASET=fairness_adult
SAMPLE_WEIGHTING_METHOD=fw-mrs-temperature
DROP=2
N_CV_REPEATS=10

python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                 --n_cv_repeats $N_CV_REPEATS --experiment_name fairness_task --drop $DROP 