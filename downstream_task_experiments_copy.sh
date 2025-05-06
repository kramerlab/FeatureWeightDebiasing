N_CV_REPEATS=10
N_CV_SPLITS=5


DROP=5
for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature-svm 
do
    for DATASET in folktables_employment 
    do
        python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
        --bias_type less_positive_class --experiment_name decomposition --bias_fraction 0.1 \
        --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS 
    done
done