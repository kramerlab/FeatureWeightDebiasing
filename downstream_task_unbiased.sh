N_CV_REPEATS=10
N_CV_SPLITS=5

DROP=1
for BIAS_TYPE in none
do
    for SAMPLE_WEIGHTING_METHOD in uniform 
    do
        for DATASET in breast_cancer loan_prediction folktables_income hr_analytics folktables_employment
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --experiment_name downstream_task --bias_fraction 0.1 \
                --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS  --load_previous_results
        done
    done
done


for SAMPLE_WEIGHTING_METHOD in uniform
do
    python src/weighting_experiment.py --dataset gbs_allensbach  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
    --bias_type none --experiment_name downstream_task --drop $DROP --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS --load_previous_results
done