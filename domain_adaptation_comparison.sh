NUMBER_OF_REPETETIONS=10


DROP=5
for BIAS_FRACTION in 0.1 0.25 0.5
do
    for BIAS_TYPE in  less_negative_class less_positive_class mean_difference
    do
        for SAMPLE_WEIGHTING_METHOD in uniform psa kmm fw-mrs-temperature mrs
        do
            for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name test_set --bias_fraction $BIAS_FRACTION \
                --load_previous_results --budget 0.01
            done
        done
    done
done