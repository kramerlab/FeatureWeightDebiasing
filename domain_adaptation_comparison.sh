NUMBER_OF_REPETETIONS=10

DROP=1
for BIAS_FRACTION in 0.1 0.2 0.3
do
    for BIAS_TYPE in  less_positive_class less_negative_class 
    do
        for SAMPLE_WEIGHTING_METHOD in  fw-mrs-temperature mrs-forest  psa kmm uniform
        do
            for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment gbs_gesis
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name test_set --bias_fraction $BIAS_FRACTION \
                --load_previous_results --budget 0.01 --drop $DROP
            done
        done
    done
done


BIAS_FRACTION=0.4
for BIAS_TYPE in  mean_difference
do
    for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest  psa kmm uniform
    do
        for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name test_set --bias_fraction $BIAS_FRACTION \
            --load_previous_results --budget 0.01 --drop $DROP
        done
    done
done