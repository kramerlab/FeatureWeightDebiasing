NUMBER_OF_REPETETIONS=5

DROP=1
for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature 
do
    for BIAS_TYPE in less_negative_class
    do
        for DATASET in gbs_gesis gbs_allensbach
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name feature_weight_budget_comparison --drop $DROP 
        done
    done
done


DROP=10
for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature
do
    for BIAS_FRACTION in 0.1 0.25 0.5
    do
        for BIAS_TYPE in less_positive_class less_negative_class mean_difference
        do
            for DATASET in folktables_income 
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name feature_weight_budget_comparison --drop $DROP \
                 --bias_fraction $BIAS_FRACTION
            done
        done
    done
done