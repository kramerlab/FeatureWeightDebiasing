NUMBER_OF_REPETETIONS=3
SAMPLE_WEIGHTING_METHOD=fw-sampling-mrs
BIAS_FRACTION=0.1

for VALIDATION_METHOD in random_forest decision_tree
do
    DROP=10
    for BIAS_TYPE in less_negative_class less_positive_class mean_difference
    do
        for DATASET in breast_cancer loan_prediction
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name feature_weight_budget_comparison --drop $DROP \
        --validation_method both --bias_fraction $BIAS_FRACTION
        done
    done

    DROP=25
    for BIAS_TYPE in less_negative_class less_positive_class mean_difference
    do
        for DATASET in hr_analytics folktables_income folktables_employment
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name feature_weight_budget_comparison --drop $DROP \
            --validation_method both --bias_fraction $BIAS_FRACTION
        done
    done
done