NUMBER_OF_REPETETIONS=3
SAMPLE_WEIGHTING_METHOD=fw-sampling-mrs
DROP=15

for TRANSFORMATION_METHOD in temperature budget 
do
    # for BIAS_TYPE in less_negative_class less_positive_class mean_difference 
    for BIAS_TYPE in less_negative_class less_positive_class  
    do
        for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name feature_weight_budget_comparison --drop $DROP \
            --transformation_method $TRANSFORMATION_METHOD
        done
    done
done