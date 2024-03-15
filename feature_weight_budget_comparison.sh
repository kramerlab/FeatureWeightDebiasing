NUMBER_OF_REPETETIONS=1
SAMPLE_WEIGHTING_METHOD=fw-sampling-mrs
DROP=10

# for BIAS_TYPE in mean_difference less_negative_class less_positive_class 
for BIAS_TYPE in less_negative_class 
do
    for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment
    #for DATASET in breast_cancer
    do
        python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
        --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name feature_weight_budget_comparison --drop $DROP
    done
done