NUMBER_OF_REPETETIONS=50

# for BIAS_TYPE in less_negative_class less_positive_class mean_difference
for BIAS_TYPE in mean_difference
do
    for METHOD in  logistic_regression random_forest random uniform
    do
        # for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment 
        for DATASET in breast_cancer 
        do
            python src/feature_weighting_experiment.py --dataset $DATASET --method $METHOD \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS
        done
    done
done