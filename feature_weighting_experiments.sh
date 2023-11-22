NUMBER_OF_REPETETIONS=3
BIAS_TYPE=mean_difference

for METHOD in uniform logistic_regression random_forest random
do
    for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment 
    do
        python src/feature_weighting_experiment.py --dataset $DATASET --method $METHOD \
        --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS
    done
done