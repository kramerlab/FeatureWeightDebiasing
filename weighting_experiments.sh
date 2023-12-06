NUMBER_OF_REPETETIONS=3

for BIAS_TYPE in less_negative_class less_positive_class mean_difference
do
    for FEATURE_WEIGHTING_METHOD in logistic_regression random_forest random uniform mutual_information
    do
        for SAMPLE_WEIGHTING_METHOD in uniform psa kmm soft-mrs mrs
        do
            for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment
            do
                python src/weighting_experiment.py --dataset $DATASET --feature_weighting_method $FEATURE_WEIGHTING_METHOD \
                --sample_weighting_method $SAMPLE_WEIGHTING_METHOD --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS
            done
        done
    done
done