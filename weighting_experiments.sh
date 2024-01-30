NUMBER_OF_REPETETIONS=50

# for BIAS_TYPE in mean_difference less_negative_class less_positive_class 
for BIAS_TYPE in mean_difference 
do
    for SAMPLE_WEIGHTING_METHOD in uniform psa kmm fw-mrs soft-mrs mrs
    do
        for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS
        done
    done
done