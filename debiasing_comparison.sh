NUMBER_OF_REPETETIONS=5
DROP=2
BIAS_FRACTION=0.9

for BIAS_TYPE in less_negative_class less_positive_class mean_difference
do
    for SAMPLE_WEIGHTING_METHOD in uniform psa kmm fw-sampling-mrs  mrs soft-mrs
    do
        for DATASET in breast_cancer loan_prediction 
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP --bias_fraction $BIAS_FRACTION
        done
    done
done

DROP=5
for BIAS_TYPE in less_negative_class less_positive_class mean_difference
do
    for SAMPLE_WEIGHTING_METHOD in uniform psa kmm fw-sampling-mrs  mrs soft-mrs
    do
        for DATASET in hr_analytics folktables_income folktables_employment
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP --bias_fraction $BIAS_FRACTION
        done 
    done
done