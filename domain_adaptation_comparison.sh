NUMBER_OF_REPETETIONS=3
BIAS_FRACTION=0.1


DROP=5
for BIAS_TYPE in  less_negative_class less_positive_class mean_difference
do
    for SAMPLE_WEIGHTING_METHOD in uniform psa kmm mrs
    do
        for DATASET in breast_cancer loan_prediction 
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name test_set --bias_fraction $BIAS_FRACTION
        done
    done
done

for BIAS_TYPE in  less_negative_class less_positive_class mean_difference
do
    for SAMPLE_WEIGHTING_METHOD in  fw-mrs-temperature 
    do
        for DATASET in breast_cancer loan_prediction 
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name test_set --bias_fraction $BIAS_FRACTION \
            --budget 0.01
        done
    done
done

for BIAS_TYPE in  less_negative_class less_positive_class mean_difference
do
    for SAMPLE_WEIGHTING_METHOD in  fw-mrs-budget 
    do
        for DATASET in breast_cancer loan_prediction 
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name test_set --bias_fraction $BIAS_FRACTION \
            --budget 0.05
        done
    done
done

DROP=10
for BIAS_TYPE in  less_negative_class less_positive_class mean_difference
do
    # for SAMPLE_WEIGHTING_METHOD in uniform psa kmm fw-mrs-temperature fw-mrs-budget soft-mrs mrs
    for SAMPLE_WEIGHTING_METHOD in uniform psa kmm mrs
    do
        for DATASET in hr_analytics folktables_income folktables_employment
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name test_set --bias_fraction $BIAS_FRACTION
        done
    done
done

for BIAS_TYPE in  less_negative_class less_positive_class mean_difference
do
    for SAMPLE_WEIGHTING_METHOD in  fw-mrs-temperature 
    do
        for DATASET in hr_analytics folktables_income folktables_employment
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name test_set --bias_fraction $BIAS_FRACTION \
            --budget 0.05
        done
    done
done

for BIAS_TYPE in  less_negative_class less_positive_class mean_difference
do
    for SAMPLE_WEIGHTING_METHOD in  fw-mrs-budget 
    do
        for DATASET in  hr_analytics folktables_income folktables_employment
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name test_set --bias_fraction $BIAS_FRACTION \
            --budget 0.05
        done
    done
done