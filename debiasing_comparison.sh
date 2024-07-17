NUMBER_OF_REPETETIONS=5
DROP=2

for BIAS_FRACTION in 0.1 0.2 0.3
do
    for BIAS_TYPE in less_negative_class less_positive_class mean_difference
    do
        for SAMPLE_WEIGHTING_METHOD in uniform psa kmm fw-sampling-mrs mrs soft-mrs-linear soft-mrs-exponential
        do
            for DATASET in breast_cancer loan_prediction gbs_gesis hr_analytics folktables_income folktables_employment
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP --bias_fraction $BIAS_FRACTION
            done
        done
    done
done

DROP=5
BIAS_FRACTION =0.4
for BIAS_TYPE in  mean_difference
do
    for SAMPLE_WEIGHTING_METHOD in uniform psa kmm fw-sampling-mrs mrs soft-mrs-linear soft-mrs-exponential
    do
        for DATASET in 
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP --bias_fraction $BIAS_FRACTION
        done 
    done
done