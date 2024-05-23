NUMBER_OF_REPETETIONS=25


for BIAS_FRACTION in 0.1 0.25 0.5
do
    for BIAS_TYPE in  less_negative_class less_positive_class mean_difference
    do
        for SAMPLE_WEIGHTING_METHOD in uniform psa kmm 
        do
            for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name test_set --bias_fraction $BIAS_FRACTION \
                --load_previous_results
            done
        done
    done
done

#DROP=2
#for BIAS_FRACTION in 0.1 0.25 0.5
#do
#    for BIAS_TYPE in  less_negative_class less_positive_class mean_difference
#    do
#        for SAMPLE_WEIGHTING_METHOD in  fw-mrs-temperature mrs
#        do
#            for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment
#            do
#                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
#                --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name test_set --bias_fraction $BIAS_FRACTION \
#                --budget 0.01 --load_previous_results
#            done
#       done
#   done
#done