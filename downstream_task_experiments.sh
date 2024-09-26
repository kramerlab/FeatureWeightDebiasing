N_CV_REPEATS=2

DROP=20
for BIAS_FRACTION in 0.1 
do
    for BIAS_TYPE in less_positive_class less_negative_class 
    do
        for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform fw-mrs-svm fw-mrs-temperature-mean
        do
            for DATASET in folktables_income hr_analytics folktables_employment
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name downstream_task --bias_fraction $BIAS_FRACTION \
                --load_previous_results --drop $DROP --n_cv_repeats $N_CV_REPEATS
            done
        done
    done
done


DROP=5
for BIAS_FRACTION in 0.1 
do
    for BIAS_TYPE in less_positive_class less_negative_class 
    do
        for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest psa kmm uniform fw-mrs-svm fw-mrs-temperature-mean
        do
            for DATASET in breast_cancer loan_prediction
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --experiment_name downstream_task --bias_fraction $BIAS_FRACTION \
                --load_previous_results --drop $DROP --n_cv_repeats $N_CV_REPEATS
            done
        done
    done
done



#BIAS_FRACTION=0.4
#for BIAS_TYPE in  mean_difference
#do
#    for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature mrs-forest  psa kmm uniform fw-mrs-temperature-mean
#    do
#        for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment
#        do
#            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
#            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name downstream_task --bias_fraction $BIAS_FRACTION \
#            --load_previous_results --drop $DROP
#        done
#    done
#done