N_CV_REPEATS=5

DROP=2
for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature-downstream fw-mrs-svm-downstream
do
    for BIAS_FRACTION in 0.1 
    # for BIAS_FRACTION in 0.1 0.2 0.3
    do
        for DATASET in breast_cancer loan_prediction 
           # gbs_gesis
        do
            for BIAS_TYPE in less_positive_class less_negative_class
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --n_cv_repeats $N_CV_REPEATS \
                --experiment_name temperature_comparison --drop $DROP \
                --bias_fraction $BIAS_FRACTION
            done
        done
    done
done

DROP=10
for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature-downstream fw-mrs-svm-downstream
do
    # for BIAS_FRACTION in 0.1 0.2 0.3
    for BIAS_FRACTION in 0.1
    do
        for DATASET in folktables_income folktables_employment hr_analytics
        do
            for BIAS_TYPE in less_positive_class less_negative_class
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --n_cv_repeats $N_CV_REPEATS \
                --experiment_name temperature_comparison --drop $DROP \
                --bias_fraction $BIAS_FRACTION
            done
        done
    done
done

#for SAMPLE_WEIGHTING_METHOD in fw-mrs-downstream
#do
#    for BIAS_FRACTION in 0.4
#    do
#        for DATASET in folktables_income breast_cancer loan_prediction hr_analytics  folktables_employment 
#        do
#            for BIAS_TYPE in mean_difference
#            do
#                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
#                --bias_type $BIAS_TYPE --n_cv_repeats $N_CV_REPEATS \
#                --experiment_name temperature_comparison --drop $DROP \
#                --bias_fraction $BIAS_FRACTION
#            done
#        done
#    done
#done