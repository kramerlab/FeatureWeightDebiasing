NUMBER_OF_REPETETIONS=5
SAMPLE_WEIGHTING_METHOD=fw-sampling-mrs
BIAS_FRACTION=0.1

# for VALIDATION_METHOD in decision_tree random_forest 
for VALIDATION_METHOD in random_forest 
do

    DROP=1
    for BIAS_TYPE in less_negative_class
    do
        for DATASET in gbs_gesis gbs_allensbach
        do
            python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name feature_weight_budget_comparison --drop $DROP \
            --validation_method $VALIDATION_METHOD --bias_fraction $BIAS_FRACTION
        done
    done

    #DROP=5
    #for BIAS_TYPE in less_negative_class less_positive_class mean_difference
    #do
    #    for DATASET in breast_cancer loan_prediction gbs_gesis gbs_allensbach
    #    do
    #        python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
    #        --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name feature_weight_budget_comparison --drop $DROP \
    #    --validation_method $VALIDATION_METHOD --bias_fraction $BIAS_FRACTION
    #    done
    #done

    DROP=10
    for BIAS_FRACTION in 0.1 0.25 0.5
    do
        # for BIAS_TYPE in less_negative_class less_positive_class mean_difference
        for BIAS_TYPE in less_negative_class less_positive_class
        do
            # for DATASET in hr_analytics folktables_income folktables_employment
            for DATASET in folktables_income 
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name feature_weight_budget_comparison --drop $DROP \
                --validation_method $VALIDATION_METHOD --bias_fraction $BIAS_FRACTION
            done
        done
    done
done