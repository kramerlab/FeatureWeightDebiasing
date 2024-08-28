NUMBER_OF_REPETETIONS=1

#bDROP=2
#bfor SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature fw-mrs-svm
# do
#     for DATASET in gbs_gesis gbs_allensbach
#     do
#         python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
#                 --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name feature_weight_budget_comparison --drop $DROP 
#     done
# done

DROP=5
# for BIAS_FRACTION in 0.1 0.2 0.3
for BIAS_FRACTION in 0.2 0.3
do
    for BIAS_TYPE in less_positive_class less_negative_class
    do
        for DATASET in folktables_income
        do
            for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature fw-mrs-svm
            do
                python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
                --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name feature_weight_budget_comparison --drop $DROP \
                    --bias_fraction $BIAS_FRACTION
            done
        done
    done
done


#BIAS_FRACTION=0.4
    #for SAMPLE_WEIGHTING_METHOD in fw-mrs-temperature fw-mrs-svm
    #do
    #for BIAS_TYPE in mean_difference
    #do
    #    for DATASET in folktables_income 
    #    do
    #        python src/weighting_experiment.py --dataset $DATASET  --sample_weighting_method $SAMPLE_WEIGHTING_METHOD  \
    #        --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS --experiment_name feature_weight_budget_comparison --drop $DROP \
    #            --bias_fraction $BIAS_FRACTION
    #    done
    #done
#done