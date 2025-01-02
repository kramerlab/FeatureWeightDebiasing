N_CV_REPEATS=10
N_CV_SPLITS=5
DROP=5

for MRS_FUNCTION in linear exponential 
do
    for BIAS_TYPE in less_positive_class less_negative_class none mean_difference
    do 
        for DATASET in folktables_income folktables_employment  
        do
            python src/soft_mrs_analysis.py --data_set_name $DATASET --bias_type $BIAS_TYPE --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS \
            --drop $DROP --bias_fraction 0.1 --mrs_function $MRS_FUNCTION
        done
    done
done

for MRS_FUNCTION in linear exponential 
do
    for BIAS_TYPE in mean_difference 
    do 
        for DATASET in folktables_income folktables_employment  
        do
            python src/soft_mrs_analysis.py --data_set_name folktables_income --bias_type $BIAS_TYPE --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS \
            --drop $DROP --bias_fraction 0.8 --mrs_function $MRS_FUNCTION
        done
    done
done