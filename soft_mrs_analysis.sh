N_CV_REPEATS=10
N_CV_SPLITS=5

python src/soft_mrs_analysis.py --data_set_name gbs_gesis --n_cv_repeats $N_CV_REPEATS \
    --n_cv_splits $N_CV_SPLITS --mrs_function linear --load_previous_results
python src/soft_mrs_analysis.py --data_set_name gbs_gesis --n_cv_repeats $N_CV_REPEATS \
    --n_cv_splits $N_CV_SPLITS --mrs_function exponential --load_previous_results
python src/soft_mrs_analysis.py --data_set_name gbs_allensbach --n_cv_repeats $N_CV_REPEATS \
    --n_cv_splits $N_CV_SPLITS --mrs_function exponential --load_previous_results
python src/soft_mrs_analysis.py --data_set_name gbs_allensbach --n_cv_repeats $N_CV_REPEATS \
    --n_cv_splits $N_CV_SPLITS --mrs_function linear --load_previous_results

for MRS_FUNCTION in linear exponential 
do
    for BIAS_TYPE in less_positive_class less_negative_class none
    do 
        python src/soft_mrs_analysis.py --data_set_name folktables_income --bias_type $BIAS_TYPE \
        --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS \
        --bias_fraction 0.1 --mrs_function $MRS_FUNCTION --load_previous_results
    done
done


for MRS_FUNCTION in linear exponential 
do
    python src/soft_mrs_analysis.py --data_set_name folktables_income --bias_type mean_difference \
    --n_cv_repeats $N_CV_REPEATS --n_cv_splits $N_CV_SPLITS \
    --bias_fraction 0.8 --mrs_function $MRS_FUNCTION --load_previous_results
done