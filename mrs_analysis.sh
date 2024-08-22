NUMBER_OF_REPETETIONS=1
DROP=5

python src/mrs_analysis.py --data_set_name gbs_gesis --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP
python src/mrs_analysis.py --data_set_name gbs_allensbach --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP

for BIAS_FRACTION in 0.1
do
    for BIAS_TYPE in less_positive_class less_negative_class mean_difference none
    do 
        python src/mrs_analysis.py --data_set_name folktables_income --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS\
         --drop $DROP --bias_fraction $BIAS_FRACTION
    done
done