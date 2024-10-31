NUMBER_OF_REPETETIONS=1
DROP=1

python src/mrs_analysis.py --data_set_name gbs_gesis --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP
python src/mrs_analysis.py --data_set_name gbs_allensbach --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP

for BIAS_FRACTION in 0.1 0.2 0.3
do
    for BIAS_TYPE in less_positive_class less_negative_class none mean_difference
    do 
        python src/mrs_analysis.py --data_set_name folktables_income --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS\
         --drop $DROP --bias_fraction $BIAS_FRACTION
    done
done

for BIAS_FRACTION in 0.8 0.9
do
    for BIAS_TYPE in mean_difference 
    do 
        python src/mrs_analysis.py --data_set_name folktables_income --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS\
         --drop $DROP --bias_fraction $BIAS_FRACTION
    done
done