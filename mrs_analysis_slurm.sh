#!/bin/bash

#SBATCH -A m2_datamining
#SBATCH -p parallel 
#SBATCH -J "feature_weighting_cross_validation" # gives SLURM_JOB_NAME
#SBATCH -n 10 # gives SLURM_NTASKS
#SBATCH -t 7200 # <time in minutes>
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=200000M 


NUMBER_OF_REPETETIONS=1
DROP=1

python srun src/mrs_analysis.py --data_set_name gbs_gesis --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP &
python srun src/mrs_analysis.py --data_set_name gbs_allensbach --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP &


DROP=5
for BIAS_FRACTION in 0.1 0.2 0.3
do
    for BIAS_TYPE in less_positive_class less_negative_class none mean_difference
    do 
        for DATASET in folktables_income folktables_employment  
        do
            srun python src/mrs_analysis.py --data_set_name $DATASET --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS\
            --drop $DROP --bias_fraction $BIAS_FRACTION &
        done
    done
done

for BIAS_FRACTION in 0.8 0.9
do
    for BIAS_TYPE in mean_difference 
    do 
        for DATASET in folktables_income folktables_employment  
        do
            srun python /src/mrs_analysis.py --data_set_name folktables_income --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS\
            --drop $DROP --bias_fraction $BIAS_FRACTION &
        done
    done
done
wait