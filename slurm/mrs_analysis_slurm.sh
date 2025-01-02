#!/bin/bash

#SBATCH -A m2_datamining
#SBATCH -p parallel 
#SBATCH -J "feature_weighting_cross_validation" # gives SLURM_JOB_NAME
#SBATCH -n 1 # gives SLURM_NTASKS
#SBATCH -t 5-00 
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=8G 


NUMBER_OF_REPETETIONS=10
N_CV_SPLITS=5
DROP=1

python srun src/mrs_analysis.py --data_set_name gbs_gesis --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP &
python srun src/mrs_analysis.py --data_set_name gbs_allensbach --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP &

source ~/.bashrc
conda_initialize
micromamba activate feature_weighted_mrs

DROP=5
for BIAS_FRACTION in 0.1 0.2 0.3
do
    for BIAS_TYPE in less_positive_class less_negative_class none mean_difference
    do 
        for DATASET in folktables_income folktables_employment  
        do
            srun python ../src/mrs_analysis.py --data_set_name $DATASET --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS\
            --drop $DROP --bias_fraction $BIAS_FRACTION 
        done
    done
done

for BIAS_FRACTION in 0.8 0.9
do
    for BIAS_TYPE in mean_difference 
    do 
        for DATASET in folktables_income folktables_employment  
        do
            srun python ../src/mrs_analysis.py --data_set_name folktables_income --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS\
            --drop $DROP --bias_fraction $BIAS_FRACTION 
        done
    done
done
wait