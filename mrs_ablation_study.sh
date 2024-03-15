NUMBER_OF_REPETETIONS=1
DROP=5

for BIAS_VARIABLE in cross-validation random
do
    python src/mrs_ablation_study.py --ablation_experiment $BIAS_VARIABLE --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP
done