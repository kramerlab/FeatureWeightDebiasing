NUMBER_OF_REPETETIONS=10
DROP=1

for BIAS_VARIABLE in cross-validation random
do
    python src/mrs_ablation_study.py --ablation_experiment $BIAS_VARIABLE --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP
done