DROP=5
NUMBER_OF_REPETETIONS=20
DATASET=gbs_gesis

for SAMPLE_WEIGHTING_METHOD in  uniform fw-mrs-temperature
do
    python src/weighting_experiment.py --drop $DROP --number_of_repetitions $NUMBER_OF_REPETETIONS \
    --dataset $DATASET --experiment_name=statistical_analysis --sample_weighting_method $SAMPLE_WEIGHTING_METHOD
done