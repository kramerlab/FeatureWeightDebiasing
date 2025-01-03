DROP=1
N_CV_REPEATS=50
DATASET=gbs_allensbach

for SAMPLE_WEIGHTING_METHOD in uniform mrs-forest soft-mrs-exponential
do
    python src/weighting_experiment.py --drop $DROP --n_cv_repeats $N_CV_REPEATS \
    --dataset $DATASET --experiment_name=statistical_analysis --sample_weighting_method $SAMPLE_WEIGHTING_METHOD
done