import numpy as np
from tqdm import trange
from .feature_weighted_maximum_representative_subsampling import (
    compute_feature_weights_with_temperature,
    mrs,
)
from utils.metrics import compute_test_metrics_fw_mrs


# Used to draw radom states
max_int = 2**32 - 1


def feature_weighted_repeated_MRS(
    N,
    R,
    columns,
    delta=0.01,
    drop=1,
    budgets=[1.0],
    random_generator=None,
    class_weight="balanced",
    n_pu_splits=10,
    n_estimators=100,
    method_name=None,
    *args,
    **attributes,
):
    """Performs MRS

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Name of the columns used in training
    :param delta: Delta for the stopping criterion, defaults to 0.001
    :param early_stopping: If true, stops before dropping all samples, defaults to False
    :param mrs_function: Function that is used in evers mrs iteration, defaults to mrs
    :param return_metrics: If true, return test metrics, defaults to False
    :param use_bias_mean: If true, compute relative bias, defaults to True
    :param bias_variable: Name of the biased variable, defaults to None
    :param cv: Number of cross-validation iterations, defaults to 5
    :param class_weights: Type of class weights, defaults to "balanced_subsample"
    :param drop: Defines how many samples are dropped per iteration, defaults to 1
    :param random_generator: Random generator to create random_states to make results reproducible
    :return: Sample weights or test metrics
    """
    number_of_iterations = (len(N) - (n_pu_splits + 1)) // drop
    dropped_N = N.copy().reset_index(drop=True)
    sample_weights = np.ones(len(N))
    best_difference_dict = {}
    best_sample_weights_dict = {}
    dropped_samples_dict = {}
    best_feature_weights_dict = {}
    auc_difference_dict = {}
    abs_feature_importance_dict = {}

    finished_dict = {}
    for temperature in budgets:
        finished_dict[temperature] = False
        best_difference_dict[temperature] = np.inf
        auc_difference_dict[temperature] = 1
        dropped_samples_dict[temperature] = 0
        abs_feature_importance_dict[temperature] = np.ones(len(columns)).tolist()

    rand_int = random_generator.randint(max_int)

    for i in trange(number_of_iterations):
        rand_int = random_generator.randint(max_int)
        for temperature in budgets:
            current_feature_importance = abs_feature_importance_dict[temperature]
            feature_weights = compute_feature_weights_with_temperature(
                temperature, current_feature_importance
            )
            drop_ids, abs_feature_importance, auroc = mrs(
                N=dropped_N,
                R=R,
                columns=columns,
                n_drop=drop,
                random_state=rand_int,
                class_weight=class_weight,
                n_splits=n_pu_splits,
                feature_weight=feature_weights
            )

            if i == 0:
                abs_feature_importance_dict[temperature] = (
                    abs_feature_importance.tolist()
                )

            auc_difference = abs(auroc - 0.5)

            if (auc_difference + delta) <= best_difference_dict[
                temperature
            ] and not finished_dict[temperature]:
                best_difference_dict[temperature] = auc_difference
                dropped_samples_dict[temperature] = i * drop
                best_sample_weights_dict[temperature] = (
                    sample_weights / np.sum(sample_weights)
                ).copy()
                best_feature_weights_dict[temperature] = feature_weights.copy()
            if (
                len(dropped_N) <= drop
                or len(dropped_N) <= n_pu_splits
                or auc_difference <= delta
            ):
                finished_dict[temperature] = True

        if all(finished_dict.values()):
            break

        dropped_N = dropped_N.drop(drop_ids)
        sample_weights[drop_ids] = 0.0

    return (
        best_sample_weights_dict,
        dropped_samples_dict,
        best_feature_weights_dict,
    )
