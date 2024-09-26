import numpy as np
from tqdm import trange
from .feature_weighted_maximum_representative_subsampling import (
    compute_feature_weights_with_temperature,
    initialize_dictionaries,
    mrs_step,
)


# Used to draw radom states
max_int = 2**32 - 1


def feature_weighted_repeated_MRS_downstream(
    N,
    R,
    columns,
    target,
    delta=0.01,
    drop=1,
    budgets=[1.0],
    random_generator=None,
    n_pu_splits=5,
    hyperparameter_list=[],
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
    best_difference_dict = {}
    best_sample_weights_dict = {}
    dropped_samples_dict = {}
    auc_difference_dict = {}
    abs_feature_importance_dict = {}
    sample_weights_dict = {}
    feature_weights_dict = {}
    feature_weighted_aurocs_dict = {}
    finished_dict = {}

    initialize_dictionaries(
        N,
        R,
        columns,
        target,
        drop,
        budgets,
        random_generator,
        n_pu_splits,
        hyperparameter_list,
        dropped_N,
        best_difference_dict,
        best_sample_weights_dict,
        dropped_samples_dict,
        auc_difference_dict,
        abs_feature_importance_dict,
        sample_weights_dict,
        feature_weights_dict,
        feature_weighted_aurocs_dict,
        finished_dict,
    )

    rand_int = random_generator.randint(max_int)

    for i in trange(number_of_iterations):
        for temperature in budgets:
            for hyperparameter in hyperparameter_list:
                if finished_dict[temperature][hyperparameter]:
                    break
                splitter = "best" if temperature is 0.0 else "feature_weighted_best"
                drop_ids, _, auroc = mrs_step(
                    N=dropped_N,
                    R=R,
                    target=target,
                    columns=columns,
                    n_drop=drop,
                    random_state=rand_int,
                    n_splits=n_pu_splits,
                    feature_weight=np.array(
                        feature_weights_dict[temperature][hyperparameter]
                    ),
                    splitter=splitter,
                    sample_weights=sample_weights_dict[temperature][hyperparameter],
                    hyperparameter=hyperparameter,
                )

                feature_weighted_aurocs_dict[temperature][hyperparameter].append(auroc)
                auc_difference = abs(auroc - 0.5)

                if (auc_difference + delta) <= best_difference_dict[temperature][
                    hyperparameter
                ]:
                    best_difference_dict[temperature][hyperparameter] = auc_difference
                    dropped_samples_dict[temperature][hyperparameter] = i * drop
                    best_sample_weights_dict[temperature][hyperparameter] = (
                        sample_weights_dict[temperature][hyperparameter]
                        / np.sum(sample_weights_dict[temperature][hyperparameter])
                    ).copy()

                sample_weights_dict[temperature][hyperparameter][drop_ids] = 0
                remaining = dropped_N[
                    sample_weights_dict[temperature][hyperparameter] != 0.0
                ]

                if len(remaining) <= drop:
                    finished_dict[temperature][hyperparameter] = True

        if all(all(finished.values()) for finished in finished_dict.values()):
            break

    return (
        feature_weighted_aurocs_dict,
        best_sample_weights_dict,
        dropped_samples_dict,
        feature_weights_dict,
        abs_feature_importance_dict,
    )
