import numpy as np
from tqdm import trange
from .fw_mrs_svm import (
    compute_feature_weights_with_temperature,
    mrs_step,
)


# Used to draw radom states
max_int = 2**32 - 1


def feature_weighted_repeated_MRS_svm_downstream(
    N,
    R,
    target,
    columns,
    delta=0.01,
    drop=1,
    budgets=[1.0],
    random_generator=None,
    class_weight=None,
    n_pu_splits=5,
    hyperparameter_list=[0.0],
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
    switched_dict = {}

    finished_dict = {}

    for temperature in budgets:
        finished_dict[temperature] = {}
        best_difference_dict[temperature] = {}
        auc_difference_dict[temperature] = {}
        dropped_samples_dict[temperature] = {}
        feature_weighted_aurocs_dict[temperature] = {}
        sample_weights_dict[temperature] = {}
        abs_feature_importance_dict[temperature] = {}
        feature_weights_dict[temperature] = {}
        best_sample_weights_dict[temperature] = {}
        switched_dict[temperature] = {}

        for C in hyperparameter_list:
            _, abs_feature_importance, _ = mrs_step(
                N=dropped_N,
                R=R,
                target=target,
                columns=columns,
                n_drop=drop,
                random_state=random_generator.randint(max_int),
                class_weight=class_weight,
                n_splits=n_pu_splits,
                feature_weight=np.ones(len(columns)),
                sample_weights=np.ones(len(N)),
                C=C,
            )
            best_sample_weights_dict[temperature][C] = {}
            finished_dict[temperature][C] = False
            best_difference_dict[temperature][C] = np.inf
            auc_difference_dict[temperature][C] = 1
            dropped_samples_dict[temperature][C] = 0
            feature_weighted_aurocs_dict[temperature][C] = []
            sample_weights_dict[temperature][C] = np.ones(len(N))
            switched_dict[temperature][C] = False
            abs_feature_importance_dict[temperature][C] = np.ones(len(columns)).tolist()
            feature_weights_dict[temperature][C] = (
                compute_feature_weights_with_temperature(
                    temperature, np.array(abs_feature_importance)
                ).tolist()
            )

    rand_int = random_generator.randint(max_int)
    for i in trange(number_of_iterations):
        rand_int = random_generator.randint(max_int)
        for temperature in budgets:
            for C in hyperparameter_list:
                if finished_dict[temperature][C]:
                    break
                splitter = "best" if temperature is None else "feature_weighted_best"
                drop_ids, abs_feature_importance, auroc = mrs_step(
                    N=dropped_N,
                    R=R,
                    target=target,
                    columns=columns,
                    n_drop=drop,
                    random_state=rand_int,
                    class_weight=class_weight,
                    n_splits=n_pu_splits,
                    feature_weight=np.array(feature_weights_dict[temperature][C]),
                    splitter=splitter,
                    sample_weights=sample_weights_dict[temperature][C],
                    C=C,
                )

                feature_weighted_aurocs_dict[temperature][C].append(auroc)
                auc_difference = abs(auroc - 0.5)

                if (
                    (auc_difference + delta) <= best_difference_dict[temperature][C]
                    or (not switched_dict[temperature][C] and auroc < 0.5)
                ) and not finished_dict[temperature][C]:
                    best_difference_dict[temperature][C] = auc_difference
                    dropped_samples_dict[temperature][C] = i * drop
                    best_sample_weights_dict[temperature][C] = (
                        sample_weights_dict[temperature][C]
                        / np.sum(sample_weights_dict[temperature][C])
                    ).copy()

                    if not switched_dict[temperature][C] and auroc < 0.5:
                        switched_dict[temperature][C] = True

                sample_weights_dict[temperature][C][drop_ids] = 0
                remaining = dropped_N[sample_weights_dict[temperature][C] != 0.0]
                n_positive = np.count_nonzero(remaining[target])
                n_negative = len(remaining) - n_positive

                if (
                    len(remaining) <= drop
                    or (n_positive <= n_pu_splits or n_negative <= n_pu_splits)
                    or auc_difference <= delta
                ):
                    finished_dict[temperature][C] = True

        if all(all(finished.values()) for finished in finished_dict.values()):
            break

    return (
        feature_weighted_aurocs_dict,
        best_sample_weights_dict,
        dropped_samples_dict,
        feature_weights_dict,
        abs_feature_importance_dict,
    )
