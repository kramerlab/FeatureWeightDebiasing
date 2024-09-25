import random
import numpy as np
import pandas as pd

from tqdm import trange

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    KFold,
)
from sklearn.metrics import roc_auc_score

from utils.metrics import (
    calculate_mean_roc,
    calculate_rbf_gamma,
    compute_relative_bias,
    compute_test_metrics_mrs,
    interpolate_roc,
    train_pu_classifier_mrs,
    weighted_maximum_mean_discrepancy,
    train_tree_classifier_mrs,
)

# Used to draw random states
max_int = 2**32 - 1


def mrs_step(
    N,
    R,
    columns,
    target,
    n_drop: int = 1,
    n_splits=5,
    random_state=None,
    calculate_roc=False,
    sample_weights=None,
    *args,
    **attributes
):
    """Performs one iteration of maximum representative sampling

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Columns names used for training
    :param n_drop: Number of samples to drop every iteration, defaults to 1
    :param cv: Number of cross-validation iterations, defaults to 5
    :param class_weight: Type of class weights, defaults to "balanced"
    :param random_state: Random state to make results reproducible
    :return: _description_
    """
    auroc_list = []
    ifpr_list = []
    itpr_list = []

    dropped_N = N[sample_weights != 0.0]
    all_predictions = np.zeros(len(dropped_N))
    skf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=2, random_state=random_state
    )
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=2, random_state=random_state)
    for (train_indices_N, test_indices_N), (train_indices_R, test_indices_R) in zip(
        skf.split(dropped_N, dropped_N[target]), kf.split(R)
    ):
        N_train, N_test = (
            dropped_N.iloc[train_indices_N],
            dropped_N.iloc[test_indices_N],
        )
        R_train, R_test = R.iloc[train_indices_R], R.iloc[test_indices_R]
        train_data = pd.concat([N_train, R_train])
        clf = train_pu_classifier_mrs(
            train_data[columns],
            train_data.label,
            random_state=random_state,
        )
        test_data = pd.concat([N_test, R_test])
        predictions = clf.predict_proba(test_data[columns])[:, 1]
        all_predictions[test_indices_N] = predictions[: len(N_test)]
        auroc_list.append(roc_auc_score(test_data.label, predictions))

        if calculate_roc:
            interpolated_fpr, interpolated_tpr = interpolate_roc(
                test_data.label, predictions
            )
            ifpr_list.append(interpolated_fpr)
            itpr_list.append(interpolated_tpr)

    drop_ids = np.argpartition(all_predictions, -n_drop)[-n_drop:]

    if calculate_roc:
        mean_ifpr_list, mean_itpr_list, std_tpr = calculate_mean_roc(
            ifpr_list, itpr_list
        )
        return (
            dropped_N.index[drop_ids],
            np.mean(auroc_list),
            mean_ifpr_list,
            mean_itpr_list,
            std_tpr,
        )
    else:
        return dropped_N.index[drop_ids], np.mean(auroc_list)


def mrs_without_cv(
    N,
    R,
    columns,
    target,
    n_drop: int = 1,
    class_weight="balanced",
    random_state=None,
    *args,
    **attributes
):
    """Performs one iteration of maximum representative sampling without cross-validation

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Name of columns used for training
    :param n_drop: Number of samples to drop every iteration, defaults to 1
    :param class_weights: Type of class weights, defaults to "balanced"
    :param random_state: Random state to make the experiment reproducible, defaults to None
    :return: The index of the element to drop
    """
    data = pd.concat([N, R])
    clf = train_pu_classifier_mrs(
        data[columns],
        data.label,
        class_weight=class_weight,
        random_state=random_state,
    )
    predictions = clf.predict_proba(N[columns])[:, 1]
    drop_ids = np.argpartition(predictions, -n_drop)[-n_drop:]

    drop_index = N.index[drop_ids]
    return N.drop(N.index[drop_ids]), drop_index


def mrs(
    N,
    R,
    columns,
    delta=0.01,
    early_stopping=False,
    mrs_function=mrs_step,
    return_metrics=False,
    compute_bias=True,
    target=None,
    n_pu_splits=5,
    drop=1,
    max_patience=5,
    random_generator=None,
    *args,
    **attributes
):
    """Performs the whole mrs

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Name of the columns used in training
    :param delta: Delta for the stopping criterion, defaults to 0.001
    :param early_stopping: If true, stops before dropping all samples, defaults to False
    :param mrs_function: Function that is used in evers mrs iteration, defaults to mrs
    :param return_metrics: If true, return test metrics, defaults to False
    :param compute_bias: If true, compute relative bias, defaults to True
    :param bias_variable: Name of the biased variable, defaults to None
    :param cv: Number of cross-validation iterations, defaults to 5
    :param class_weights: Type of class weights, defaults to "balanced"
    :param drop: Defines how many samples are dropped per iteration, defaults to 1
    :param random_generator: Random generator to create random_states to make results reproducible
    :return: Sample weights or test metrics
    """
    auc_list = []
    relative_bias_list = []
    mmd_list = []
    roc_list = []
    number_of_iterations = (len(N) - n_pu_splits) // drop
    mrs_iteration = 0
    roc_iteration = (len(N) // drop // 3.5) + 1
    sample_weights = np.ones(len(N))
    best_difference = np.inf
    current_patience = 0
    switched = False
    N = N.reset_index(drop=True)
    # Compute and save mmd inputs to save time
    # Start values
    if return_metrics:
        # Compute and save mmd inputs to save time
        gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))
        x_x_rbf_matrix = rbf_kernel(N[columns], N[columns], gamma=gamma)
        x_y_rbf_matrix = rbf_kernel(N[columns], R[columns], gamma=gamma)
        y_y_rbf_matrix = rbf_kernel(R[columns], R[columns], gamma=gamma)
        mmd_list.append(
            weighted_maximum_mean_discrepancy(
                N[columns],
                R[columns],
                sample_weights,
                gamma=gamma,
                x_x_rbf_matrix=x_x_rbf_matrix,
                x_y_rbf_matrix=x_y_rbf_matrix,
                y_y_rbf_matrix=y_y_rbf_matrix,
            )
        )

    for i in trange(number_of_iterations):
        if i % roc_iteration == 0 and return_metrics:
            drop_ids, auroc, mean_ifpr_list, mean_itpr_list, std_tpr = mrs_function(
                N=N,
                R=R,
                columns=columns,
                target=target,
                n_drop=drop,
                n_splits=n_pu_splits,
                random_state=random_generator.randint(max_int),
                calculate_roc=True,
                sample_weights=sample_weights,
            )
            roc_list.append([mean_ifpr_list, mean_itpr_list, std_tpr, i * drop])
        else:
            drop_ids, auroc = mrs_function(
                N=N,
                R=R,
                columns=columns,
                target=target,
                n_drop=drop,
                n_splits=n_pu_splits,
                random_state=random_generator.randint(max_int),
                sample_weights=sample_weights,
            )

        if compute_bias and target is not None:
            relative_bias = compute_relative_bias(N[target], R[target], sample_weights)
            relative_bias_list.append(relative_bias)

        auc_difference = abs(auroc - 0.5)
        if (auc_difference <= best_difference or auroc <= 0.5) and not switched:
            best_weights = sample_weights.copy().astype(np.float64)
            mrs_iteration = i * drop
            best_difference = auc_difference
            current_patience = 0
            if auroc <= 0.5 and not switched:
                switched = True
        else:
            current_patience += 1

        sample_weights[drop_ids] = 0.0
        remaining = N[sample_weights != 0.0]
        n_positive = np.count_nonzero(remaining[target])
        n_negative = len(remaining) - n_positive

        if (
            ((best_difference <= delta or switched) and early_stopping)
            or len(remaining) <= drop
            # or (n_positive <= n_pu_splits or n_negative <= n_pu_splits)
            or ((current_patience == max_patience) and early_stopping)
        ):
            break

        if return_metrics:
            auc_list.append(auroc)
            mmd_list.append(
                weighted_maximum_mean_discrepancy(
                    N[columns],
                    R[columns],
                    sample_weights,
                    gamma=gamma,
                    x_x_rbf_matrix=x_x_rbf_matrix,
                    x_y_rbf_matrix=x_y_rbf_matrix,
                    y_y_rbf_matrix=y_y_rbf_matrix,
                )
            )

    best_weights = best_weights.astype(np.float64)

    if return_metrics:
        return auc_list, mmd_list, relative_bias_list, mrs_iteration, roc_list
    else:
        return (best_weights / best_weights.sum()).tolist(), None


def random_drops(N, n_drop: int = 1, *args, **attributes):
    """MRS variant that drops sample randomly

    :param N: Non-representative data set
    :param n_drop: Defines how many samples are dropped per iteration, defaults to 1
    :return: Index of the samples to drop
    """
    drop_ids = random.sample(range(0, len(N)), n_drop)
    return N.drop(N.index[drop_ids]), N.index[drop_ids]
