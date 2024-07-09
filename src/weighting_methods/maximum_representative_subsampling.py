import random
import numpy as np
import pandas as pd

from tqdm import trange

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold
from utils.metrics import (
    calculate_rbf_gamma,
    compute_relative_bias,
    compute_test_metrics_mrs,
    train_pu_classifier,
    weighted_maximum_mean_discrepancy,
    train_tree_classifier_mrs,
    train_random_forest_classifier_mrs,
)

# Used to draw random states
max_int = 2**32 - 1


def mrs_step(
    N,
    R,
    columns,
    n_drop: int = 1,
    n_splits=5,
    class_weights="balanced",
    random_state=None,
    *args,
    **attributes
):
    """Performs one iteration of maximum representative sampling

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Columns names used for training
    :param n_drop: Number of samples to drop every iteration, defaults to 1
    :param cv: Number of cross-validation iterations, defaults to 5
    :param class_weights: Type of class weights, defaults to "balanced"
    :param random_state: Random state to make results reproducible
    :return: _description_
    """
    all_predictions = np.zeros(len(N))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(N):
        N_train, N_test = N.iloc[train_index], N.iloc[test_index]
        data = pd.concat([N_train, R])
        clf = train_pu_classifier(
            data[columns],
            data.label,
            class_weight=class_weights,
            random_state=random_state,
        )
        predictions = clf.predict_proba(N_test[columns])[:, 1]
        all_predictions[test_index] = predictions

    drop_ids = np.argpartition(all_predictions, -n_drop)[-n_drop:]
    drop_index = N.index[drop_ids]
    return N.drop(N.index[drop_ids]), drop_index


def mrs_without_cv(
    N,
    R,
    columns,
    n_drop: int = 1,
    class_weights="balanced",
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
    clf = train_pu_classifier(
        data[columns],
        data.label,
        class_weight=class_weights,
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
    delta=0.005,
    early_stopping=False,
    mrs_function=mrs_step,
    return_metrics=False,
    compute_bias=True,
    bias_variable=None,
    n_pu_cv=5,
    n_test_splits=5,
    class_weights="balanced",
    drop=1,
    random_generator=None,
    method_name="",
    max_patience=20,
    validation_iteration=10,
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
    number_of_iterations = (len(N) - n_test_splits) // drop
    mrs_iteration = 0
    roc_iteration = (len(N) // drop // 3.5) + 1
    dropped_N = N.copy()
    weights = np.ones(len(N))
    dropped_N = dropped_N.reset_index(drop=True)
    best_difference = np.inf
    current_patience = 0

    validation_method = (
        train_tree_classifier_mrs
        if method_name == "mrs-tree"
        else train_random_forest_classifier_mrs
    )
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
                weights,
                gamma=gamma,
                x_x_rbf_matrix=x_x_rbf_matrix,
                x_y_rbf_matrix=x_y_rbf_matrix,
                y_y_rbf_matrix=y_y_rbf_matrix,
            )
        )

    for i in trange(number_of_iterations):
        dropped_N, drop_ids = mrs_function(
            N=dropped_N,
            R=R,
            columns=columns,
            n_drop=drop,
            class_weights=class_weights,
            n_splits=n_pu_cv,
            random_state=random_generator.randint(max_int),
        )

        if i % roc_iteration == 0 and return_metrics:
            auroc, mean_ifpr_list, mean_itpr_list, std_tpr = compute_test_metrics_mrs(
                dropped_N,
                R,
                columns,
                calculate_roc=True,
                n_test_splits=n_test_splits,
                validation_method=validation_method,
            )
            roc_list.append([mean_ifpr_list, mean_itpr_list, std_tpr, i * drop])
        elif ((i + 1) % validation_iteration) == 0:
            auroc = compute_test_metrics_mrs(
                dropped_N,
                R,
                columns,
                random_state=random_generator.randint(max_int),
                n_test_splits=n_test_splits,
                validation_method=validation_method,
            )
            if compute_bias and bias_variable is not None:
                relative_bias = compute_relative_bias(
                    N[bias_variable], R[bias_variable], weights
                )
                relative_bias_list.append(relative_bias)

            auc_difference = abs(auroc - 0.5)
            if auc_difference <= best_difference:
                best_weights = weights.copy().astype(np.float64)
                mrs_iteration = (i + 1) * drop
                best_difference = auc_difference
                current_patience = 0
            else:
                current_patience += 1

            if (
                (best_difference <= delta and early_stopping)
                or len(dropped_N) <= drop
                or len(dropped_N) <= n_test_splits
                or (current_patience == max_patience and early_stopping)
            ):
                break
            
        weights[drop_ids] = 0.0

        if return_metrics and (
            (((i + 1) % validation_iteration) == 0) or (i % roc_iteration == 0)
        ):
            auc_list.append(auroc)
            mmd_list.append(
                weighted_maximum_mean_discrepancy(
                    N[columns],
                    R[columns],
                    weights,
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
        return (best_weights / best_weights.sum()).tolist(), None, None


def random_drops(N, n_drop: int = 1, *args, **attributes):
    """MRS variant that drops sample randomly

    :param N: Non-representative data set
    :param n_drop: Defines how many samples are dropped per iteration, defaults to 1
    :return: Index of the samples to drop
    """
    drop_ids = random.sample(range(0, len(N)), n_drop)
    return N.drop(N.index[drop_ids]), N.index[drop_ids]
