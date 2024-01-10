import random
import numpy as np
import pandas as pd
import json
from tqdm import trange

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold
from utils.metrics import (
    calculate_rbf_gamma,
    compute_relative_bias,
    compute_test_metrics_mrs,
    train_pu_classifier,
    weighted_maximum_mean_discrepancy,
)

# Used to draw radom states
max_int = 2**32 - 1


def mrs(
    N,
    R,
    columns,
    n_drop: int = 1,
    cv=5,
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
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
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


def repeated_MRS(
    N,
    R,
    columns,
    delta=0.001,
    early_stopping=False,
    return_metrics=False,
    cv=5,
    drop=1,
    random_generator=None,
    *args,
    **attributes
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
    :param class_weights: Type of class weights, defaults to "balanced"
    :param drop: Defines how many samples are dropped per iteration, defaults to 1
    :param random_generator: Random generator to create random_states to make results reproducible
    :return: Sample weights or test metrics
    """
    auc_list = []
    relative_bias_list = []
    mmd_list = []
    roc_list = []
    number_of_iterations = len(N) // drop
    mrs_iteration = 0
    dropped_N = N.copy()
    sample_weights = np.ones(len(N))
    dropped_N = dropped_N.reset_index(drop=True)
    best_difference = np.inf
    best_mmd = 1

    # Compute and save mmd inputs to save time
    gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))
    x_x_rbf_matrix = rbf_kernel(N[columns], N[columns], gamma=gamma)
    x_y_rbf_matrix = rbf_kernel(N[columns], R[columns], gamma=gamma)
    y_y_rbf_matrix = rbf_kernel(R[columns], R[columns], gamma=gamma)

    # Start values
    mmd_list.append(
        weighted_maximum_mean_discrepancy(
            N[columns],
            R[columns],
            sample_weights,
            feature_weights=None,
            gamma=gamma,
            x_x_rbf_matrix=x_x_rbf_matrix,
            x_y_rbf_matrix=x_y_rbf_matrix,
            y_y_rbf_matrix=y_y_rbf_matrix,
        )
    )
    auroc = compute_test_metrics_mrs(
        pd.concat([dropped_N, R]),
        columns,
        random_state=random_generator.randint(max_int),
    )
    patience = 0

    auc_list.append(auroc)
    mutual_information_list = []
    for i in trange(number_of_iterations):
        # TODO something with feature weights

        dropped_N, drop_ids = mrs(
            N=dropped_N,
            R=R,
            columns=columns,
            n_drop=drop,
            cv=cv,
            random_state=random_generator.randint(max_int),
        )
        sample_weights[drop_ids] = 0

        # auroc = compute_test_metrics_mrs(
        #    pd.concat([dropped_N, R]),
        #    columns,
        #    random_state=random_generator.randint(max_int),
        # )
        # auc_list.append(auroc)

        mutual_information = compute_feature_weights(dropped_N, R, columns)
        feature_weights = 1 - (mutual_information * 50)
        feature_weights[feature_weights < 0] = 0
        feature_weights = (feature_weights / np.sum(feature_weights)) * dropped_N.shape[1]
        mutual_information_sum = np.sum(mutual_information)
        mutual_information_list.append(mutual_information_sum)

        mmd = weighted_maximum_mean_discrepancy(
            N[columns],
            R[columns],
            sample_weights,
            feature_weights=feature_weights,
            gamma=gamma,
            #x_x_rbf_matrix=x_x_rbf_matrix,
            #x_y_rbf_matrix=x_y_rbf_matrix,
            #y_y_rbf_matrix=y_y_rbf_matrix,
        )
        mmd_list.append(mmd)

        # if mmd < best_mmd:
        #   best_weights = weights.copy()
        #   mrs_iteration = (i + 1) * drop
        #    best_mmd = mmd
        #    patience = 0
        # else:
        #    patience += 1

        # auc_difference = abs(auroc - 0.5)
        # if (auc_difference + delta) <= best_difference:
        #     best_weights = weights.copy()
        #     mrs_iteration = (i + 1) * drop
        #     best_difference = auc_difference

        if (
            len(dropped_N) <= cv
            #   or ((best_difference <= delta) and early_stopping)
            or len(dropped_N) <= drop
            #   or patience >= 25
        ):
            break

    best_weights = sample_weights
    best_weights = best_weights.astype(np.float64)

    with open("mutual_information_sum.json", "w") as file:
        json.dump(mutual_information_list, file)

    with open("mmd_without_feature_weights.json", "w") as file:
        json.dump(mmd_list, file)

    if return_metrics:
        return auc_list, mmd_list, relative_bias_list, mrs_iteration, roc_list
    else:
        return best_weights / best_weights.sum()


from sklearn.feature_selection import mutual_info_classif


def compute_feature_weights(dropped_N, R, columns):
    data = pd.concat([dropped_N[columns], R[columns]])
    targets = np.concatenate([np.ones(len(dropped_N)), np.zeros(len(R))])
    mutual_information = mutual_info_classif(data, targets, n_neighbors=5)
    return mutual_information 
