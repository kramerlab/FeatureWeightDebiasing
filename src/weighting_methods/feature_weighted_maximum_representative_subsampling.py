import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import trange

from sklearn.model_selection import  RepeatedKFold, RepeatedStratifiedKFold
from utils.metrics import (
    calculate_feature_importance,
    compute_feature_weights_with_temperature,
    train_pu_classifier,
)

# Used to draw radom states
max_int = 2**32 - 1


def mrs_step(
    N,
    R,
    columns,
    target,
    n_drop: int = 1,
    n_splits=5,
    random_state=None,
    feature_weight=None,
    splitter="feature_weighted_best",
    sample_weights=None,
    compute_feature_importance=False,
    hyperparameter=0.0,
    *args,
    **attributes,
):
    """Performs one iteration of maximum representative sampling

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Columns names used for training
    :param n_drop: Number of samples to drop every iteration, defaults to 1
    :param cv: Number of cross-validation iterations, defaults to 5
    :param class_weight: Type of class weights, defaults to "balanced_subsample"
    :param random_state: Random state to make results reproducible
    :return: _description_
    """
    auroc_list = []
    abs_feature_importance_list = []
    dropped_N = N[sample_weights != 0]
    all_predictions = np.zeros(len(dropped_N))
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=2, random_state=random_state)
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
        clf = train_pu_classifier(
            train_data[columns],
            train_data.label,
            random_state=random_state,
            feature_weight=feature_weight,
            splitter=splitter,
            hyperparameter=hyperparameter
        )
        test_data = pd.concat([N_test, R_test])
        predictions = clf.predict_proba(test_data[columns])[:, 1]
        all_predictions[test_indices_N] = predictions[: len(N_test)]
        auroc_list.append(roc_auc_score(test_data.label, predictions))

        if compute_feature_importance:
            abs_feature_importance = calculate_feature_importance(
                test_N=N_test[columns].values,
                clf=clf,
                background=train_data[columns],
            )
            abs_feature_importance_list.append(abs_feature_importance)

    if compute_feature_importance:
        abs_mean_feature_importance = np.nanmean(abs_feature_importance_list, axis=0)
    else:
        abs_mean_feature_importance = None
    drop_ids = np.argpartition(all_predictions, -n_drop)[-n_drop:]

    return dropped_N.index[drop_ids], abs_mean_feature_importance, np.mean(auroc_list)


def mrs_without_cv(
    N,
    R,
    columns,
    n_drop: int = 1,
    class_weight="balanced",
    random_state=None,
    feature_weights=None,
    *args,
    **attributes,
):
    """Performs one iteration of maximum representative sampling without cross-validation

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Name of columns used for training
    :param n_drop: Number of samples to drop every iteration, defaults to 1
    :param class_weight: Type of class weights, defaults to "balanced"
    :param random_state: Random state to make the experiment reproducible, defaults to None
    :return: The index of the element to drop
    """
    data = pd.concat([N, R])
    clf = train_pu_classifier(
        data[columns],
        data.label,
        class_weight=class_weight,
        random_state=random_state,
        feature_weight=feature_weights,
    )
    predictions_N = clf.predict_proba(N[columns])[:, 1]
    feature_importance, _ = calculate_feature_importance(
        test_N=N[columns].values,
        clf=clf,
    )
    predictions = clf.predict_proba(data[columns])[:, 1]
    auroc = roc_auc_score(data.label, predictions)

    drop_ids = np.argpartition(predictions_N, -n_drop)[-n_drop:]
    drop_index = N.index[drop_ids]

    return drop_index, feature_importance, auroc


def compute_feature_weights_with_budget(budget, feature_importance):
    if budget is None:
        return np.ones(len(feature_importance))
    else:
        max_importance = np.max(feature_importance)
        min_importance = np.min(feature_importance)
        feature_importance = (feature_importance - min_importance) / (
            max_importance - min_importance
        )
        scaled_feature_importance = feature_importance * budget
        scaled_feature_importance = 1 + scaled_feature_importance
        return scaled_feature_importance


def feature_weighted_repeated_MRS(
    N,
    R,
    target,
    columns,
    delta=0.01,
    early_stopping=False,
    drop=1,
    budgets=[1.0],
    random_generator=None,
    class_weight=None,
    n_pu_splits=5,
    max_patience=5,
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
    :param class_weight: Type of class weights, defaults to "balanced_subsample"
    :param drop: Defines how many samples are dropped per iteration, defaults to 1
    :param random_generator: Random generator to create random_states to make results reproducible
    :return: Sample weights or test metrics
    """
    number_of_iterations = (len(N) - (n_pu_splits + 1)) // drop
    dropped_N = N.copy().reset_index(drop=True)
    sample_weights_dict = {}
    feature_weighted_aurocs_dict = {}
    feature_weights_dict = {}
    dropped_samples_dict = {}
    finished_dict = {}
    best_difference_dict = {}
    best_sample_weights_dict = {}
    dropped_samples_dict = {}
    auc_difference_dict = {}
    current_patience = {}
    switched = False

    finished_dict = {}

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
        splitter="best",
        sample_weights=np.ones(len(N)),
        compute_feature_importance=True
    )

    for temperature in budgets:
        finished_dict[temperature] = False
        best_difference_dict[temperature] = np.inf
        auc_difference_dict[temperature] = 1
        dropped_samples_dict[temperature] = 0
        current_patience[temperature] = 0
        feature_weighted_aurocs_dict[temperature] = []
        sample_weights_dict[temperature] = np.ones(len(N))
        feature_weights_dict[temperature] = compute_feature_weights_with_temperature(
            temperature, np.array(abs_feature_importance)
        ).tolist()
        best_sample_weights_dict[temperature] = np.ones(len(N)).tolist()

    for i in trange(number_of_iterations):
        for temperature in budgets:
            if finished_dict[temperature]:
                break
            splitter = "best" if temperature is None else "feature_weighted_best"
            drop_ids, abs_feature_importance, auroc = mrs_step(
                N=dropped_N,
                R=R,
                target=target,
                columns=columns,
                n_drop=drop,
                random_state=random_generator.randint(max_int),
                class_weight=class_weight,
                n_splits=n_pu_splits,
                feature_weight=np.array(feature_weights_dict[temperature]),
                splitter=splitter,
                sample_weights=sample_weights_dict[temperature],
            )

            feature_weighted_aurocs_dict[temperature].append(auroc)

            auc_difference = abs(auroc - 0.5)

            if (
                (auc_difference + delta) <= best_difference_dict[temperature]
                or (not switched and auroc < 0.5)
            ) and not finished_dict[temperature]:
                best_difference_dict[temperature] = auc_difference
                dropped_samples_dict[temperature] = i * drop
                sample_weights = sample_weights_dict[temperature]
                best_sample_weights_dict[temperature] = (
                    (sample_weights / np.sum(sample_weights)).tolist().copy()
                )
                current_patience[temperature] = 0
                if not switched and auroc < 0.5:
                    switched = True
            else:
                current_patience[temperature] += 1
            if (
                len(dropped_N) <= drop
                or len(dropped_N) <= n_pu_splits
                or (auc_difference <= delta and early_stopping)
                or (current_patience[temperature] == max_patience and early_stopping)
                or auroc < 0.5
            ):
                finished_dict[temperature] = True

            sample_weights_dict[temperature][drop_ids] = 0

        if all(finished_dict.values()) and early_stopping:
            break

    return (
        best_sample_weights_dict,
        feature_weights_dict,
    )
