import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from tqdm import trange

from sklearn.model_selection import KFold
from utils.metrics import (
    compute_test_metrics_fw_mrs,
    train_pu_classifier,
    train_feature_weighted_classifier_forest,
    train_feature_weighted_classifier_tree,
)
import json

# Used to draw radom states
max_int = 2**32 - 1


def mrs(
    N,
    R,
    columns,
    n_drop: int = 1,
    n_splits=2,
    n_repeats=5,
    class_weights="balanced",
    random_state=None,
    *args,
    **attributes,
):
    """Performs one iteration of maximum representative sampling

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Columns names used for training
    :param n_drop: Number of samples to drop every iteration, defaults to 1
    :param cv: Number of cross-validation iterations, defaults to 5
    :param class_weights: Type of class weights, defaults to "balanced_subsample"
    :param random_state: Random state to make results reproducible
    :return: _description_
    """
    all_predictions = np.zeros(len(N))
    feature_importance_list = []
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
        data = pd.concat([N_test, R])
        predictions = clf.predict_proba(N_test[columns])[:, 1]
        all_predictions[test_index] = predictions
        feature_importance = permutation_importance(
            clf,
            data[columns],
            data.label,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
            scoring="roc_auc",
        )
        feature_importance_list.append(feature_importance.importances_mean)

    mean_feature_importance = np.mean(feature_importance_list, axis=0)
    drop_ids = np.argpartition(all_predictions, -n_drop)[-n_drop:]
    drop_index = N.index[drop_ids]

    return drop_index, mean_feature_importance


def mrs_without_cv(
    N,
    R,
    columns,
    n_drop: int = 1,
    class_weight="balanced",
    random_state=None,
    n_repeats=5,
    *args,
    **attributes,
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
    clf = train_feature_weighted_classifier_forest(
        data[columns],
        data.label,
        class_weight=class_weight,
        random_state=random_state,
        splitter="best",
        draw_with_feature_weights=False,
        max_features=None,
    )
    predictions = clf.predict_proba(N[columns])[:, 1]
    feature_importance = permutation_importance(
        clf,
        data[columns],
        data.label,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="roc_auc",
    ).importances_mean

    drop_ids = np.argpartition(predictions, -n_drop)[-n_drop:]
    drop_index = N.index[drop_ids]

    return drop_index, feature_importance


def feature_weighted_repeated_MRS(
    N,
    R,
    columns,
    delta=0.005,
    early_stopping=False,
    drop=1,
    budgets=[0.0],
    random_generator=None,
    max_patience=10,
    class_weight="balanced",
    return_auroc=False,
    save_path=None,
    n_test_splits=5,
    n_pu_splits=5, 
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
    number_of_iterations = len(N) // drop
    dropped_N = N.copy().reset_index(drop=True)
    sample_weights = np.ones(len(N))
    best_difference = np.inf
    feature_weights = np.ones(len(columns))
    current_patience = 0
    draw_with_feature_weights = True
    feature_weighted_aurocs = {}
    feature_importance_list = []
    for budget in budgets:
        feature_weighted_aurocs[budget] = []

    auroc_method = train_feature_weighted_classifier_tree
    rand_int = random_generator.randint(max_int)
    auroc = compute_test_metrics_fw_mrs(
        dropped_N,
        R,
        columns,
        random_state=rand_int,
        feature_weights=feature_weights,
        method=auroc_method,
        draw_with_feature_weights=draw_with_feature_weights,
        class_weight=class_weight,
        speedup=True,
        max_features="sqrt",
        splitter="feature_weighted_best",
        n_splits_test=n_test_splits,
    )
    for budget in budgets:
        feature_weighted_aurocs[budget].append(auroc)

    for i in trange(number_of_iterations):
        rand_int = random_generator.randint(max_int)
        drop_ids, feature_importance = mrs_without_cv(
            N=dropped_N,
            R=R,
            columns=columns,
            n_drop=drop,
            random_state=rand_int,
            feature_weights=feature_weights,
            draw_with_feature_weights=draw_with_feature_weights,
            class_weight=class_weight,
            n_repeats=2,
            n_splits=n_pu_splits,
        )
        dropped_N = dropped_N.drop(drop_ids)

        feature_importance_list.append(feature_importance)
        dir = save_path / "feature_weights"
        dir.mkdir(exist_ok=True)
        with open(f"{dir}/feature_weights_{i}.json", "w", encoding="utf-8") as file:
            json.dump(list(feature_importance), file, indent=4)

        feature_weights_list = []
        for budget in budgets:
            feature_weights = compute_feature_weights(budget, feature_importance)
            feature_weights_list.append(feature_weights)
            # max_index = np.argmax(feature_weights)
            # feature_weights = np.zeros(len(feature_weights))
            # feature_weights[max_index] = 1
            max_featues = "sqrt" if budget == 0 else "sqrt"
            auroc = compute_test_metrics_fw_mrs(
                dropped_N,
                R,
                columns,
                random_state=rand_int,
                feature_weights=feature_weights,
                method=auroc_method,
                draw_with_feature_weights=draw_with_feature_weights,
                class_weight=class_weight,
                speedup=True,
                max_features=max_featues,
                splitter="feature_weighted_best",
                n_splits_test=n_test_splits,
            )
            feature_weighted_aurocs[budget].append(auroc)

        auc_difference = abs(auroc - 0.5)
        if (auc_difference + delta) <= best_difference:
            best_weights = sample_weights.copy().astype(np.float64)
            mrs_iteration = (i + 1) * drop
            best_difference = auc_difference
            best_feature_weights = feature_weights.copy()
            current_patience = 0
        else:
            current_patience += 1

        if (
            ((best_difference <= delta) and early_stopping)
            or len(dropped_N) <= drop
            or current_patience >= max_patience
        ):
            break

        sample_weights[drop_ids] = 0

        with open("feature_weighted_auroc.json", "w", encoding="utf-8") as file:
            json.dump(feature_weighted_aurocs, file, indent=4)

    if return_auroc:
        return (
            feature_weighted_aurocs,
            feature_importance_list,
            mrs_iteration,
        )

    else:
        return best_weights / best_weights.sum(), best_feature_weights


def compute_feature_weights(budget, feature_importances):
    """_summary_

    :param budget: _description_
    :param feature_importances: _description_
    :return: _description_
    """
    max_importance = np.max(np.abs(feature_importances))
    feature_importances = feature_importances / max_importance
    if max_importance == 0 or budget == 0:
        return np.ones(len(feature_importances)) / len(feature_importances)
    budget_feature_importances = 1 - (feature_importances * budget)
    budget_feature_importances = budget_feature_importances / np.sum(
        budget_feature_importances
    )
    return budget_feature_importances


def softmax(x):
    """_summary_

    :param x: _description_
    :return: _description_
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
