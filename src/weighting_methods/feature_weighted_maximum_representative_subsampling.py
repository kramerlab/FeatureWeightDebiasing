import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from tqdm import trange

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from utils.metrics import (
    calculate_rbf_gamma,
    compute_relative_bias,
    compute_test_metrics_mrs,
    weighted_maximum_mean_discrepancy,
    train_classifier_auroc_feature_weighted,
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
    feature_importance_list = []
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
        data = pd.concat([N_test, R])
        predictions = clf.predict_proba(N_test[columns])[:, 1]
        all_predictions[test_index] = predictions
        feature_importance = permutation_importance(
            clf,
            data[columns],
            data.label,
            n_repeats=25,
            random_state=random_state,
            n_jobs=5,
            scoring="roc_auc",
        )
        feature_importance_list.append(feature_importance.importances_mean)

    mean_feature_importance = np.mean(feature_importance_list, axis=0)
    drop_ids = np.argpartition(all_predictions, -n_drop)[-n_drop:]
    drop_index = N.index[drop_ids]
    return N.drop(drop_index), drop_index, mean_feature_importance


def mrs_without_cv(
    N,
    R,
    columns,
    n_drop: int = 1,
    class_weights="balanced",
    random_state=None,
    feature_weights=None,
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
    feature_importance = permutation_importance(
        clf,
        data[columns],
        data.label,
        n_repeats=25,
        random_state=random_state,
        n_jobs=5,
        scoring="roc_auc",
    ).importances_mean
    drop_ids = np.argpartition(predictions, -n_drop)[-n_drop:]
    drop_index = N.index[drop_ids]

    return N.drop(drop_index), drop_index, feature_importance


def feature_weighted_repeated_MRS(
    N,
    R,
    columns,
    delta=0.001,
    early_stopping=False,
    return_metrics=False,
    cv=5,
    drop=1,
    budget=0.1,
    random_generator=None,
    max_patience=25,
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
    mmd_list_with_feature_weights = []
    roc_list = []
    number_of_iterations = len(N) // drop
    mrs_iteration = 0
    dropped_N = N.copy()
    sample_weights = np.ones(len(N))
    dropped_N = dropped_N.reset_index(drop=True)
    best_difference = np.inf
    feature_weights = np.ones(len(columns))
    feature_importance_list = []

    auroc = compute_test_metrics_mrs(
        pd.concat([dropped_N, R]),
        columns,
        random_state=random_generator.randint(max_int),
        feature_weights=feature_weights,
        splitter="feature_weighted",
        method=train_classifier_auroc_feature_weighted,
    )
    current_patience = 0

    auc_list.append(auroc)
    for i in trange(number_of_iterations // drop):
        dropped_N, drop_ids, feature_importance = mrs(
            N=dropped_N,
            R=R,
            columns=columns,
            n_drop=drop,
            cv=cv,
            random_state=random_generator.randint(max_int),
        )
        feature_importance_list.append(np.sum(feature_importance))
        feature_weights = compute_feature_weights(
            len(columns), budget, feature_importance
        )
        auroc = compute_test_metrics_mrs(
            pd.concat([dropped_N, R]),
            columns,
            random_state=random_generator.randint(max_int),
            cv=5,
            feature_weights=feature_weights,
            splitter="feature_weighted",
            method=train_classifier_auroc_feature_weighted,
        )
        auc_list.append(auroc)

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
            len(dropped_N) <= cv
            or ((best_difference <= delta) and early_stopping)
            or len(dropped_N) <= drop
            or current_patience >= max_patience
        ):
            break

        sample_weights[drop_ids] = 0

    if return_metrics:
        return (
            auc_list,
            mmd_list_with_feature_weights,
            relative_bias_list,
            mrs_iteration,
            roc_list,
        )
    else:
        return (best_weights / best_weights.sum()), best_feature_weights


def compute_feature_weights(n_columns, budget, feature_importances):
    max_importance = np.max(np.abs(feature_importances))
    budget_feature_importances = (feature_importances / max_importance) * budget
    budget_feature_importances = 1 - budget_feature_importances
    budget_feature_importances = (
        budget_feature_importances / np.sum(budget_feature_importances)
    ) * n_columns
    return budget_feature_importances




def train_pu_classifier(X_train, y_train, class_weight="balanced", random_state=None):
    """Train the positive unlabeled classifier

    :param X_train: Training features
    :param y_train: Training target
    :param class_weight: Sample weights, defaults to "balanced"
    :return: Trained positive unlabeled classifier
    """
    clf = RandomForestClassifier(
        class_weight=class_weight,
        n_estimators=25,
        n_jobs=1,
        min_weight_fraction_leaf=0.1,
        random_state=random_state,
    )
    return clf.fit(
        X_train,
        y_train,
    )
