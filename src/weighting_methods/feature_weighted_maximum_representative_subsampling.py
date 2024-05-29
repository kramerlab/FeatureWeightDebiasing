import numpy as np
import pandas as pd
from tqdm import trange
import shap

from sklearn.model_selection import KFold
from utils.metrics import (
    compute_test_metrics_fw_mrs,
    train_fw_mrs_pu_classifier,
    train_feature_weighted_random_forest,
)

# Used to draw radom states
max_int = 2**32 - 1


def mrs(
    N,
    R,
    columns,
    n_drop: int = 1,
    n_splits=5,
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
    for N_train_index, N_test_index in kf.split(N):
        N_train, N_test = N.iloc[N_train_index], N.iloc[N_test_index]
        train = pd.concat([N_train, R])
        clf = train_fw_mrs_pu_classifier(
            train[columns],
            train.label,
            class_weight=class_weights,
            random_state=random_state,
        )
        predictions = clf.predict_proba(N_test[columns])[:, 1]
        all_predictions[N_test_index] = predictions
        feature_importance = calculate_feature_importance(
            columns,
            test_N=N_test,
            clf=clf,
        )
        feature_importance_list.append(feature_importance)

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
    clf = train_fw_mrs_pu_classifier(
        data[columns],
        data.label,
        class_weight=class_weight,
        random_state=random_state,
    )
    predictions = clf.predict_proba(N[columns])[:, 1]
    feature_importance = calculate_feature_importance(
        columns,
        test_N=N,
        clf=clf,
    )

    drop_ids = np.argpartition(predictions, -n_drop)[-n_drop:]
    drop_index = N.index[drop_ids]

    return drop_index, feature_importance


def calculate_feature_importance(columns, test_N, clf):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(test_N[columns], tree_limit=-1)
    return np.average(np.abs(shap_values[1]), axis=0)


def compute_feature_weights_with_temperature(temperature, feature_importance):
    """_summary_

    :param temperature: _description_
    :param feature_importance: _description_
    :return: _description_
    """
    if temperature is None or temperature == 0.0:
        return np.ones(len(feature_importance)) / len(feature_importance)
    feature_weights = np.exp(-feature_importance / temperature)
    return feature_weights / np.sum(feature_weights)


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
    columns,
    delta=0.02,
    early_stopping=False,
    drop=1,
    budgets=[1.0],
    random_generator=None,
    class_weight="balanced",
    return_auroc=False,
    n_test_splits=10,
    n_pu_splits=5,
    max_patience=10,
    validation_method="random_forest",
    splitter="feature_weighted_best",
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
    number_of_iterations = (len(N) - (n_test_splits + 1)) // drop
    dropped_N = N.copy().reset_index(drop=True)
    sample_weights = np.ones(len(N))
    feature_weights = np.ones(len(columns))
    feature_importance_list = []
    feature_weighted_aurocs_dict = {}
    feature_weights_dict = {}
    dropped_samples_dict = {}
    finished_dict = {}
    best_difference_dict = {}
    best_inverse_feature_weights_dict = {}
    best_sample_weights_dict = {}
    dropped_samples_dict = {}
    best_feature_weights_dict = {}
    auc_difference_dict = {}
    current_patience_dict = {}

    finished_dict = {}
    for temperature in budgets:
        finished_dict[temperature] = False
        best_difference_dict[temperature] = np.inf
        auc_difference_dict[temperature] = 1
        dropped_samples_dict[temperature] = 0
        current_patience_dict[temperature] = 0

    feature_weights = np.ones(len(columns))
    rand_int = random_generator.randint(max_int)

    for i in trange(number_of_iterations):
        rand_int = random_generator.randint(max_int)
        drop_ids, feature_importance = mrs(
            N=dropped_N,
            R=R,
            columns=columns,
            n_drop=drop,
            random_state=rand_int,
            class_weight=class_weight,
            n_splits=n_pu_splits,
        )
        feature_importance_list.append(feature_importance.tolist())

        for budget in budgets:
            feature_weights = compute_feature_weights_with_temperature(
                budget, feature_importance
            )

            auroc = compute_test_metrics_fw_mrs(
                dropped_N,
                R,
                columns,
                random_state=rand_int,
                feature_weights=feature_weights,
                method=train_feature_weighted_random_forest,
                class_weight="balanced",
                max_features="sqrt",
                splitter=splitter,
                n_splits_test=n_test_splits,
                n_estimators=n_estimators,
                draw_with_feature_weights=True,
            )

            if budget not in feature_weighted_aurocs_dict:
                feature_weighted_aurocs_dict[budget] = []
                feature_weights_dict[budget] = []
            feature_weighted_aurocs_dict[budget].append(auroc)
            feature_weights_dict[budget].append(list(feature_weights))

            auc_difference = abs(auroc - 0.5)

            if (auc_difference + delta) <= best_difference_dict[
                temperature
            ] and not finished_dict[temperature]:
                best_difference_dict[temperature] = auc_difference
                dropped_samples_dict[temperature] = i * drop
                best_sample_weights_dict[temperature] = (
                    sample_weights / np.sum(sample_weights)
                ).tolist()
                best_feature_weights_dict[temperature] = feature_weights.tolist().copy()
                best_inverse_feature_weights_dict[temperature] = (
                    compute_feature_weights_with_temperature(
                        temperature, -feature_importance
                    )
                ).tolist()
                current_patience_dict[temperature] += 0
            else:
                current_patience_dict[temperature] += 1
            if (
                len(dropped_N) <= drop
                or len(dropped_N) <= n_test_splits
                or (
                    current_patience_dict[temperature] >= max_patience
                    and early_stopping
                )
                or (auc_difference <= delta and early_stopping)
            ):
                finished_dict[temperature] = True

        if all(finished_dict.values()):
            break

        dropped_N = dropped_N.drop(drop_ids)
        sample_weights[drop_ids] = 0.0

    if return_auroc:
        return (
            feature_weighted_aurocs_dict,
            feature_importance_list,
            feature_weights_dict,
            dropped_samples_dict,
            sample_weights,
        )

    else:
        return (
            best_sample_weights_dict,
            best_feature_weights_dict,
            best_inverse_feature_weights_dict,
        )
