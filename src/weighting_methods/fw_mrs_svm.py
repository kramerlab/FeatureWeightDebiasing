import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from tqdm import trange

from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from utils.metrics import (
    compute_feature_weights_with_temperature,
    train_svm_pu_classifier,
)

# Used to draw radom states
max_int = 2**32 - 1


def mrs_step(
    N,
    R,
    target,
    columns,
    n_drop: int = 1,
    n_splits=5,
    random_state=None,
    feature_weight=None,
    sample_weights=None,
    C=None,
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
    skf = RepeatedStratifiedKFold(n_splits=n_splits, random_state=random_state)
    kf = RepeatedKFold(n_splits=n_splits, random_state=random_state)
    for (train_indices_N, test_indices_N), (train_indices_R, test_indices_R) in zip(
        skf.split(dropped_N, dropped_N[target]), kf.split(R)
    ):
        N_train, N_test = (
            dropped_N.iloc[train_indices_N],
            dropped_N.iloc[test_indices_N],
        )
        R_train, R_test = R.iloc[train_indices_R], R.iloc[test_indices_R]
        train_data = pd.concat([N_train, R_train])
        train_data[columns] = train_data[columns] * np.array(feature_weight)

        clf = train_svm_pu_classifier(
            train_data[columns], train_data.label, random_state=random_state, C=C
        )
        test_data = pd.concat([N_test, R_test])
        distances = clf.decision_function(test_data[columns])
        all_predictions[test_indices_N] = distances[: len(N_test)]
        auroc_list.append(roc_auc_score(test_data.label, distances))

        abs_feature_importance = np.abs(clf.coef_[0])
        abs_feature_importance_list.append(abs_feature_importance)

    abs_mean_feature_importance = np.mean(abs_feature_importance_list, axis=0)
    drop_ids = np.argpartition(all_predictions, -n_drop)[-n_drop:]

    return dropped_N.index[drop_ids], abs_mean_feature_importance, np.mean(auroc_list)


def calculate_feature_importance(test_N, clf, background=None):
    explainer = shap.KernelExplainer(clf.predict, background)
    explainer = explainer(test_N)
    shap_values = explainer.values[:, 1]
    abs_feature_importance = np.mean(np.abs(shap_values), axis=0)

    return abs_feature_importance


def mrs_without_cv(
    N,
    R,
    target,
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
    clf = train_svm_pu_classifier(
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


def fw_MRS_SVM(
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
    temperature=0.0,
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

    for C in hyperparameter_list:
        finished_dict[C] = False
        best_difference_dict[C] = np.inf
        auc_difference_dict[C] = 1
        dropped_samples_dict[C] = 0
        current_patience[C] = 0
        feature_weighted_aurocs_dict[C] = []
        sample_weights_dict[C] = np.ones(len(N))
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
            C=C,
        )
        feature_weights_dict[C] = compute_feature_weights_with_temperature(
            temperature, np.array(abs_feature_importance)
        ).tolist()

    for i in trange(number_of_iterations):
        for C in budgets:
            if finished_dict[C]:
                break
            drop_ids, abs_feature_importance, auroc = mrs_step(
                N=dropped_N,
                R=R,
                target=target,
                columns=columns,
                n_drop=drop,
                random_state=random_generator.randint(max_int),
                class_weight=class_weight,
                n_splits=n_pu_splits,
                feature_weight=feature_weights_dict[C],
                sample_weights=sample_weights_dict[C],
                C=C,
            )

            feature_weighted_aurocs_dict[C].append(auroc)

            auc_difference = abs(auroc - 0.5)

            if (
                (auc_difference + delta) <= best_difference_dict[C]
                or (not switched and auroc < 0.5)
            ) and not finished_dict[C]:
                best_difference_dict[C] = auc_difference
                dropped_samples_dict[C] = i * drop
                sample_weights = sample_weights_dict[C]
                best_sample_weights_dict[C] = (
                    (sample_weights / np.sum(sample_weights)).tolist().copy()
                )
                current_patience[C] = 0
                if not switched and auroc < 0.5:
                    switched = True
            else:
                current_patience[C] += 1

            sample_weights_dict[C][drop_ids] = 0
            remaining = N[sample_weights_dict[C] != 0.0]
            n_positive = np.count_nonzero(remaining[target])
            n_negative = len(remaining) - n_positive

            if (
                len(remaining) <= drop
                or (n_positive <= n_pu_splits or n_negative <= n_pu_splits)
                or (auc_difference <= delta and early_stopping)
                or (current_patience[C] == max_patience and early_stopping)
                or auroc < 0.5
            ):
                finished_dict[C] = True


        if all(finished_dict.values()) and early_stopping:
            break

    return (
        best_sample_weights_dict,
        feature_weights_dict,
    )
