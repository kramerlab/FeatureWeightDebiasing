import shap
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from tqdm import trange

from sklearn.model_selection import KFold, StratifiedKFold
from utils.metrics import train_svm_pu_classifier
from weighting_methods.feature_weighted_maximum_representative_subsampling import (
    initialize_dictionaries,
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
    hyperparameter=0.0,
    stratify_R=False,
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
    dropped_N = N[sample_weights != 0.0]
    all_predictions = np.zeros(len(dropped_N))
    y = N[target]
    target_sum = np.sum(y)
    if target_sum < n_splits:
        n_splits = target_sum
    elif (len(y) - target_sum) < n_splits:
        n_splits = len(y) - target_sum
    if n_splits in (1, 0):
        n_splits = 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if stratify_R:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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
            train_data[columns],
            train_data.label,
            random_state=random_state,
            C=hyperparameter,
        )
        test_data = pd.concat([N_test, R_test])
        distances = clf.decision_function(test_data[columns])
        all_predictions[test_indices_N] = distances[: len(N_test)]
        auroc_list.append(roc_auc_score(test_data.label, distances))

        abs_feature_importance = np.abs(clf.coef_[0])
        abs_feature_importance_list.append(abs_feature_importance)

    abs_mean_feature_importance = np.mean(abs_feature_importance_list, axis=0)
    abs_mean_feature_importance = abs_mean_feature_importance / np.sum(
        abs_mean_feature_importance
    )
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
    temperature=0.0,
    hyperparameter_list=[0.0],
    return_metrics=False,
    stratify_R=False,
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
    switched_dict = {}
    finished_dict = {}
    abs_feature_importance_dict = {}
    auroc_dict = {}

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
        switched_dict,
        auc_dict=auroc_dict,
        mrs_step=mrs_step,
    )

    for i in trange(number_of_iterations):
        for temperature in budgets:
            for hyperparameter in hyperparameter_list:
                if finished_dict[temperature][hyperparameter] and not return_metrics:
                    break
                splitter = "best" if temperature is None else "feature_weighted_best"
                drop_ids, _, auroc = mrs_step(
                    N=dropped_N,
                    R=R,
                    target=target,
                    columns=columns,
                    n_drop=drop,
                    random_state=random_generator.randint(max_int),
                    class_weight=class_weight,
                    n_splits=n_pu_splits,
                    feature_weight=np.array(
                        feature_weights_dict[temperature][hyperparameter]
                    ),
                    splitter=splitter,
                    sample_weights=sample_weights_dict[temperature][hyperparameter],
                    hyperparameter=hyperparameter,
                    stratify_R=stratify_R,
                )

                feature_weighted_aurocs_dict[temperature][hyperparameter].append(auroc)
                auc_difference = abs(auroc - 0.5)

                if return_metrics:
                    auroc_dict[temperature][hyperparameter].append(auroc)

                if (
                    auc_difference <= best_difference_dict[temperature][hyperparameter]
                    or (not switched_dict[temperature][hyperparameter] and auroc <= 0.5)
                ) and not finished_dict[temperature][hyperparameter]:
                    best_difference_dict[temperature][hyperparameter] = auc_difference
                    dropped_samples_dict[temperature][hyperparameter] = i * drop
                    sample_weights = sample_weights_dict[temperature][hyperparameter]
                    best_sample_weights_dict[temperature][hyperparameter] = (
                        (sample_weights / np.sum(sample_weights)).tolist().copy()
                    )
                    if not switched_dict[temperature][hyperparameter] and auroc <= 0.5:
                        switched_dict[temperature][hyperparameter] = True

                remaining = dropped_N[
                    sample_weights_dict[temperature][hyperparameter] != 0.0
                ]
                if (
                    len(remaining) <= drop
                    or len(remaining) <= n_pu_splits
                    or (auc_difference <= delta and early_stopping)
                    or switched_dict[temperature][hyperparameter]
                ):
                    finished_dict[temperature][hyperparameter] = True

                sample_weights_dict[temperature][hyperparameter][drop_ids] = 0

        if (
            all(all(finished.values()) for finished in finished_dict.values())
            and early_stopping
        ):
            break

    if return_metrics:
        return (
            auroc_dict,
            best_sample_weights_dict,
            feature_weights_dict,
            abs_feature_importance_dict,
        )
    else:
        return (
            best_sample_weights_dict,
            feature_weights_dict,
        )
