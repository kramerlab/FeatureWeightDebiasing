import pandas as pd
import shap
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

from sklearn.metrics import make_scorer
from sklearn import set_config

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
    PredefinedSplit,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.svm import SVC, LinearSVC
from fairlearn.reductions import DemographicParity, ExponentiatedGradient

min_weight_fractions_leaf_list = [
    0.0,
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.2,
    0.3,
]

class_weight_list = [None, "balanced"]
param_grid = {
    "min_weight_fraction_leaf": min_weight_fractions_leaf_list,
    "class_weight": class_weight_list,
}


def compute_weighted_means(N, weights):
    """Compute the weighted mean

    :param N: Non-representative data set
    :param weights: Sample weights
    :return: Weighted mean
    """
    return np.average(N, weights=weights, axis=0)


def compute_relative_bias(N, R, sample_weights):
    """Compute the relative bias

    :param N: Non-representative data set
    :param R: Representative data set
    :param weights: Sample weights
    :return: Relative biases
    """
    weighted_means = compute_weighted_means(N, sample_weights)
    population_means = np.mean(R, axis=0)
    relative_bias = np.abs((weighted_means - population_means) / population_means * 100)

    return relative_bias


def calculate_rbf_gamma(aggregate_set, feature_weights=None):
    """Calculate the gamma for the RBF-kernel

    :param aggregate_set: Aggregated data set
    :return: Gamma
    """
    if feature_weights:
        feature_weights = (
            feature_weights / np.sum(feature_weights) * len(feature_weights)
        )
        aggregate_set = aggregate_set * np.sqrt(feature_weights)
    all_distances = pdist(aggregate_set, "euclid")
    sigma = np.median(all_distances)
    return 1 / (2 * (sigma**2))


def scale_df(df, columns):
    """Scale the data set

    :param df: Data set
    :param columns: Scaling columns
    :return: Scaled data set and scaler
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns]).copy()
    return df, scaler


def weighted_maximum_mean_discrepancy(
    x,
    y,
    sample_weights,
    feature_weights=None,
    gamma=None,
    x_x_rbf_matrix=None,
    y_y_rbf_matrix=None,
    x_y_rbf_matrix=None,
):
    """Wrapper function to calculate the MMD between a weighted data set and a uniform weighted data set

    :param x: The first data set
    :param y: The second data set
    :param weights: Weights for the first data set
    :param gamma: Gamma of the RBF kernel, defaults to None
    :param x_x_rbf_matrix: Pre-computed pairwise rbf matrix to save computing time, defaults to None
    :param y_y_rbf_matrix: Pre-computed pairwise rbf matrix to save computing time, defaults to None
    :param x_y_rbf_matrix: Pre-computed pairwise rbf matrix to save computing time, defaults to None
    :return: The MMD between a weighted data set and a uniform weighted reference data set
    """
    if gamma is None:
        gamma = calculate_rbf_gamma(np.append(x, y, axis=0))
    if feature_weights is None:
        return compute_weighted_maximum_mean_discrepancy(
            x,
            y,
            sample_weights,
            gamma,
            x_x_rbf_matrix,
            y_y_rbf_matrix,
            x_y_rbf_matrix,
        )
    else:
        return compute_feature_weighted_maximum_mean_discrepancy(
            x,
            y,
            sample_weights,
            feature_weights,
            gamma,
            x_x_rbf_matrix,
            y_y_rbf_matrix,
            x_y_rbf_matrix,
        )


def compute_weighted_maximum_mean_discrepancy(
    n,
    r,
    sample_weights,
    gamma,
    n_n_rbf_matrix=None,
    r_r_rbf_matrix=None,
    n_r_rbf_matrix=None,
):
    """_summary_

    :param gamma: _description_
    :param x: The first data set
    :param y: The second data set
    :param weights: Weights for the first data set (x)
    :param n_n_rbf_matrix: Pre-computed pairwise rbf matrix to save computing time, defaults to None
    :param r_r_rbf_matrix: Pre-computed pairwise rbf matrix to save computing time, defaults to None
    :param n_r_rbf_matrix: Pre-computed pairwise rbf matrix to save computing time, defaults to None
    :return: The MMD between a weighted data set and a uniform weighted reference data set
    """
    uniform_weights = np.ones(len(r)) / len(r)
    sample_weights = sample_weights / np.sum(sample_weights)

    if n_n_rbf_matrix is None:
        n_n_rbf_matrix = rbf_kernel(n, gamma=gamma)
    weights_n_n = np.matmul(
        np.expand_dims(sample_weights, 1), np.expand_dims(sample_weights, 0)
    )
    n_n_mean = (weights_n_n * n_n_rbf_matrix).sum()

    r_r_rbf_matrix = (
        rbf_kernel(
            r,
            gamma=gamma,
        )
        if r_r_rbf_matrix is None
        else r_r_rbf_matrix
    )
    weight_matrix_r_r = np.matmul(
        np.expand_dims(uniform_weights, 1), np.expand_dims(uniform_weights, 0)
    )
    r_r_mean = (weight_matrix_r_r * r_r_rbf_matrix).sum()

    if n_r_rbf_matrix is None:
        n_r_rbf_matrix = rbf_kernel(n, r, gamma=gamma)
    weight_matrix_n_r = np.matmul(
        np.expand_dims(sample_weights, 1), np.expand_dims(uniform_weights, 0)
    )
    n_r_mean = (weight_matrix_n_r * n_r_rbf_matrix).sum()

    mmd = np.sqrt(n_n_mean + r_r_mean - 2 * n_r_mean)
    return mmd


def compute_feature_weighted_maximum_mean_discrepancy(
    n,
    r,
    sample_weights,
    feature_weights,
    gamma,
    n_n_rbf_matrix=None,
    r_r_rbf_matrix=None,
    n_r_rbf_matrix=None,
):
    """_summary_

    :param gamma: _description_
    :param x: The first data set
    :param y: The second data set
    :param weights: Weights for the first data set (x)
    :param n_n_rbf_matrix: Pre-computed pairwise rbf matrix to save computing time, defaults to None
    :param r_r_rbf_matrix: Pre-computed pairwise rbf matrix to save computing time, defaults to None
    :param n_r_rbf_matrix: Pre-computed pairwise rbf matrix to save computing time, defaults to None
    :return: The MMD between a weighted data set and a uniform weighted reference data set
    """
    uniform_weights = np.ones(len(r)) / len(r)
    sample_weights = sample_weights / np.sum(sample_weights)
    feature_weights = feature_weights / np.sum(feature_weights) * len(feature_weights)

    weighted_n = n * np.sqrt(feature_weights)
    weighted_r = r * np.sqrt(feature_weights)

    if n_n_rbf_matrix is None:
        n_n_rbf_matrix = rbf_kernel(weighted_n, gamma=gamma)
    weights_n_n = np.matmul(
        np.expand_dims(sample_weights, 1), np.expand_dims(sample_weights, 0)
    )
    n_n_mean = (weights_n_n * n_n_rbf_matrix).sum()

    r_r_rbf_matrix = (
        rbf_kernel(
            weighted_r,
            gamma=gamma,
        )
        if r_r_rbf_matrix is None
        else r_r_rbf_matrix
    )
    weight_matrix_r_r = np.matmul(
        np.expand_dims(uniform_weights, 1), np.expand_dims(uniform_weights, 0)
    )
    r_r_mean = (weight_matrix_r_r * r_r_rbf_matrix).sum()

    if n_r_rbf_matrix is None:
        n_r_rbf_matrix = rbf_kernel(weighted_n, weighted_r, gamma=gamma)
    weight_matrix_n_r = np.matmul(
        np.expand_dims(sample_weights, 1), np.expand_dims(uniform_weights, 0)
    )
    n_r_mean = (weight_matrix_n_r * n_r_rbf_matrix).sum()

    mmd = np.sqrt(n_n_mean + r_r_mean - 2 * n_r_mean)
    return mmd


def compute_metrics(
    scaled_N,
    scaled_R,
    scaler,
    columns,
    target,
    sample_weights_list,
    feature_weights,
    gamma,
    return_sample_weights=False,
):
    """Computes the metrics for an experiment

    :param scaled_N: Standardized non-representative data set
    :param scaled_R: standardized representative data set
    :param weights: Sample weights
    :param scaler: Standard Scaler
    :param scale_columns: Names of scaled columns
    :param columns: Names of columns used for training
    :param gamma: Gamma for the rbf kernel
    :return: Result metrics
    """
    final_wasserstein_distances = []
    N_dropped = scaled_N[columns].values
    R_dropped = scaled_R[columns].values
    columns_and_target = np.append(columns, target)

    final_wasserstein_distances = []
    weighted_mmd = weighted_maximum_mean_discrepancy(
        N_dropped,
        R_dropped,
        sample_weights_list,
        feature_weights,
        gamma=gamma,
    )

    for feature_name in columns_and_target:
        u_values = scaled_N[feature_name].values
        v_values = scaled_R[feature_name].values
        wasserstein_distance_value = wasserstein_distance(
            u_values, v_values, sample_weights_list
        )
        final_wasserstein_distances.append(wasserstein_distance_value)

    unscaled_N = scaled_N.copy()
    unscaled_R = scaled_R.copy()
    unscaled_N[columns] = scaler.inverse_transform(scaled_N[columns])
    unscaled_R[columns] = scaler.inverse_transform(scaled_R[columns])
    final_sample_biases = compute_relative_bias(
        unscaled_N[columns_and_target],
        unscaled_R[columns_and_target],
        sample_weights_list,
    )
    best_sample_weights = sample_weights_list

    if return_sample_weights:
        return (
            weighted_mmd,
            final_sample_biases,
            final_wasserstein_distances,
            best_sample_weights,
        )
    else:
        return (
            weighted_mmd,
            final_sample_biases,
            final_wasserstein_distances,
        )


def compute_classification_metrics_random_forest(
    N,
    R,
    T,
    columns,
    sample_weights_list,
    feature_weights,
    target,
    random_state=None,
    n_splits=5,
    splitter="feature_weighted_best",
    n_estimators=500,
    compute_feature_importance=True,
    draw_with_feature_weights=False,
    max_features="sqrt",
):
    """Computes classification metrics for downstream tasks

    :param N: Non representative data set
    :param R: Representative data set
    :param columns: Columns used in the training
    :param weights: Computed sample weights
    :param label: Name of the target variable
    :return: Downstream classification metrics
    """

    if isinstance(sample_weights_list, dict):
        best_clf = None
        best_score = -1
        best_sample_weights = None
        best_temperature = -1
        best_hyperparameter = -1
        for temperature in sample_weights_list.keys():
            for hyperparameter, sample_weights in sample_weights_list[
                temperature
            ].items():
                feature_weight = feature_weights[temperature][hyperparameter]
                for train_sample_weights in sample_weights.values():
                    clf, score = train_random_forest_classifier(
                        N[columns].values,
                        N[target].values,
                        R[columns].values,
                        train_sample_weights,
                        np.array(feature_weight),
                        random_state=random_state,
                        n_splits=n_splits,
                        draw_with_feature_weights=draw_with_feature_weights,
                        splitter=splitter,
                        n_estimators=n_estimators,
                        max_features=max_features,
                    )

                    if score > best_score:
                        best_feature_weight = feature_weight
                        best_score = score
                        best_clf = clf
                        best_sample_weights = train_sample_weights
                        best_temperature = temperature
                        best_hyperparameter = hyperparameter
                    if best_score == 1:
                        break
    else:
        best_temperature = 0
        train_sample_weights = sample_weights_list.copy()

        best_clf, best_score = train_random_forest_classifier(
            N[columns].values,
            N[target].values,
            R[columns].values,
            train_sample_weights,
            np.array(feature_weights),
            random_state=random_state,
            n_splits=n_splits,
            draw_with_feature_weights=draw_with_feature_weights,
            splitter=splitter,
            n_estimators=n_estimators,
            max_features=max_features,
        )
        best_hyperparameter = None
        best_sample_weights = sample_weights_list
        best_feature_weight = feature_weights

    if best_clf is not None:
        y_probabilitites = best_clf.predict_proba(T[columns].values)[:, 1]
        y_predictions = best_clf.predict(T[columns].values)
        fpr, tpr, _ = roc_curve(T[target], y_probabilitites)
    else:
        y_probabilitites = np.ones(len(T))
        y_predictions = np.ones(len(T))
        fpr = np.ones(len(T))
        tpr = np.ones(len(T))

    if compute_feature_importance and best_clf is not None:
        abs_feature_importance = calculate_feature_importance(
            T[columns].values,
            best_clf,
        )
    else:
        abs_feature_importance = np.ones_like(columns)

    auroc_score = roc_auc_score(T[target], y_probabilitites)
    auprc = average_precision_score(T[target], y_probabilitites)
    accuracy = accuracy_score(T[target], y_predictions)

    return (
        auroc_score,
        auprc,
        accuracy,
        best_sample_weights,
        best_feature_weight,
        abs_feature_importance,
        (fpr.tolist(), tpr.tolist()),
        best_temperature,
        best_hyperparameter,
        best_clf,
        best_score,
    )


def compute_decomposition_metrics_random_forest(
    N,
    T,
    columns,
    sample_weights_list,
    feature_weights,
    label,
    random_state=None,
    n_splits=5,
    splitter="feature_weighted_best",
    n_estimators=500,
    draw_with_feature_weights=False,
    max_features="sqrt",
):
    """Computes classification metrics for downstream tasks

    :param N: Non representative data set
    :param R: Representative data set
    :param columns: Columns used in the training
    :param weights: Computed sample weights
    :param label: Name of the target variable
    :return: Downstream classification metrics
    """

    if isinstance(sample_weights_list, dict):
        best_clf = None
        best_score = -1
        for temperature in sample_weights_list.keys():
            for hyperparameter, sample_weights in sample_weights_list[
                temperature
            ].items():
                feature_weight = feature_weights[temperature][hyperparameter]
                for train_sample_weights in sample_weights.values():

                    clf, score = train_random_forest_classifier(
                        N[columns].values,
                        N[label].values,
                        None,
                        np.array(train_sample_weights),
                        np.array(feature_weight),
                        random_state=random_state,
                        n_splits=n_splits,
                        draw_with_feature_weights=draw_with_feature_weights,
                        splitter=splitter,
                        n_estimators=n_estimators,
                        max_features=max_features,
                    )
                    if score > best_score:
                        best_score = score
                        best_clf = clf
    else:
        train_sample_weights = sample_weights_list.copy()

        best_clf, _ = train_random_forest_classifier(
            N[columns].values,
            N[label].values,
            None,
            np.array(train_sample_weights),
            np.array(feature_weights),
            random_state=random_state,
            n_splits=n_splits,
            draw_with_feature_weights=draw_with_feature_weights,
            splitter=splitter,
            n_estimators=n_estimators,
            max_features=max_features,
        )

    y_predictions = best_clf.predict(T[columns].values)
    probabilities = best_clf.predict_proba(T[columns].values)[:, 1]

    return y_predictions, probabilities


def train_pu_classifier(
    X,
    y,
    n_estimators=200,
    feature_weight=None,
    random_state=None,
    splitter="feature_weighted_best",
    hyperparameter=0.0,
    class_weight="balanced",
):
    """Train the positive unlabeled classifier

    :param X_train: Training features
    :param y_train: Training target
    :param class_weight: Sample weights, defaults to "balanced"
    :return: Trained positive unlabeled classifier
    """
    draw_with_feature_weight = False if feature_weight is None else True
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        min_weight_fraction_leaf=hyperparameter,
        splitter=splitter,
        n_jobs=-1,
        class_weight=class_weight,
    )

    return clf.fit(
        X,
        y,
        draw_with_feature_weights=draw_with_feature_weight,
        feature_weights=feature_weight,
    )


def train_pu_classifier_mrs(
    X,
    y,
    n_estimators=200,
    random_state=None,
    hyperparameter=0.0,
):
    """Train the positive unlabeled classifier

    :param X_train: Training features
    :param y_train: Training target
    :param class_weight: Sample weights, defaults to "balanced"
    :return: Trained positive unlabeled classifier
    """

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        min_weight_fraction_leaf=hyperparameter,
        n_jobs=-1,
        splitter="feature_weighted_best",
    )

    return clf.fit(
        X,
        y,
        draw_with_feature_weights=False,
    )


def interpolate_roc(y_test, y_predict):
    """Interpolate rocs

    :param y_test: True test targets
    :param y_predict: Predicted test targets
    :return: Interpolated roc
    """
    interpolation_points = 250
    interpolated_fpr = np.linspace(0, 1, interpolation_points)
    fpr, tpr, _ = roc_curve(y_test, y_predict)
    interpolated_tpr = np.interp(interpolated_fpr, fpr, tpr)
    interpolated_tpr[0] = 0.0
    return interpolated_fpr, interpolated_tpr


def train_random_forest_classifier(
    X,
    y,
    R,
    sample_weights,
    feature_weights=None,
    n_splits=5,
    draw_with_feature_weights=False,
    random_state=None,
    splitter="feature_weighted_best",
    n_estimators=500,
    max_features="sqrt",
    scoring="roc_auc",
    **kwargs,
):
    """Train a classifier to measure the auroc

    :param X_train: Training features
    :param y_train: Training targets
    :param weights: Sample weights, defaults to None
    :param speedup: If true, use only a subset of the cost complexities, defaults to True
    :param n_splits: Number of cross-validation iterations, defaults to 5
    :return: Trained classifier
    """

    target_sum = np.sum(y)
    if target_sum < n_splits:
        n_splits = target_sum
    elif (len(y) - target_sum) < n_splits:
        n_splits = len(y) - target_sum
    if n_splits in (1, 0):
        n_splits = 2
    skf = StratifiedKFold(
        n_splits=int(n_splits), shuffle=True, random_state=random_state
    )

    if draw_with_feature_weights:
        feature_weights = np.array(feature_weights)

    clf = RandomForestClassifier(
        random_state=random_state,
        splitter=splitter,
        n_estimators=n_estimators,
        max_features=max_features,
    )

    grid_cv = GridSearchCV(
        clf,
        param_grid,
        cv=skf,
        n_jobs=-1,
        scoring=scoring,
        refit=True,
    )

    grid_cv.fit(
        X,
        y,
        sample_weight=sample_weights,
        feature_weights=feature_weights,
        draw_with_feature_weights=draw_with_feature_weights,
    )
    return grid_cv.best_estimator_, grid_cv.best_score_


def calculate_mean_rocs(rocs_dict_list):
    """Compute mean rocs

    :param rocs: Rocs list
    :return: Mean rocs
    """
    mean_rocs_dict = {hyperparameter: [] for hyperparameter in rocs_dict_list[0].keys()}
    for i, hyperparameter in enumerate(rocs_dict_list[0].keys()):
        rocs_list = []
        for dictionary in rocs_dict_list:
            dictionary = {float(k): v for k, v in dictionary.items()}
            rocs_list.append(dictionary[float(hyperparameter)])
        for i in range(len(rocs_list[0])):
            rocs_at_iteration = [rocs[i] for rocs in rocs_list]
            fpr = [rocs[0] for rocs in rocs_at_iteration]
            tpr = [rocs[1] for rocs in rocs_at_iteration]
            mean_fpr, mean_tpr, std_tpr = calculate_mean_roc(fpr, tpr)
            removed_samples = rocs_at_iteration[0][2]
            mean_rocs_dict[hyperparameter].append(
                (mean_fpr, mean_tpr, std_tpr, removed_samples)
            )
    return mean_rocs_dict


def calculate_mean_roc(interpolated_fpr, interpolated_tpr):
    """Compute mean roc

    :param interpolated_fpr: Interpolated false positive rate
    :param interpolated_tpr: Interpolated true positive rate
    :return: Mean roc
    """
    mean_fpr = np.mean(interpolated_fpr, axis=0)
    mean_tpr = np.mean(interpolated_tpr, axis=0)
    std_tpr = np.std(interpolated_tpr, axis=0)
    return mean_fpr, mean_tpr, std_tpr


def compute_feature_weights_with_temperature(temperature, feature_importance):
    """_summary_

    :param temperature: _description_
    :param feature_importance: _description_
    :return: _description_
    """
    if temperature == 0.0:
        return np.ones(len(feature_importance)) / len(feature_importance)
    feature_weights = np.exp(np.array(-feature_importance) / temperature)
    return feature_weights / np.sum(feature_weights)


def calculate_feature_importance(test_N, clf, background=None):
    explainer = shap.TreeExplainer(clf, data=background)
    explainer = explainer(test_N, check_additivity=False)
    shap_values = np.abs(explainer.values[:, :, 1])
    abs_feature_importance = np.mean(shap_values, axis=0)

    return abs_feature_importance


def train_svm_pu_classifier(X, y, random_state=None, C=1, class_weight="balanced"):
    """Train the positive unlabeled classifier

    :param X_train: Training features
    :param y_train: Training target
    :param class_weight: Sample weights, defaults to "balanced"
    :return: Trained positive unlabeled classifier
    """
    clf = LinearSVC(
        dual="auto", random_state=random_state, C=C, class_weight=class_weight
    )

    clf.fit(
        X,
        y,
    )

    return clf


def train_svc(
    X,
    y,
    sample_weights,
    feature_weights=None,
    n_splits=5,
    draw_with_feature_weights=False,
    random_state=None,
    **kwargs,
):
    """Train a classifier to measure the auroc

    :param X_train: Training features
    :param y_train: Training targets
    :param weights: Sample weights, defaults to None
    :param speedup: If true, use only a subset of the cost complexities, defaults to True
    :param n_splits: Number of cross-validation iterations, defaults to 5
    :return: Trained classifier
    """

    if draw_with_feature_weights:
        X = X * feature_weights

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    param_grid = {
        "kernel": ["linear", "rbf"],
        "C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e2],
    }
    clf = SVC(
        random_state=random_state,
    )
    grid_cv = GridSearchCV(
        clf,
        param_grid,
        cv=skf,
        n_jobs=-1,
        scoring="roc_auc",
        refit=True,
    )

    grid_cv.fit(
        X,
        y,
        sample_weight=sample_weights,
    )

    return grid_cv.best_estimator_, grid_cv.best_score_


def compute_classification_metrics_random_forest_perfect(
    N,
    T,
    columns,
    target,
    random_state=None,
    n_splits=5,
    n_estimators=500,
):
    """Computes classification metrics for downstream tasks

    :param N: Non representative data set
    :param R: Representative data set
    :param columns: Columns used in the training
    :param weights: Computed sample weights
    :param label: Name of the target variable
    :return: Downstream classification metrics
    """

    clf = RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
    )
    skf = StratifiedKFold(
        n_splits=int(n_splits), shuffle=True, random_state=random_state
    )
    grid_cv = GridSearchCV(
        clf,
        param_grid,
        cv=skf,
        n_jobs=-1,
        scoring="roc_auc",
        refit=True,
    )

    grid_cv.fit(
        N[columns].values,
        N[target],
    )

    y_probabilitites = grid_cv.predict_proba(T[columns].values)[:, 1]

    auroc = roc_auc_score(T[target], y_probabilitites)
    auprc = average_precision_score(T[target], y_probabilitites)

    return (
        auroc,
        auprc,
    )


pad_param_grid = {"C": np.logspace(-5, 5, 11)}


def compute_pad(
    N,
    R,
    columns,
    sample_weights,
    feature_weights,
    seed,
    n_repeats=10,
    sample=False,
):
    set_config(enable_metadata_routing=True)
    sample_weights_copy = np.array(sample_weights).copy()
    mask = sample_weights_copy != 0
    sample_weights_copy = sample_weights_copy[mask]
    N_copy = N[mask].copy()
    N_copy["domain"] = 0
    R_copy = R.copy()
    R_copy["domain"] = 1
    min_size = min(len(R_copy), len(N_copy))
    R_sample_weights = np.ones(len(R_copy)) / len(R_copy)

    balanced_accuracy = make_scorer(
        balanced_accuracy_score, greater_is_better=True, response_method="predict"
    ).set_score_request(sample_weight=True)
    random_state = np.random.RandomState(seed=seed)

    error_sum = 0.0
    for _ in range(n_repeats):
        if sample:
            N_indices = np.random.choice(len(N_copy), min_size, replace=False)
            N_sampled = N_copy.iloc[N_indices]
            N_sample_weights = sample_weights_copy[N_indices]
            N_sample_weights = N_sample_weights / np.sum(N_sample_weights)
        else:
            N_sampled = N_copy
            N_sample_weights = sample_weights_copy / np.sum(sample_weights_copy)

        all_sample_weights = np.concatenate([N_sample_weights, R_sample_weights])
        data = pd.concat([N_sampled, R_copy]).reset_index().copy()
        X = data[columns]
        y = data["domain"]
        X = X * feature_weights
        best_error = 1
        X_train, _, _, _ = train_test_split(
            X, y, test_size=0.5, random_state=random_state, stratify=y
        )
        test_fold = np.zeros(len(X))
        test_fold[X_train.index] = -1
        predefined_split = PredefinedSplit(test_fold)

        skf = GridSearchCV(
            LinearSVC(dual="auto").set_fit_request(sample_weight=True),
            param_grid=pad_param_grid,
            cv=predefined_split,
            refit=False,
            scoring=balanced_accuracy,
        )
        skf.fit(
            X,
            y,
            sample_weight=all_sample_weights,
        )
        accuracy = skf.best_score_
        error = 1 - accuracy

        if error > 0.5:
            error = 1.0 - error

        best_error = min(best_error, error)

        error_sum += best_error
    mean_error = error_sum / n_repeats
    set_config(enable_metadata_routing=False)

    return 2 * (1 - 2 * mean_error)


def compute_domain_classifier_auroc(
    N, R, columns, sample_weights, feature_weights, seed, n_repeats=10
):
    set_config(enable_metadata_routing=True)
    sample_weights_copy = np.array(sample_weights).copy()
    mask = sample_weights_copy != 0
    sample_weights_copy = sample_weights_copy[mask]
    N_copy = N[mask].copy()
    N_copy["domain"] = 0
    R_copy = R.copy()
    R_copy["domain"] = 1

    N_sample_weights = sample_weights_copy / np.sum(sample_weights_copy)
    R_sample_weights = np.ones(len(R_copy)) / len(R_copy)
    all_sample_weights = np.concatenate([N_sample_weights, R_sample_weights])
    random_state = np.random.RandomState(seed=seed)

    balanced_auroc = make_scorer(
        roc_auc_score,
        greater_is_better=True,
        response_method="predict_proba",
    ).set_score_request(sample_weight=True)

    auroc_sum = 0.0
    for _ in range(n_repeats):

        data = pd.concat([N_copy, R_copy]).reset_index().copy()
        X = data[columns]
        y = data["domain"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=random_state, stratify=y
        )

        train_sample_weights = all_sample_weights[X_train.index]
        test_sample_weights = all_sample_weights[X_test.index]

        skf = GridSearchCV(
            RandomForestClassifier(splitter="feature_weighted_best").set_fit_request(
                sample_weight=True,
                feature_weights=True,
                draw_with_feature_weights=True,
            ),
            param_grid=param_grid,
            refit=True,
            scoring=balanced_auroc,
        )
        skf.fit(
            X_train,
            y_train,
            sample_weight=train_sample_weights,
            feature_weights=np.array(feature_weights),
            draw_with_feature_weights=True,
        )
        proba_predictions = skf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(
            y_test, proba_predictions, sample_weight=test_sample_weights
        )

        if auroc < 0.5:
            auroc = 1.0 - auroc

        auroc_sum += auroc
    mean_auroc = auroc_sum / n_repeats
    set_config(enable_metadata_routing=False)
    return mean_auroc
