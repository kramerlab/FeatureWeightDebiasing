import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import pdist

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
)


min_weights_fraction_leaf = [
    0.0,
    0.01,
    0.015,
    0.025,
    0.05,
    0.1,
    0.15,
    0.20,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
]

short_min_weights_fraction_leaf = [0.0, 0.01, 0.4]


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
    relative_bias = (abs(weighted_means - population_means) / population_means) * 100

    return relative_bias


def calculate_rbf_gamma(aggregate_set):
    """Calculate the gamma for the RBF-kernel

    :param aggregate_set: Aggregated data set
    :return: Gamma
    """
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
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def weighted_maximum_mean_discrepancy(
    x,
    y,
    sample_weights,
    feature_weights,
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
    return compute_weighted_maximum_mean_discrepancy(
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

    if n_n_rbf_matrix is None:
        n_n_rbf_matrix = weighted_rbf_kernel(
            n, feature_weights=feature_weights, gamma=gamma
        )
    weights_n_n = np.matmul(
        np.expand_dims(sample_weights, 1), np.expand_dims(sample_weights, 0)
    )
    n_n_mean = (weights_n_n * n_n_rbf_matrix).sum()

    r_r_rbf_matrix = (
        weighted_rbf_kernel(
            r,
            feature_weights=feature_weights,
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
        n_r_rbf_matrix = weighted_rbf_kernel(
            n, r, feature_weights=feature_weights, gamma=gamma
        )
    weight_matrix_n_r = np.matmul(
        np.expand_dims(sample_weights, 1), np.expand_dims(uniform_weights, 0)
    )
    n_r_mean = (weight_matrix_n_r * n_r_rbf_matrix).sum()

    mmd = np.sqrt(n_n_mean + r_r_mean - 2 * n_r_mean)
    return mmd


def weighted_rbf_kernel(
    X,
    Y=None,
    feature_weights=None,
    gamma=None,
):
    if feature_weights is None:
        feature_weights = np.ones(X.shape[1])
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(Y, pd.DataFrame):
        Y = Y.values
    if Y is None:
        Y = X
    difference = X[..., np.newaxis] - Y[..., np.newaxis].T
    K = (difference * difference) * feature_weights[..., np.newaxis]
    K = K.sum(1)
    K *= -gamma
    np.exp(K, K)
    return K


def compute_metrics(
    scaled_N,
    scaled_R,
    scaler,
    scale_columns,
    columns,
    sample_weights,
    feature_weights,
    gamma,
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
    scaled_N_dropped = scaled_N[columns].values
    scaled_R_dropped = scaled_R[columns].values

    weighted_mmd = weighted_maximum_mean_discrepancy(
        scaled_N_dropped,
        scaled_R_dropped,
        sample_weights,
        feature_weights,
        gamma,
    )

    unscaled_N = scaled_N.copy()
    unscaled_R = scaled_R.copy()
    unscaled_N[scale_columns] = scaler.inverse_transform(scaled_N[scale_columns])
    unscaled_R[scale_columns] = scaler.inverse_transform(scaled_R[scale_columns])

    sample_biases = compute_relative_bias(unscaled_N, unscaled_R, sample_weights)

    return (
        weighted_mmd,
        sample_biases,
    )


def compute_classification_metrics_tree(
    N,
    R,
    columns,
    sample_weights,
    feature_weights,
    label,
    random_state=None,
    draw_with_feature_weights=False,
):
    """Computes classification metrics for downstream tasks

    :param N: Non representative data set
    :param R: Representative data set
    :param columns: Columns used in the training
    :param weights: Computed sample weights
    :param label: Name of the target variable
    :return: Downstream classification metrics
    """
    clf = train_tree_classifier_auroc(
        N[columns],
        N[label],
        sample_weights,
        feature_weights,
        random_state=random_state,
        n_splits=5,
        speedup=False,
        draw_with_feature_weights=draw_with_feature_weights,
    )
    y_predictions = clf.predict_proba(R[columns])[:, 1]
    auroc_score = roc_auc_score(R[label], y_predictions)
    auprc = average_precision_score(R[label], y_predictions)

    return auroc_score, auprc


def compute_classification_metrics_random_forest(
    N,
    R,
    columns,
    sample_weights,
    feature_weights,
    label,
    draw_with_feature_weights=False,
    random_state=None,
):
    """Computes classification metrics for downstream tasks

    :param N: Non representative data set
    :param R: Representative data set
    :param columns: Columns used in the training
    :param weights: Computed sample weights
    :param label: Name of the target variable
    :return: Downstream classification metrics
    """
    clf = train_forest_classifier_auroc(
        N[columns],
        N[label],
        sample_weights,
        feature_weights,
        random_state=random_state,
        n_splits=5,
        draw_with_feature_weights=draw_with_feature_weights,
    )
    y_predictions = clf.predict_proba(R[columns])[:, 1]
    auroc_score = roc_auc_score(R[label], y_predictions)
    auprc = average_precision_score(R[label], y_predictions)

    return auroc_score, auprc


def train_classifier_auroc_feature_weighted(
    N,
    R,
    columns,
    feature_weights=None,
    draw_with_feature_weights=False,
    random_state=None,
    class_weight="balanced",
    splitter="feature_weighted_best",
    max_features="sqrt",
    faster=False,
):
    """Train a classifier to measure the auroc

    :param X_train: Training features
    :param y_train: Training targets
    :param weights: Sample weights, defaults to None
    :param speedup: If true, use only a subset of the cost complexities, defaults to True
    :param n_splits: Number of cross-validation iterations, defaults to 3
    :return: Trained classifier
    """
    data = pd.concat([N, R])
    X = data[columns]
    y = data.label
    weights = short_min_weights_fraction_leaf if faster else min_weights_fraction_leaf
    with Pool(len(weights)) as pool:
        results = pool.map(
            partial(
                oob_scoring,
                X_train=X,
                y_train=y,
                feature_weights=feature_weights,
                draw_with_feature_weights=draw_with_feature_weights,
                random_state=random_state,
                class_weight=class_weight,
                splitter=splitter,
                max_features=max_features,
            ),
            weights,
        )
    grids, aurocs = zip(*results)
    best_index = np.argmax(aurocs)
    return grids[best_index], aurocs[best_index]


def train_classifier_auroc_feature_weighted_cv(
    N,
    R,
    columns,
    feature_weights=None,
    draw_with_feature_weights=False,
    random_state=None,
    class_weight="balanced_subsample",
    splitter="feature_weighted_best",
    max_features="sqrt",
    faster=False,
    cv=5,
):
    """Train a classifier to measure the auroc

    :param X_train: Training features
    :param y_train: Training targets
    :param weights: Sample weights, defaults to None
    :param speedup: If true, use only a subset of the cost complexities, defaults to True
    :param n_splits: Number of cross-validation iterations, defaults to 3
    :return: Trained classifier
    """
    weights = short_min_weights_fraction_leaf if faster else min_weights_fraction_leaf
    # kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    data = pd.concat([N, R])
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        splitter=splitter,
        class_weight=class_weight,
        max_features=max_features,
        n_jobs=-1,
    )
    param_grid = {"min_weight_fraction_leaf": weights}
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv)
    grid.fit(
        data[columns],
        data.label,
        feature_weights=feature_weights,
        draw_with_feature_weights=draw_with_feature_weights,
    )
    best_min_weight = grid.best_params_["min_weight_fraction_leaf"]
    best_clf = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        splitter=splitter,
        class_weight=class_weight,
        max_features=max_features,
        n_jobs=-1,
        min_weight_fraction_leaf=best_min_weight,
        oob_score=roc_auc_score,
    )
    best_clf.fit(data[columns], data.label)
    auroc = best_clf.oob_score_

    return best_clf, auroc


def oob_scoring(
    min_weight_fraction_leaf,
    X_train,
    y_train,
    feature_weights,
    draw_with_feature_weights,
    random_state,
    class_weight,
    splitter,
    max_features,
):
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        splitter=splitter,
        class_weight=class_weight,
        oob_score=roc_auc_score,
        max_features=max_features,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        n_jobs=-1,
    )
    clf.fit(
        X_train,
        y_train,
        feature_weights=feature_weights,
        draw_with_feature_weights=draw_with_feature_weights,
    )
    current_auroc = clf.oob_score_
    return clf, current_auroc


def train_classifier_auroc(
    X_train,
    y_train,
    weights=None,
    speedup=True,
    n_splits=5,
    random_state=None,
    class_weight=None,
    **kwargs,
):
    """Train a classifier to measure the auroc

    :param X_train: Training features
    :param y_train: Training targets
    :param weights: Sample weights, defaults to None
    :param speedup: If true, use only a subset of the cost complexities, defaults to True
    :param cv: Number of cross-validation iterations, defaults to 3
    :return: Trained classifier
    """
    if weights is None:
        weights = np.ones(len(X_train)) / len(X_train)
    clf = DecisionTreeClassifier(
        random_state=np.random.RandomState(random_state),
        class_weight=class_weight,
    )
    path = clf.cost_complexity_pruning_path(X_train, y_train, sample_weight=weights)
    ccp_alphas = path.ccp_alphas
    ccp_alphas[ccp_alphas < 0] = 0
    ccp_alphas_unique = np.unique(ccp_alphas)

    if speedup:
        if len(ccp_alphas_unique) > 10:
            shortened_ccp_alphas_unique = ccp_alphas_unique[0::10]
            ccp_alphas_unique = np.append(
                ccp_alphas_unique[-10:], shortened_ccp_alphas_unique
            )

    param_grid = {"ccp_alpha": ccp_alphas_unique}
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=np.random.RandomState(random_state),
    )
    grid = GridSearchCV(
        clf,
        param_grid=param_grid,
        cv=cv,
        n_jobs=5,
        refit=True,
    )

    grid.fit(
        X_train,
        y_train,
        sample_weight=weights,
    )

    return grid


def compute_test_metrics_mrs_cv(
    data,
    columns,
    calculate_roc=False,
    weights=None,
    cv=5,
    speedup=True,
    random_state=None,
):
    """Compute test metrics for mrs

    :param data: Data set as pandas.DataFrame
    :param columns: Names of the columns use for training
    :param calculate_roc: If true, compute roc, defaults to False
    :param weights: Sample weights, defaults to None
    :param cv: Number of cross-validation iterations, defaults to 5
    :return: Test metrics for mrs
    """
    if weights is None:
        weights = np.ones(len(data)) / len(data)
    auroc_scores = []
    ifpr_list = []
    itpr_list = []
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    for train_indices, test_indices in kf.split(data[columns], data["label"]):
        train, test = data.iloc[train_indices], data.iloc[test_indices]
        train_weights = weights[train_indices]
        clf = train_classifier_auroc(
            train[columns],
            train.label,
            weights=train_weights,
            random_state=random_state,
            speedup=speedup,
        )
        y_predict = clf.predict_proba(test[columns])[:, 1]
        auroc = roc_auc_score(test.label, y_predict)
        auroc_scores.append(auroc)
        if calculate_roc:
            interpolated_fpr, interpolated_tpr = interpolate_roc(test.label, y_predict)
            ifpr_list.append(interpolated_fpr)
            itpr_list.append(interpolated_tpr)
    if calculate_roc:
        mean_ifpr_list, mean_itpr_list, std_tpr = calculate_mean_roc(
            ifpr_list, itpr_list
        )
        return np.mean(auroc_scores), mean_ifpr_list, mean_itpr_list, std_tpr
    else:
        return np.mean(auroc_scores)


def compute_test_metrics_mrs(
    N,
    R,
    columns,
    calculate_roc=False,
    random_state=None,
    feature_weights=None,
    draw_with_feature_weights=False,
    method=train_classifier_auroc,
    class_weight="balanced_subsample",
    faster=False,
    splitter="feature_weighted_best",
    max_features="sqrt",
):
    """Compute test metrics for mrs

    :param data: Data set as pandas.DataFrame
    :param columns: Names of the columns use for training
    :param calculate_roc: If true, compute roc, defaults to False
    :param weights: Sample weights, defaults to None
    :param cv: Number of cross-validation iterations, defaults to 3
    :return: Test metrics for mrs
    """
    data = pd.concat([N, R])
    clf, auroc = method(
        N,
        R,
        columns,
        feature_weights=feature_weights,
        draw_with_feature_weights=draw_with_feature_weights,
        random_state=random_state,
        class_weight=class_weight,
        faster=faster,
        splitter=splitter,
        max_features=max_features,
    )
    if calculate_roc:
        ifpr_list = []
        itpr_list = []
        y_predict = clf.predict_proba(data[columns])[:, 1]
        interpolated_fpr, interpolated_tpr = interpolate_roc(data.label, y_predict)
        ifpr_list.append(interpolated_fpr)
        itpr_list.append(interpolated_tpr)
        mean_ifpr_list, mean_itpr_list, std_tpr = calculate_mean_roc(
            ifpr_list, itpr_list
        )
        return auroc, mean_ifpr_list, mean_itpr_list, std_tpr
    else:
        return auroc


def train_pu_classifier(
    X_train, y_train, class_weight="balanced_subsample", random_state=None
):
    """Train the positive unlabeled classifier

    :param X_train: Training features
    :param y_train: Training target
    :param class_weight: Sample weights, defaults to "balanced"
    :return: Trained positive unlabeled classifier
    """
    clf = RandomForestClassifier(
        class_weight=class_weight,
        n_estimators=100,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


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


def train_tree_classifier_auroc(
    X_train,
    y_train,
    sample_weights=None,
    feature_weights=None,
    speedup=True,
    n_splits=3,
    random_state=None,
    draw_with_feature_weights=False,
    **kwargs,
):
    """Train a classifier to measure the auroc

    :param X_train: Training features
    :param y_train: Training targets
    :param weights: Sample weights, defaults to None
    :param speedup: If true, use only a subset of the cost complexities, defaults to True
    :param cv: Number of cross-validation iterations, defaults to 3
    :return: Trained classifier
    """
    if sample_weights is None:
        sample_weights = np.ones(len(X_train)) / len(X_train)
    max_features = "sqrt" if draw_with_feature_weights else None
    splitter = "feature_weighted_best" if draw_with_feature_weights else "best"
    clf = DecisionTreeClassifier(
        random_state=np.random.RandomState(random_state),
        max_features=max_features,
        splitter=splitter,
    )
    path = clf.cost_complexity_pruning_path(
        X_train,
        y_train,
        sample_weight=sample_weights,
        feature_weights=feature_weights,
        draw_with_feature_weights=True,
    )
    ccp_alphas = path.ccp_alphas
    ccp_alphas[ccp_alphas < 0] = 0
    ccp_alphas_unique = np.unique(ccp_alphas)

    if speedup:
        if len(ccp_alphas_unique) > 10:
            shortened_ccp_alphas_unique = ccp_alphas_unique[0::10]
            ccp_alphas_unique = np.append(
                ccp_alphas_unique[-10:], shortened_ccp_alphas_unique
            )

    param_grid = {"ccp_alpha": ccp_alphas_unique, "max_features": [max_features]}
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=np.random.RandomState(random_state),
    )
    grid = GridSearchCV(
        clf,
        param_grid=param_grid,
        cv=cv,
        n_jobs=5,
        refit=True,
    )

    return grid.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        feature_weights=feature_weights,
        draw_with_feature_weights=True,
    )


def train_forest_classifier_auroc(
    X_train,
    y_train,
    sample_weights=None,
    feature_weights=None,
    n_splits=5,
    draw_with_feature_weights=False,
    random_state=None,
):
    """Train a classifier to measure the auroc

    :param X_train: Training features
    :param y_train: Training targets
    :param weights: Sample weights, defaults to None
    :param speedup: If true, use only a subset of the cost complexities, defaults to True
    :param n_splits: Number of cross-validation iterations, defaults to 5
    :return: Trained classifier
    """
    if sample_weights is None:
        sample_weights = np.ones(len(X_train)) / len(X_train)

    max_features = (
        ["sqrt", "log2"] if draw_with_feature_weights else ["sqrt", None, "log2"]
    )
    splitter = "feature_weighted_best" if draw_with_feature_weights else "best"
    clf = RandomForestClassifier(
        random_state=random_state,
        splitter=splitter,
    )

    param_grid = {
        "max_features": max_features,
        "n_estimators": [50, 100, 200, 500],
        "min_weight_fraction_leaf": min_weights_fraction_leaf,
    }
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=np.random.RandomState(random_state),
    )
    grid = GridSearchCV(
        clf,
        param_grid=param_grid,
        cv=cv,
        n_jobs=5,
        refit=True,
    )
    clf = grid.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        feature_weights=feature_weights,
        draw_with_feature_weights=draw_with_feature_weights,
    )
    return clf


def calculate_mean_rocs(rocs):
    """Compute mean rocs

    :param rocs: Rocs list
    :return: Mean rocs
    """
    rocs = np.array(rocs, dtype=object)
    mean_rocs = []
    for i in range(rocs.shape[1]):
        rocs_at_iteration = rocs[:, i]
        mean_fpr, mean_tpr, std_tpr = calculate_mean_roc(
            rocs_at_iteration[:, 0], rocs_at_iteration[:, 1]
        )
        removed_samples = rocs_at_iteration[0, 3]
        mean_rocs.append((mean_fpr, mean_tpr, std_tpr, removed_samples))
    return mean_rocs


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
