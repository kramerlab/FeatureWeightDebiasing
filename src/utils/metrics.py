import pandas as pd
import shap
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from sklearn.svm import SVC, LinearSVC


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
    df[columns] = scaler.fit_transform(df[columns]).copy()
    return df, scaler


def weighted_maximum_mean_discrepancy(
    x,
    y,
    sample_weights,
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


def compute_metrics(
    scaled_N,
    scaled_R,
    scaler,
    scale_columns,
    columns,
    sample_weights_list,
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
    if isinstance(sample_weights_list, dict):
        best_weighted_mmd = np.inf
        best_sample_biases = np.inf
        best_wasserstein_distances = np.inf
        sample_weights_list = sample_weights_list[0.0]
        for sample_weights in sample_weights_list.values():
            wasserstein_distances = []
            scaled_N_dropped = scaled_N[columns].values
            scaled_R_dropped = scaled_R[columns].values

            weighted_mmd = weighted_maximum_mean_discrepancy(
                scaled_N_dropped,
                scaled_R_dropped,
                sample_weights,
                gamma,
            )

            for i in range(scaled_N.values.shape[1]):
                u_values = scaled_N.values[:, i]
                v_values = scaled_R.values[:, i]
                wasserstein_distance_value = wasserstein_distance(
                    u_values, v_values, sample_weights
                )
                wasserstein_distances.append(wasserstein_distance_value)

            unscaled_N = scaled_N.copy()
            unscaled_R = scaled_R.copy()
            unscaled_N[scale_columns] = scaler.inverse_transform(
                scaled_N[scale_columns]
            )
            unscaled_R[scale_columns] = scaler.inverse_transform(
                scaled_R[scale_columns]
            )
            sample_biases = compute_relative_bias(
                unscaled_N, unscaled_R, sample_weights
            )
            if weighted_mmd < best_weighted_mmd:
                best_weighted_mmd = weighted_mmd
                best_sample_biases = sample_biases
                best_wasserstein_distances = wasserstein_distances

    else:
        best_wasserstein_distances = []
        scaled_N_dropped = scaled_N[columns].values
        scaled_R_dropped = scaled_R[columns].values

        best_weighted_mmd = weighted_maximum_mean_discrepancy(
            scaled_N_dropped,
            scaled_R_dropped,
            sample_weights_list,
            gamma,
        )

        for i in range(scaled_N.values.shape[1]):
            u_values = scaled_N.values[:, i]
            v_values = scaled_R.values[:, i]
            wasserstein_distance_value = wasserstein_distance(
                u_values, v_values, sample_weights_list
            )
            best_wasserstein_distances.append(wasserstein_distance_value)

        unscaled_N = scaled_N.copy()
        unscaled_R = scaled_R.copy()
        unscaled_N[scale_columns] = scaler.inverse_transform(scaled_N[scale_columns])
        unscaled_R[scale_columns] = scaler.inverse_transform(scaled_R[scale_columns])
        best_sample_biases = compute_relative_bias(
            unscaled_N, unscaled_R, sample_weights_list
        )

    return (
        best_weighted_mmd,
        best_sample_biases,
        best_wasserstein_distances,
    )


def compute_classification_metrics_tree(
    N,
    R,
    T,
    columns,
    sample_weights,
    feature_weights,
    label,
    random_state=None,
    n_splits=10,
    splitter="feature_weighted_best",
    max_features="sqrt",
    draw_with_feature_weights=False,
    speedup=False,
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
        N[columns].values,
        N[label].values,
        R[columns].values,
        sample_weights,
        np.array(feature_weights),
        random_state=random_state,
        n_splits=n_splits,
        speedup=speedup,
        draw_with_feature_weights=draw_with_feature_weights,
        splitter=splitter,
        max_features=max_features,
    )
    y_predictions = clf.predict_proba(T[columns].values)[:, 1]
    auroc_score = roc_auc_score(T[label], y_predictions)
    auprc = average_precision_score(T[label], y_predictions)

    return auroc_score, auprc


def compute_classification_metrics_random_forest(
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
        for temperature in sample_weights_list.keys():
            for hyperparameter, sample_weights in sample_weights_list[
                temperature
            ].items():
                feature_weight = feature_weights[temperature][hyperparameter]
                not_zero_indices = np.array(sample_weights) > 0
                N_train = N.loc[not_zero_indices, :]
                train_sample_weights = np.array(sample_weights)[not_zero_indices]
                
                clf, score = train_random_forest_classifier(
                    N_train[columns].values,
                    N_train[label].values,
                    train_sample_weights,
                    feature_weight,
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
                    best_weights = sample_weights
                    best_temperature = temperature
                    best_hyperparameter = hyperparameter
    else:
        best_temperature = 0
        N_train = N.copy()
        train_sample_weights = sample_weights_list.copy()

        best_clf, _ = train_random_forest_classifier(
            N_train[columns].values,
            N_train[label].values,
            train_sample_weights,
            feature_weights,
            random_state=random_state,
            n_splits=n_splits,
            draw_with_feature_weights=draw_with_feature_weights,
            splitter=splitter,
            n_estimators=n_estimators,
            max_features=max_features,
        )
        best_hyperparameter = None
        best_weights = sample_weights_list
    y_predictions = best_clf.predict_proba(T[columns].values)[:, 1]
    fpr, tpr, _ = roc_curve(T[label], y_predictions)

    if compute_feature_importance:
        abs_feature_importance = calculate_feature_importance(
            T[columns].values,
            best_clf.best_estimator_,
        )
    else:
        abs_feature_importance = None

    auroc_score = roc_auc_score(T[label], y_predictions)
    auprc = average_precision_score(T[label], y_predictions)

    return (
        auroc_score,
        auprc,
        best_weights,
        abs_feature_importance,
        (fpr.tolist(), tpr.tolist()),
        best_temperature,
        best_hyperparameter,
        best_clf,
    )


def compute_classification_metrics_svm(
    N,
    T,
    columns,
    sample_weights_list,
    feature_weights,
    label,
    random_state=None,
    n_splits=5,
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
    if isinstance(sample_weights_list, dict):
        best_clf = None
        best_score = -1
        for temperature in sample_weights_list.keys():
            for hyperparameter, sample_weights in sample_weights_list[
                temperature
            ].items():
                feature_weight = feature_weights[temperature][hyperparameter]
                N_train = N.copy()
                N_train[columns] = N_train[columns]
                train_sample_weights = sample_weights.copy()

                clf, score = train_svc(
                    N_train[columns].values,
                    N_train[label].values,
                    train_sample_weights,
                    feature_weight,
                    random_state=random_state,
                    n_splits=n_splits,
                    draw_with_feature_weights=draw_with_feature_weights,
                )
                if score > best_score:
                    best_score = score
                    best_clf = clf
    else:
        N_train = N.copy()
        N_train[columns] = N_train[columns]
        train_sample_weights = sample_weights_list.copy()

        best_clf, _ = train_svc(
            N_train[columns].values,
            N_train[label].values,
            train_sample_weights,
            feature_weights,
            random_state=random_state,
            n_splits=n_splits,
            draw_with_feature_weights=draw_with_feature_weights,
        )
    y_predictions = best_clf.decision_function(T[columns].values)

    auroc_score = roc_auc_score(T[label], y_predictions)
    auprc = average_precision_score(T[label], y_predictions)

    return (
        auroc_score,
        auprc,
    )


def compute_classification_metrics_pseudo_labels(
    R,
    T,
    columns,
    best_clf,
    target,
    random_state,
    n_splits=10,
    n_estimators=500,
):
    calibrated_clf = CalibratedClassifierCV(best_clf, cv="prefit")
    calibrated_clf.fit(R[columns].values, R[target])
    pseudo_targets = calibrated_clf.predict(R[columns].values)
    clf, auroc = train_random_forest_classifier(
        R[columns].values,
        pseudo_targets,
        None,
        None,
        random_state=random_state,
        n_splits=n_splits,
        draw_with_feature_weights=False,
        splitter="best",
        n_estimators=n_estimators,
    )
    predictions = clf.predict_proba(T[columns].values)[:, 1]
    auroc = roc_auc_score(T[target], predictions)
    auprc = average_precision_score(T[target], predictions)

    return auroc, auprc


def compute_classification_metrics_random_forest_gbs(
    N,
    R,
    T,
    columns,
    sample_weights_list,
    feature_weights,
    label,
    random_state=None,
    n_splits=5,
    splitter="feature_weighted_best",
    n_estimators=500,
    max_depth=None,
    compute_feature_importance=True,
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
    if isinstance(sample_weights_list, dict):
        best_clf = None
        best_score = -1
        for sample_weights, feature_weight in zip(
            sample_weights_list.values(), feature_weights.values()
        ):
            clf, score = train_random_forest_classifier(
                N[columns].values,
                N[label].values,
                R[columns].values,
                sample_weights,
                feature_weight,
                random_state=random_state,
                n_splits=n_splits,
                draw_with_feature_weights=draw_with_feature_weights,
                splitter=splitter,
                n_estimators=n_estimators,
            )
            if score > best_score:
                best_score = score
                best_clf = clf
                best_weights = sample_weights
    else:
        best_clf, _ = train_random_forest_classifier(
            N[columns].values,
            N[label].values,
            R[columns].values,
            sample_weights_list,
            feature_weights,
            random_state=random_state,
            n_splits=n_splits,
            draw_with_feature_weights=draw_with_feature_weights,
            splitter=splitter,
            n_estimators=n_estimators,
        )
        best_weights = sample_weights_list
    y_predictions = best_clf.predict_proba(T[columns].values)[:, 1]
    fpr, tpr, _ = roc_curve(T[label], y_predictions)

    if compute_feature_importance:
        abs_feature_importance = calculate_feature_importance(
            T[columns].values,
            best_clf.best_estimator_,
            label,
            N[columns].values,
        )
    else:
        abs_feature_importance = None
        feature_importance = None

    auroc_score = roc_auc_score(T[label], y_predictions)
    auprc = average_precision_score(T[label], y_predictions)

    return (
        auroc_score,
        auprc,
        best_weights,
        abs_feature_importance,
        feature_importance,
        (fpr.tolist(), tpr.tolist()),
    )


def train_tree_classifier_mrs(
    X_train, y_train, speedup=True, n_splits=10, random_state=None
):
    """Train a classifier to measure the auroc

    :param X_train: Training features
    :param y_train: Training targets
    :param weights: Sample weights, defaults to None
    :param speedup: If true, use only a subset of the cost complexities, defaults to True
    :param cv: Number of cross-validation iterations, defaults to 3
    :return: Trained classifier
    """
    clf = DecisionTreeClassifier(random_state=np.random.RandomState(random_state))
    path = clf.cost_complexity_pruning_path(
        X_train,
        y_train,
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
            ccp_alphas_unique = np.unique(ccp_alphas_unique)

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
        n_jobs=-1,
        refit=True,
        scoring="roc_auc",
    )

    return grid.fit(
        X_train,
        y_train,
    )


def compute_test_metrics_mrs(
    N,
    R,
    columns,
    calculate_roc=False,
    n_test_splits=5,
    random_state=None,
    validation_method=train_tree_classifier_mrs,
    **kwargs,
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
    auroc_scores = []
    ifpr_list = []
    itpr_list = []
    kf = StratifiedKFold(
        n_splits=n_test_splits, shuffle=True, random_state=random_state
    )
    for train_indices, test_indices in kf.split(data[columns], data.label):
        train, test = data.iloc[train_indices], data.iloc[test_indices]
        clf = validation_method(
            train[columns],
            train.label,
            random_state=random_state,
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


def train_pu_classifier(
    X,
    y,
    n_estimators=200,
    feature_weight=None,
    random_state=None,
    splitter="feature_weighted_best",
    hyperparameter=0.0,
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
        n_jobs=-1,
        random_state=random_state,
        min_weight_fraction_leaf=hyperparameter,
        splitter=splitter,
        class_weight="balanced",
    )

    clf.fit(
        X,
        y,
        draw_with_feature_weights=draw_with_feature_weight,
        feature_weights=feature_weight,
    )

    return clf


def train_svm_pu_classifier(
    X,
    y,
    random_state=None,
    C=1,
):
    """Train the positive unlabeled classifier

    :param X_train: Training features
    :param y_train: Training target
    :param class_weight: Sample weights, defaults to "balanced"
    :return: Trained positive unlabeled classifier
    """
    clf = LinearSVC(dual="auto", random_state=random_state, C=C)

    clf.fit(
        X,
        y,
    )

    return clf


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
        class_weight="balanced",
        min_weight_fraction_leaf=hyperparameter,
    )

    return clf.fit(
        X,
        y,
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


def train_tree_classifier_auroc(
    X,
    y,
    R,
    sample_weights=None,
    feature_weights=None,
    speedup=False,
    n_splits=10,
    random_state=None,
    draw_with_feature_weights=False,
    splitter="feature_weighted_best",
    max_features="sqrt",
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
    best_auroc = -np.inf
    best_ccp_alpha = None
    r_sample_weights = np.ones(len(R)) / len(R)
    clf = DecisionTreeClassifier(
        random_state=np.random.RandomState(random_state),
        max_features=max_features,
        splitter=splitter,
    )
    path = clf.cost_complexity_pruning_path(
        X,
        y,
        sample_weight=sample_weights,
        feature_weights=feature_weights,
        draw_with_feature_weights=draw_with_feature_weights,
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

    for ccp_alpha in ccp_alphas_unique:
        auroc_list = []
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        for train_indices_n, val_indices_n in skf.split(X, y):
            X_train, y_train = X[train_indices_n], y[train_indices_n]
            sample_weights_train = sample_weights[train_indices_n]
            sample_weights_val = sample_weights[val_indices_n]
            X_val, y_val = X[val_indices_n], y[val_indices_n]
            clf = DecisionTreeClassifier(
                random_state=np.random.RandomState(random_state),
                max_features=max_features,
                splitter=splitter,
                ccp_alpha=ccp_alpha,
            )
            clf.fit(
                X_train,
                y_train,
                sample_weight=sample_weights_train,
                feature_weights=feature_weights,
                draw_with_feature_weights=draw_with_feature_weights,
            )
            self_labeled_targets = clf.predict(R)
            clf.fit(
                R,
                self_labeled_targets,
                sample_weight=r_sample_weights,
                feature_weights=feature_weights,
            )
            reverse_probs = clf.predict_proba(X_val)
            if reverse_probs.shape[1] == 2:
                reverse_probs = reverse_probs[:, 1]

            auroc = roc_auc_score(
                y_val, reverse_probs, sample_weight=sample_weights_val
            )
            auroc_list.append(auroc)

        mean_auroc = np.mean(auroc_list)
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_ccp_alpha = ccp_alpha

    clf = DecisionTreeClassifier(
        random_state=np.random.RandomState(random_state),
        max_features=max_features,
        splitter=splitter,
        ccp_alpha=best_ccp_alpha,
    )
    clf.fit(
        X,
        y,
        sample_weight=sample_weights,
        feature_weights=feature_weights,
    )

    return clf


def train_random_forest_classifier(
    X,
    y,
    sample_weights,
    feature_weights=None,
    n_splits=5,
    draw_with_feature_weights=False,
    random_state=None,
    splitter="feature_weighted_best",
    n_estimators=500,
    max_features="sqrt",
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

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if draw_with_feature_weights:
        feature_weights = np.array(feature_weights)
    param_grid = {
        "min_weight_fraction_leaf": [
            0.0,
            0.01,
            0.025,
            0.05,
            0.1,
        ],
        "class_weight": ["balanced", None],
    }
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
        scoring="roc_auc",
        refit=True,
    )

    grid_cv.fit(
        X,
        y,
        sample_weight=sample_weights,
        feature_weights=feature_weights,
        draw_with_feature_weights=draw_with_feature_weights,
    )

    return grid_cv, grid_cv.best_score_


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


def compute_classification_metrics_feature_weights(
    N,
    R,
    columns,
    sample_weights,
    label,
    random_state=None,
    n_splits=10,
    class_weight=None,
    splitter="feature_weighted_best",
    n_estimators=500,
    max_depth=None,
    feature_weights_list=None,
    drop_ids_list=None,
):
    """Computes classification metrics for downstream tasks

    :param N: Non representative data set
    :param R: Representative data set
    :param columns: Columns used in the training
    :param weights: Computed sample weights
    :param label: Name of the target variable
    :return: Downstream classification metrics
    """
    n_estimators_per_budget = n_estimators // len(feature_weights_list)
    clf_list = []
    for feature_weights, drop_ids in zip(feature_weights_list, drop_ids_list):
        draw_with_feature_weights = False if feature_weights is None else True
        iteration_sample_weights = sample_weights.copy()
        iteration_sample_weights[drop_ids] = 0
        clf = train_random_forest_classifier(
            N[columns],
            N[label],
            iteration_sample_weights,
            feature_weights,
            random_state=random_state,
            n_splits=n_splits,
            draw_with_feature_weights=draw_with_feature_weights,
            class_weight=class_weight,
            splitter=splitter,
            n_estimators=n_estimators_per_budget,
            max_depth=max_depth,
        )
        clf_list.append(clf)

    y_predictions = []
    for clf in clf_list:
        y_predictions.append(clf.predict_proba(R[columns])[:, 1])
    y_predictions = np.mean(y_predictions, axis=0)
    auroc_score = roc_auc_score(R[label], y_predictions)
    auprc = average_precision_score(R[label], y_predictions)

    return auroc_score, auprc


def compute_feature_weights_with_temperature(temperature, feature_importance):
    """_summary_

    :param temperature: _description_
    :param feature_importance: _description_
    :return: _description_
    """
    if temperature == 0.0:
        return np.ones(len(feature_importance)) / len(feature_importance)
    feature_weights = np.exp(-np.array(feature_importance) / temperature)
    return feature_weights / np.sum(feature_weights)


def calculate_feature_importance(test_N, clf, background=None):

    explainer = shap.TreeExplainer(clf, data=background)
    explainer = explainer(test_N, check_additivity=False)
    shap_values = explainer.values[:, :, 1]
    abs_feature_importance = np.mean(np.abs(shap_values), axis=0)

    return abs_feature_importance


from fairlearn.reductions import DemographicParity, ExponentiatedGradient


def compute_classification_metrics_random_forest_fairness(
    N,
    columns,
    sensitive_attribute,
    sample_weights_list,
    feature_weights,
    label,
    random_state=None,
    n_splits=5,
    splitter="feature_weighted_best",
    n_estimators=500,
    draw_with_feature_weights=False,
    max_features="sqrt",
    mitigate=False,
):
    """Computes classification metrics for downstream tasks

    :param N: Non representative data set
    :param R: Representative data set
    :param columns: Columns used in the training
    :param weights: Computed sample weights
    :param label: Name of the target variable
    :return: Downstream classification metrics
    """
    clf_list = []
    if isinstance(sample_weights_list, dict):
        for sample_weights, feature_weight in zip(
            sample_weights_list.values(), feature_weights.values()
        ):
            N_train = N.copy()
            train_sample_weights = sample_weights.copy()

            clf = train_random_forest_classifier_fairness(
                N_train[columns].values,
                N_train[label].values,
                N_train[sensitive_attribute],
                train_sample_weights,
                feature_weight,
                random_state=random_state,
                n_splits=n_splits,
                draw_with_feature_weights=draw_with_feature_weights,
                splitter=splitter,
                n_estimators=n_estimators,
                max_features=max_features,
            )
            clf_list.append(clf)
    else:
        N_train = N.copy()
        clf = train_random_forest_classifier_fairness(
            N_train[columns].values,
            N_train[label].values,
            N_train[sensitive_attribute],
            sample_weights_list,
            feature_weights,
            random_state=random_state,
            n_splits=n_splits,
            draw_with_feature_weights=draw_with_feature_weights,
            splitter=splitter,
            n_estimators=n_estimators,
            max_features=max_features,
            mitigate=mitigate,
        )
        clf_list = clf

    return clf_list


def train_random_forest_classifier_fairness(
    X,
    y,
    sensitive_attribute,
    sample_weights,
    feature_weights=None,
    n_splits=5,
    draw_with_feature_weights=False,
    random_state=None,
    splitter="feature_weighted_best",
    n_estimators=500,
    max_features="sqrt",
    mitigate=False,
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
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if draw_with_feature_weights:
        feature_weights = np.array(feature_weights)
    param_grid = {
        "min_weight_fraction_leaf": [
            0.0,
            0.01,
            0.1,
        ],
        "class_weight": ["balanced", None],
    }
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
        scoring="roc_auc",
        refit=True,
    )
    if mitigate:
        constraint = DemographicParity()
        mitigator = ExponentiatedGradient(grid_cv, constraint)
        grid_cv = mitigator.fit(X, y, sensitive_features=sensitive_attribute)
    else:
        grid_cv.fit(
            X,
            y,
            sample_weight=sample_weights,
            feature_weights=feature_weights,
            draw_with_feature_weights=draw_with_feature_weights,
        )

    return grid_cv
