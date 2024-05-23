import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


def compute_classification_metrics_gradient_descent(
    N,
    R,
    T,
    columns,
    sample_weights,
    feature_weights,
    label,
    random_state=None,
    regularization_name="scad",
    n_splits=10,
):
    clf = train_gradient_descent_classifier(
        N[columns].values,
        N[label].values,
        R[columns].values,
        sample_weights,
        feature_weights,
        random_state,
        regularization_name=regularization_name,
        n_splits=n_splits,
    )
    print(clf.lambda_value)
    y_predictions = clf.predict_proba(T[columns])[:, 1]
    auroc_score = roc_auc_score(T[label], y_predictions)
    auprc = average_precision_score(T[label], y_predictions)

    return auroc_score, auprc


def train_gradient_descent_classifier(
    X,
    y,
    R,
    sample_weights,
    feature_weights=None,
    random_state=None,
    regularization_name=None,
    n_splits=5,
):
    lambda_values = [0.1, 0.01, 0.001, 0.0001]
    # lambda_values = [0.1]
    best_auroc = -np.inf
    best_lambda = None
    r_sample_weights = np.ones(len(R)) / len(R)
    r_feature_weights = np.ones(len(feature_weights)) / len(feature_weights)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for lambda_value in lambda_values:
        auroc_list = []
        for train_indices_n, val_indices_n in skf.split(X, y):
            X_train, y_train = X[train_indices_n], y[train_indices_n]
            sample_weights_train = sample_weights[train_indices_n]
            sample_weights_val = sample_weights[val_indices_n]
            X_val, y_val = X[val_indices_n], y[val_indices_n]

            clf = GradientDescentModel(
                regularization_name=regularization_name,
                lambda_value=lambda_value,
            )
            clf.fit(
                X_train,
                y_train,
                sample_weights=sample_weights_train,
                feature_weights=feature_weights,
            )
            self_labeled_targets = clf.predict(R)
            clf.fit(
                R,
                self_labeled_targets,
                sample_weights=r_sample_weights,
                feature_weights=r_feature_weights,
            )
            reverse_probs = clf.predict_proba(X_val)[:, 1]

            auroc = roc_auc_score(
                y_val, reverse_probs, sample_weight=sample_weights_val
            )
            auroc_list.append(auroc)

        mean_auroc = np.mean(auroc_list)
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_lambda = lambda_value

    clf = GradientDescentModel(
        regularization_name=regularization_name,
        lambda_value=best_lambda,
    )
    clf.fit(
        X,
        y,
        sample_weights=sample_weights,
        feature_weights=feature_weights,
    )

    return clf


class GradientDescentModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        epsilon=1e-4,
        learning_rate=0.01,
        max_patience=100,
        regularization_name=None,
        lambda_value=0.0,
    ) -> None:
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_patience = max_patience
        self.regularization_name = regularization_name
        self.regularization_method = self.get_regularization_function(
            regularization_name
        )
        self.lambda_value = lambda_value
        self.weights = None

    def fit(
        self,
        X,
        y,
        sample_weights,
        feature_weights,
    ) -> None:
        self.weights = np.zeros(len(feature_weights) + 1)
        current_patience = 0
        lowest_gradient = np.inf
        self.classes_ = unique_labels(y)
        X_with_intercept = np.append(np.ones(len(X))[:, np.newaxis], X, axis=1)
        feature_weights = np.append(np.max(feature_weights), feature_weights)
        feature_weights = feature_weights / np.sum(feature_weights)
        while True:
            gradient_norm = self.gradient_descent_step(
                X_with_intercept,
                y,
                sample_weights,
                feature_weights,
                self.regularization_method,
            )
            if gradient_norm < self.epsilon:
                break
            if gradient_norm < lowest_gradient:
                lowest_gradient = gradient_norm
                current_patience = 0
            else:
                current_patience += 1
            if current_patience >= self.max_patience:
                break

    def gradient_descent_step(
        self,
        X,
        y,
        sample_weights,
        feature_weights,
        regularization_method,
    ):
        # feature_weights = feature_weights / np.sum(feature_weights)
        predicted_probabilities = self.predict_proba(X)[:, 1]
        target_difference = y - predicted_probabilities
        gradients = np.average(
            X * target_difference[:, np.newaxis],
            weights=sample_weights,
            axis=0,
        )
        if regularization_method is not None:
            regularization_gradients = regularization_method(
                self.weights, self.lambda_value
            )
            regularization_gradients = (1 - feature_weights) * regularization_gradients
        else:
            regularization_gradients = 0
        weighted_gradients = self.learning_rate * (
            -gradients + regularization_gradients
        )
        self.weights -= weighted_gradients

        return np.linalg.norm(weighted_gradients)

    def smoothly_clipped_absolute_deviation(self, weights, lambda_value=0.4, a=3.7):
        gradients = np.zeros(len(weights))
        first_indices = np.abs(weights) <= lambda_value
        gradients[first_indices] = lambda_value * np.sign(weights[first_indices])

        second_indices = (lambda_value < np.abs(weights)) & (
            np.abs(weights) <= a * lambda_value
        )
        gradients[second_indices] = (
            (a * lambda_value - np.abs(weights[second_indices]))
            * np.sign(weights[second_indices])
        ) / (a - 1)

        return gradients

    def get_regularization_function(self, regularization_method_name):
        if regularization_method_name == "l1":
            return self.l1
        elif regularization_method_name == "l2":
            return self.l2
        elif regularization_method_name == "scad":
            return self.smoothly_clipped_absolute_deviation
        elif regularization_method_name == "mcp":
            return self.minimax_concave_penalty
        else:
            return None

    def minimax_concave_penalty(self, weights, lambda_value, a=3):
        gradients = np.zeros(len(weights))
        indices = np.abs(weights) <= lambda_value * a
        gradients[indices] = (np.sign(weights[indices]) * lambda_value) - (
            weights[indices] / a
        )

        return gradients

    def l1(self, weights, lambda_value):
        return np.sign(weights) * lambda_value

    def l2(self, weights, lambda_value):
        return weights * lambda_value

    def predict_proba(self, X):
        if X.shape[1] < len(self.weights):
            X_with_intercept = np.append(np.ones(len(X))[:, np.newaxis], X, axis=1)
        else:
            X_with_intercept = X
        probabilities = self.logistic_function(
            np.sum(X_with_intercept * self.weights, axis=1)
        )
        return np.stack([1 - probabilities, probabilities], axis=1)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X_train, y_test):
        y_train = self.predict_proba(X_train)[:, 1]
        return roc_auc_score(y_test, y_train)

    def logistic_function(self, X):
        np.seterr(over="ignore")
        return 1 / (1 + np.exp(-X))
