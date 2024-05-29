import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted
from utils.reverse_validation import ReverseScorer
from sklearn import set_config


def compute_classification_metrics_gradient_descent(
    N,
    R,
    T,
    columns,
    sample_weights,
    feature_weights,
    label,
    random_state=None,
    n_splits=5,
):
    set_config(enable_metadata_routing=True)
    if isinstance(sample_weights, dict):
        best_clf = None
        best_score = -1
        for sample_weight, feature_weight in zip(sample_weights.values(), feature_weights.values()):
            clf, score = train_gradient_descent_classifier(
                N[columns].values,
                N[label].values,
                R[columns].values,
                sample_weight,
                feature_weight,
                random_state,
                n_splits=n_splits,
            )
            if score > best_score:
                best_score = score
                best_clf = clf
    else:
        best_clf, _ = train_gradient_descent_classifier(
            N[columns].values,
            N[label].values,
            R[columns].values,
            sample_weights,
            feature_weights,
            random_state,
            n_splits=n_splits,
        )

    y_predictions = best_clf.predict_proba(T[columns])[:, 1]
    auroc_score = roc_auc_score(T[label], y_predictions)
    auprc = average_precision_score(T[label], y_predictions)

    set_config(enable_metadata_routing=False)
    return auroc_score, auprc


def train_gradient_descent_classifier(
    X,
    y,
    R,
    sample_weights,
    feature_weights=None,
    random_state=None,
    n_splits=5,
):

    param_grid = {
        "lambda_value": [10, 1, 0.1, 0.01, 0.001],
        "learning_rate": [0.01],
        "regularization_name": ["scad", "l1", "l2"],
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scorer = ReverseScorer(R, feature_weights).set_score_request(sample_weight=True)
    clf = GradientDescentModel().set_fit_request(
        sample_weight=True,
        feature_weights=True,
    )
    grid_cv = GridSearchCV(
        clf, param_grid, cv=skf, n_jobs=-1, scoring=scorer, refit=True
    )
    grid_cv.fit(X, y, sample_weight=sample_weights, feature_weights=feature_weights)
    return grid_cv, grid_cv.best_score_


class GradientDescentModel(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        learning_rate=0.1,
        regularization_name="scad",
        lambda_value=0.01,
        epsilon=1e-3,
        max_patience=100,
    ) -> None:
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_patience = max_patience
        self.regularization_name = regularization_name
        self.lambda_value = lambda_value

    def fit(
        self,
        X,
        y,
        sample_weight,
        feature_weights,
    ) -> None:
        self.regularization_method = self.get_regularization_function(
            self.regularization_name
        )
        current_patience = 0
        lowest_gradient = np.inf
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        X_with_intercept = np.append(np.ones(len(X))[:, np.newaxis], X, axis=1)
        feature_weights = np.append(np.min(feature_weights), feature_weights)
        self.coefficients_ = np.zeros(len(feature_weights))

        while True:
            gradient_norm = self.gradient_descent_step(
                X_with_intercept,
                y,
                sample_weight,
                feature_weights,
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
        return self

    def gradient_descent_step(
        self,
        X,
        y,
        sample_weights,
        feature_weights,
    ):
        predicted_probabilities = self.predict_proba(X)[:, 1]
        target_difference = predicted_probabilities - y
        gradients = np.average(
            X * target_difference[:, np.newaxis],
            weights=sample_weights,
            axis=0,
        )
        if self.regularization_method is not None:
            regularization_gradients = self.regularization_method(
                self.coefficients_, self.lambda_value * feature_weights
            )
            weighted_regularization_gradients = regularization_gradients
        else:
            weighted_regularization_gradients = 0
        regularized_gradients = gradients + weighted_regularization_gradients
        weighted_gradients = self.learning_rate * regularized_gradients

        self.coefficients_ -= weighted_gradients

        return np.linalg.norm(weighted_gradients)

    def smoothly_clipped_absolute_deviation(
        self, coefficients, lambda_value=0.4, a=3.7
    ):
        gradients = np.zeros(len(coefficients))
        first_indices = np.abs(coefficients) <= lambda_value
        if first_indices.any():
            gradients[first_indices] = lambda_value[first_indices] * np.sign(
                coefficients[first_indices]
            )

        second_indices = (lambda_value < np.abs(coefficients)) & (
            np.abs(coefficients) <= a * lambda_value
        )
        if second_indices.any():
            gradients[second_indices] = (
                (
                    a * lambda_value[second_indices]
                    - np.abs(coefficients[second_indices])
                )
                * np.sign(coefficients[second_indices])
            ) / ((a - 1) * lambda_value[second_indices])

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
            return self.l1

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
        if X.shape[1] < len(self.coefficients_):
            X_with_intercept = np.append(np.ones(len(X))[:, np.newaxis], X, axis=1)
        else:
            X_with_intercept = X
        probabilities = self.logistic_function(
            np.sum(X_with_intercept * self.coefficients_, axis=1)
        )
        return np.stack([1 - probabilities, probabilities], axis=1)

    def predict(self, X):
        check_is_fitted(self)
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X_train, y_test):
        y_train = self.predict_proba(X_train)[:, 1]
        return roc_auc_score(y_test, y_train)

    def logistic_function(self, X):
        np.seterr(over="ignore")
        return 1 / (1 + np.exp(-X))

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "max_patience": self.max_patience,
            "regularization_name": self.regularization_name,
            "lambda_value": self.lambda_value,
        }
