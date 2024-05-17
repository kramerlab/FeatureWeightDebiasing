import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator


def compute_classification_metrics_gradient_ascent(
    N,
    R,
    columns,
    sample_weights,
    feature_weights,
    label,
    random_state=None,
):
    clf = train_gradient_ascent_classifier(
        N[columns],
        N[label],
        sample_weights,
        feature_weights,
        random_state=random_state,
    )
    y_predictions = clf.predict_proba(R[columns])
    auroc_score = roc_auc_score(R[label], y_predictions)
    auprc = average_precision_score(R[label], y_predictions)

    return auroc_score, auprc


def train_gradient_ascent_classifier(
    X_train,
    y_train,
    sample_weights,
    feature_weights=None,
    random_state=None,
):
    
    clf = GradientAscentModel()
    clf.fit(
        X_train, y_train, sample_weights=sample_weights, feature_weights=feature_weights
    )
    return clf


class GradientAscentModel(BaseEstimator):
    def __init__(self, epsilon=0.001, learning_rate=0.1) -> None:
        self.weights = None
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def fit(self, X, y, sample_weights, feature_weights) -> None:
        self.weights = np.zeros(len(feature_weights))
        while True:
            updated_model_weights = self.gradient_ascent_step(
                X.values, y.values, sample_weights, feature_weights
            )

            difference_sum = np.sum(np.abs(self.weights - updated_model_weights))
            self.weights = updated_model_weights
            if difference_sum < self.epsilon:
                break

    def predict_proba(self, X):
        return self.logistic_function(np.sum(X * self.weights, axis=1))

    def score(self, X_train, y_test):
        y_train = self.predict_proba(X_train)
        return roc_auc_score(y_test, y_train)

    def gradient_ascent_step(self, X, y, sample_weights, feature_weights):
        # feature_weights = feature_weights / np.sum(feature_weights)
        predicted_probabilities = self.predict_proba(X)
        target_difference = y - predicted_probabilities
        upgrade_step = np.average(
            X * target_difference[:, np.newaxis], weights=sample_weights, axis=0
        )
        upgraded_weights = self.weights + (
            (self.learning_rate + feature_weights) * upgrade_step
        )

        return upgraded_weights

    def logistic_function(self, X):
        return 1 / (1 + np.exp(-X))
