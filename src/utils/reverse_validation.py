from typing import Any

from numpy import ones
from sklearn.metrics import roc_auc_score
from sklearn.base import ClassifierMixin, BaseEstimator


class ReverseScorer(BaseEstimator, ClassifierMixin):
    def __init__(self, R) -> None:
        self.R = R
        self.feature_weights = ones(R.shape[1]) / R.shape[1]
        self.r_sample_weights = ones(len(R)) / len(R)

    def __call__(self, estimator, X, y) -> Any:
        self_labeled_targets = estimator.predict(self.R)
        estimator.fit(
            self.R,
            self_labeled_targets,
            sample_weight=self.r_sample_weights,
            feature_weights=self.feature_weights,
        )

        reverse_probs = estimator.predict_proba(X)
        if reverse_probs.shape[1] == 2:
            reverse_probs = reverse_probs[:, 1]
        auroc = roc_auc_score(y, reverse_probs)

        return auroc
