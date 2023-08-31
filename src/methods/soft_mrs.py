import sys

import pandas as pd
import numpy as np

from utils.metrics import calculate_rbf_gamma
from utils.soft_mrs_cross_validation import FullSample, MMDScoring, update_weights
from utils.weighted_mmd_loss import WeightedMMDLoss
from sklearn.model_selection import GridSearchCV
from sklearn.metrics._scorer import make_scorer
from sklearn.tree import DecisionTreeClassifier


# Used to draw radom states
max_int = 2**32 - 1

# Parameter for MMD cross-validation
param_grid = {
    "min_weight_fraction_leaf": [
        0.001,
        0.0025,
        0.005,
        0.01,
        0.03,
        0.02,
        0.04,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
}


def soft_mrs_weighting(
    N, R, columns, random_generator, exponential=False, patience=None, *args, **kwargs
):
    """Soft MRS method

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Training columns
    :param random_generator: Random generator to create random_states to make results reproducible
    :return: Sample weights
    """
    weights_N = np.ones(len(N)) / len(N)
    weights_R = np.ones(len(R)) / len(R)
    concat_data = pd.concat([N, R])
    best_mmd = np.inf
    current_patience = 0

    gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))
    loss_function = WeightedMMDLoss(gamma, N[columns], R[columns])

    # Optimize until MMD stagnates
    while True:
        predictions, mmd = train_weighted_random_forest(
            concat_data[columns],
            concat_data["label"],
            weights=np.concatenate([weights_N, weights_R]),
            loss_function=loss_function,
            random_state=random_generator.randint(max_int),
            exponential=exponential,
        )

        if mmd < best_mmd:
            best_mmd = mmd
            best_weights = weights_N.copy()
            current_patience = 0

        else:
            if current_patience == patience:
                break
            else:
                current_patience += 1

        predictions_N = predictions[: len(N), 1]
        weights_N = update_weights(weights_N, predictions_N, exponential=exponential)

    return best_weights


def train_weighted_random_forest(
    x, label, weights, loss_function, random_state, exponential
):
    """Trains a random forest and returns the predicted probabilties

    :param x: Training data
    :param label: Target label
    :param weights: Current weights
    :param random_state: Random state to make results reproducible
    :return: Predicted probabilities
    """
    scorer = make_scorer(
        MMDScoring(
            loss_function,
            weights,
            exponential=exponential,
        ),
        greater_is_better=False,
        needs_proba=True,
    )
    tree = DecisionTreeClassifier(
        max_features="sqrt", splitter="random", random_state=np.random.RandomState(random_state)
    )

    grid_cv = GridSearchCV(
        tree,
        param_grid,
        cv=FullSample(1),
        n_jobs=-1,
        scoring=scorer,
    )
    grid_cv = grid_cv.fit(x, label, sample_weight=weights)
    best_predictions = grid_cv.predict_proba(x)
    best_mmd = -grid_cv.best_score_
    return best_predictions, best_mmd
