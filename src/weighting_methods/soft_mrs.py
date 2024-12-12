import pandas as pd
import numpy as np

from utils.metrics import calculate_rbf_gamma, compute_relative_bias
from utils.soft_mrs_cross_validation import FullSample, MMDScoring, update_weights
from utils.weighted_mmd_loss import WeightedMMDLoss
from sklearn.model_selection import GridSearchCV
from sklearn.metrics._scorer import make_scorer, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import wasserstein_distance

# Used to draw radom states
max_int = 2**32 - 1

# Parameter for MMD cross-validation
min_weight_fraction_leaf = [
    0.0,
    0.01,
    0.02,
    0.04,
    0.05,
    0.075,
    0.09,
    0.1,
    0.15,
    0.25,
    0.4,
    0.5,
]


def soft_mrs_weighting(
    N,
    R,
    columns,
    random_generator=None,
    exponential=False,
    patience=50,
    method_name="",
    return_metrics=False,
    compute_bias=False,
    target=None,
    n_iterations=None,
    early_stopping=True,
    wasserstein_target=None,
    *args,
    **kwargs
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
    mmd_list = []
    sample_weights_list = []
    relative_bias_list = []
    auroc_list = []
    wasserstein_list = []
    iteration = 0

    gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))
    loss_function = WeightedMMDLoss(gamma, N[columns], R[columns])
    exponential = (
        True
        if (
            method_name in ("soft-mrs-exponential", "exponential")
            or exponential is True
        )
        else False
    )
    wasserstein_target = target if wasserstein_target is None else wasserstein_target

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

        if return_metrics:
            mmd_list.append(mmd)
            sample_weights_list.append(weights_N.tolist())

            auroc = roc_auc_score(concat_data["label"], predictions[:, 1])
            auroc_list.append(auroc)

            wasserstein_distance_value = wasserstein_distance(
                N[wasserstein_target], R[wasserstein_target], weights_N
            )

            wasserstein_list.append(wasserstein_distance_value)
            if compute_bias and target is not None:
                relative_bias = compute_relative_bias(N[wasserstein_target], R[wasserstein_target], weights_N)
                relative_bias_list.append(relative_bias.tolist())

        if mmd < best_mmd:
            best_mmd = mmd
            best_weights = weights_N.copy()
            current_patience = 0
        else:
            if current_patience == patience and early_stopping:
                break
            else:
                current_patience += 1

        predictions_N = predictions[: len(N), 1]
        weights_N = update_weights(
            weights_N, predictions_N, exponential=exponential, k=0.2
        )
        iteration += 1
        if iteration == n_iterations:
            break

    if return_metrics:
        return mmd_list, relative_bias_list, sample_weights_list, auroc_list, wasserstein_list
    else:
        return best_weights.tolist(), (np.ones(len(columns)) / len(columns)).tolist()


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
        response_method="predict_proba",
    )
    tree = DecisionTreeClassifier(
        max_features="sqrt",
        splitter="random",
        random_state=np.random.RandomState(random_state),
    )

    param_grid = {"min_weight_fraction_leaf": min_weight_fraction_leaf}
    grid_cv = GridSearchCV(
        tree,
        param_grid,
        cv=FullSample(1),
        n_jobs=5,
        scoring=scorer,
    )
    grid_cv = grid_cv.fit(x, label, sample_weight=weights)
    best_predictions = grid_cv.predict_proba(x)
    best_mmd = -grid_cv.best_score_
    return best_predictions, best_mmd
