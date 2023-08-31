import numpy as np


class MMDScoring:
    """Update weights and compute the MMD"""

    def __init__(self, loss_function, weights, exponential) -> None:
        """Init

        :param loss_function: The mmd loss function
        :param weights: The current weights
        """
        self.loss_function = loss_function
        self.weights = weights
        self.exponential = exponential

    def __call__(self, ground_truth, predictions):
        """Update weights and returns the mmd calculated with updated weights

        :param ground_truth: True values
        :param predictions: Predicted probabilities
        :return: Weighted MMD
        """
        N_indices = np.where(ground_truth == 1)[0]
        tmp_weights = update_weights(
            self.weights[N_indices], predictions[N_indices], self.exponential
        )
        loss = self.loss_function(tmp_weights)
        return loss


class FullSample:
    """Returns indices for all samples"""

    def __init__(self, n_splits) -> None:
        """_summary_

        :param n_splits: #defines how many splits should be returned
        """
        self.n_splits = n_splits

    def get_n_splits(self, x, y, group=None):
        """Return the number of splits

        :param x: Not used
        :param y: Not used
        :param group: Not use, defaults to None
        :return: Number of splits
        """
        return self.n_splits

    def split(self, X, y, group=None):
        """Returns all input indices

        :param X: All samples
        :param y: Not used
        :param group: Not used, defaults to None
        :yield: All indices
        """
        for _ in range(self.n_splits):
            yield list(range(len(X))), list(range(len(X)))


def update_weights(weights, predictions, exponential):
    """Updates sample weights based on the prediction probabilities

    :param weights: Sample weights
    :param predictions: Prediction probabilities that are used to compute the
        new weights
    :param exponential: If true, use exponential weight update
    :return: Updated sample weights
    """
    y_codes = np.array([-1.0, 1.0])
    epsilon = np.finfo(weights.dtype).eps
    predictions = np.clip(predictions, a_min=epsilon, a_max=None)
    if exponential:
        p_difference = np.abs(0.5 - predictions)
        y_coding = y_codes.take(predictions < 0.5)
        weight_modificator = y_coding * np.power(p_difference, 0.5) + 1
        weights *= weight_modificator
    else:
        weight_modificator = 1.5 - predictions
        weights *= weight_modificator
    weights = weights / weights.sum()
    return weights
