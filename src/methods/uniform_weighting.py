import numpy as np


def uniform_weighting(N, *args, **attributes):
    """Uniform weighting

    :param N: Non-representative data set
    :return: Sample weights
    """
    weights = np.ones(len(N)) / len(N)
    return weights / np.sum(weights)
