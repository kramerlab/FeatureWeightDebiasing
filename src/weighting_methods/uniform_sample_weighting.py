import numpy as np


def uniform_sample_weighting(N, columns, *args, **attributes):
    """Uniform weighting

    :param N: Non-representative data set
    :param columns: List of feature column names
    :return: Sample weights
    """
    weights = np.ones(len(N)) / len(N)
    return (weights / np.sum(weights)).tolist(), np.ones(len(columns)).tolist()
