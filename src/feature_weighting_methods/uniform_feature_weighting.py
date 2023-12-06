import numpy as np


def uniform_feature_weighting(columns, *args, **attributes):
    """Uniform weighting

    :param N: Non-representative data set
    :return: Sample weights
    """
    weights = np.ones(len(columns))
    return weights
