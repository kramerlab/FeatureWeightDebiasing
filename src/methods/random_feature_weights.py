import numpy as np
seed = 5
generator = np.random.default_rng(seed)

def random_weighting(N, R, columns, *args, **attributes):
    """Random Feature weighting

    :param N: Non-representative data set
    :return: Feature weights
    """
    weights = generator.uniform(0, 1, len(columns))
    weights = (weights / sum(weights)) * len(columns)
    return weights
