import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np

def mutual_information(N, R, columns, *args, **attributes):
    """Propensity score adjustment

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Training columns
    :return: Sample weights
    """
    train = pd.concat([N, R])
    X = train[columns].values
    y = train.label
    mi = mutual_info_classif(X, y)
    feature_weights = 1  - (mi * 50)
    feature_weights[feature_weights < 0] = 0
    feature_weights = (feature_weights / sum(feature_weights)) * len(columns)
    return feature_weights

