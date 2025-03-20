import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def propensity_score_adjustment(
    N, R, columns, hyperparameter_list, *args, **attributes
):
    """Propensity score adjustment

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Training columns
    :return: Sample weights
    """
    sample_weights_dict = {}
    feature_weights_dict = {}
    train = pd.concat([N, R])
    x = train[columns].values
    y = train.label
    for hyperparameter in hyperparameter_list:
        clf = train_logistic_regression(x, y, hyperparameter)
        predictions = clf.predict_proba(N[columns].values)[:, 1]
        weights = (1 - predictions) / predictions
        sample_weights_dict[hyperparameter] = {0: (weights / weights.sum()).tolist()}
        feature_weights_dict[hyperparameter] = (
            np.ones(len(columns)) / len(columns)
        ).tolist()

    return sample_weights_dict, feature_weights_dict


def train_logistic_regression(X_train, y_train, hyperparameter):
    """Trains a logistic regression to distinguish N and R

    :param X_train: Training data
    :param y_train: Training target
    :return: Trained logistic regression
    """
    logistic_regression = LogisticRegression(max_iter=10000, C=hyperparameter)
    logistic_regression = logistic_regression.fit(X_train, y_train)
    return logistic_regression
