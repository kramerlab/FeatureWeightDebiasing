import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np


def logistic_regression(N, R, columns, *args, **attributes):
    """Propensity score adjustment

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Training columns
    :return: Sample weights
    """
    train = pd.concat([N, R])
    x = train[columns].values
    y = train.label
    clf = train_logistic_regression(x, y)
    weights = np.abs(clf.coef_)[0]
    weights = (1 - weights) 
    return weights


def train_logistic_regression(X_train, y_train):
    """Trains a logistic regression to distinguish between N and R

    :param X_train: Training data
    :param y_train: Training target
    :return: Trained logistic regression
    """
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression = logistic_regression.fit(X_train, y_train)
    return logistic_regression
