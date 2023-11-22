import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {"min_samples_split": [5], "min_samples_leaf": [5], "n_estimators": [5]}


def random_forest_weighting(N, R, columns, *args, **attributes):
    """Propensity score adjustment

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Training columns
    :return: Sample weights
    """
    train = pd.concat([N, R])
    X = train[columns].values
    y = train.label
    clf = train_random_forest(X, y)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X)
    feature_importance = np.abs(shap_values.values[:, :, 1]).mean(0)
    feature_weights = 1 - (feature_importance * 5)
    return feature_weights


def train_random_forest(X_train, y_train):
    """Trains a logistic regression to distinguish between N and R

    :param X_train: Training data
    :param y_train: Training target
    :return: Trained logistic regression
    """
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=5),
        param_grid=param_grid,
        refit=True,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
