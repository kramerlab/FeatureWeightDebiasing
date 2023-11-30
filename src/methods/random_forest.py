import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


param_grid = {"min_samples_split": [3, 5, 10], "min_samples_leaf": [3, 5, 10], "n_estimators": [5, 10, 25]}


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
    feature_weights = 1 - (feature_importance * 10)
    feature_weights[feature_weights < 0] = 0
    feature_weights = (feature_weights / sum(feature_weights)) * len(columns)
    return feature_weights


def train_random_forest(X_train, y_train):
    """Trains a logistic regression to distinguish between N and R

    :param X_train: Training data
    :param y_train: Training target
    :return: Trained logistic regression
    """
    grid_search = DecisionTreeClassifier(random_state=5)
    grid_search.fit(X_train, y_train)
    return grid_search
