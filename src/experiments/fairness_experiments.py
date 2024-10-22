import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    average_precision_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

from utils.data_loader import load_weights, save_weights
from utils.metrics import compute_classification_metrics_random_forest_fairness
from utils.statistics import create_result_path
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_negative_rate,
    true_positive_rate,
    count,
    plot_model_comparison,
)
import matplotlib.pyplot as plt

metrics = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Selection Rate": selection_rate,
    "False Positive Rate": true_positive_rate,
    "False Negative Rate": false_negative_rate,
    "Count": count,
}

seed = 5
sampling_random_generator = np.random.RandomState(seed)


def fairness_tasks_experiment(
    df,
    columns,
    sample_weighting_method,
    target: str,
    n_cv_repeats: int,
    n_cv_splits: int,
    data_set_name: str = "",
    random_generator=None,
    load_previous_results=True,
    drop=1,
    method_name=None,
    **args,
):
    """The function uses the weighting method to compute the sample weights and
    computes the metrics, visualizes the results and saves the result in a file.

    :param df: pandas.DataFrame with the data
    :param columns: Name of training columns
    :param weighting_method: The weighting function
    :param target: Target name
    :param method: Method name, defaults to ""
    :param number_of_repetitions: Number of repetetions of the experiment,
        defaults to 100
    :param bias_type: Name of the bias that will be induced, defaults to None
    :param data_set_name: Data set name, defaults to ""
    """
    sensitive_attribute, target = target
    prediction_proba_dict = {}
    prediction_dict = {}
    target_list = []
    sensitive_attribute_list = []

    result_path = create_result_path(
        method_name,
        "",
        data_set_name,
        experiment_name="fairness_task",
        bias_fraction="",
    )
    sample_weights_save_path = result_path / "sample_weights"
    feature_weights_save_path = result_path / "feature_weights"

    result_path.mkdir(exist_ok=True)
    sample_weights_save_path.mkdir(exist_ok=True)
    feature_weights_save_path.mkdir(exist_ok=True)

    sample_weight_list = load_weights(sample_weights_save_path)
    feature_weight_list = load_weights(feature_weights_save_path)

    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    sample_df = df.copy()

    if method_name == "fw-mrs-temperature":
        temperatures = [0.0, 0.1, 0.05, 0.01, 0.005]
        hyperparameter_list = [0.0]
        draw_with_feature_weights = True
    elif method_name == "fw-mrs-svm":
        temperatures = [0.0, 0.1, 0.05, 0.01, 0.005]
        hyperparameter_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        draw_with_feature_weights = True

    method_names = temperatures + ["Uniform", "Exponentiated Gradient"]
    for key in method_names:
        prediction_proba_dict[key] = []
        prediction_dict[key] = []

    skf = RepeatedStratifiedKFold(
        n_splits=n_cv_splits,
        n_repeats=n_cv_repeats,
        random_state=seed,
    )
    for i, (train_indices, test_indices) in enumerate(
        skf.split(sample_df, sample_df[target])
    ):
        N, R, T = create_sets(df, sensitive_attribute, train_indices, test_indices)

        if len(sample_weight_list) > i and load_previous_results:
            sample_weights_n = sample_weight_list[i]
            feature_weights = feature_weight_list[i]
        else:
            sample_weights_n, feature_weights = sample_weighting_method(
                N=N,
                R=R,
                columns=columns,
                save_path=result_path,
                bias_variable=target,
                drop=drop,
                early_stopping=True,
                random_generator=random_generator,
                target=target,
                budgets=temperatures,
                hyperparameter_list=hyperparameter_list,
                method_name=method_name,
            )

            feature_weight_list.append(feature_weights)
            sample_weight_list.append(sample_weights_n)

            save_weights(sample_weights_save_path, sample_weight_list)
            save_weights(feature_weights_save_path, feature_weight_list)

        sample_weights_r = np.ones(len(R)) / len(R)
        for key, value in sample_weights_n.items():
            value = value / np.sum(value)
            value = np.concatenate([value, sample_weights_r]).tolist()
            sample_weights_n[key] = value

        n_splits = 5
        n_estimators = 200
        train = pd.concat([N, R])
        weighted_clf_list = compute_classification_metrics_random_forest_fairness(
            train,
            columns,
            sensitive_attribute,
            sample_weights_n,
            feature_weights,
            target,
            random_state=seed,
            draw_with_feature_weights=draw_with_feature_weights,
            splitter="feature_weighted_best",
            n_estimators=n_estimators,
            n_splits=n_splits,
        )

        for clf, temperature in zip(weighted_clf_list, temperatures):
            weighted_prediction = clf.predict(T[columns].values)
            prediction_proba_dict[temperature].extend(
                clf.predict_proba(T[columns].values)[:, 1]
            )
            prediction_dict[temperature].extend(clf.predict(T[columns].values))
            metric_frame = MetricFrame(
                metrics=metrics,
                y_true=T[target],
                y_pred=weighted_prediction,
                sensitive_features=T[sensitive_attribute],
            )
            metric_frame.by_group.plot.bar(
                subplots=True,
                layout=[2, 3],
                legend=False,
                figsize=[16, 8],
                title="Show all metrics",
            )
            plt.savefig(result_path / f"{temperature}_weighted_metrics.pdf")
            plt.clf()

        uniform_clf = compute_classification_metrics_random_forest_fairness(
            train,
            columns,
            sensitive_attribute,
            None,
            None,
            target,
            random_state=seed,
            draw_with_feature_weights=False,
            splitter="best",
            n_estimators=n_estimators,
            n_splits=n_splits,
        )
        uniform_prediction = uniform_clf.predict(T[columns].values)
        prediction_proba_dict["Uniform"].extend(
            uniform_clf.predict_proba(T[columns].values)[:, 1]
        )
        prediction_dict["Uniform"].extend(uniform_clf.predict(T[columns].values))
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=T[target],
            y_pred=uniform_prediction,
            sensitive_features=T[sensitive_attribute],
        )
        metric_frame.by_group.plot.bar(
            subplots=True,
            layout=[2, 3],
            legend=False,
            figsize=[16, 8],
            title="Show all metrics",
        )
        plt.savefig(result_path / "uniform_weighted_metrics.pdf")
        plt.clf()

        mitigated_clf = compute_classification_metrics_random_forest_fairness(
            train,
            columns,
            sensitive_attribute,
            None,
            None,
            target,
            random_state=seed,
            draw_with_feature_weights=False,
            splitter="best",
            n_estimators=n_estimators,
            n_splits=n_splits,
            mitigate=True,
        )
        mitigated_prediction = mitigated_clf.predict(T[columns].values)
        prediction_proba_dict["Exponentiated Gradient"].extend(
            mitigated_clf.predict(T[columns].values)
        )
        prediction_dict["Exponentiated Gradient"].extend(
            mitigated_clf.predict(T[columns].values)
        )
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=T[target],
            y_pred=mitigated_prediction,
            sensitive_features=T[sensitive_attribute],
        )
        metric_frame.by_group.plot.bar(
            subplots=True,
            layout=[2, 3],
            legend=False,
            figsize=[16, 8],
            title="Show all metrics",
        )
        plt.savefig(result_path / "mitigated_metrics.pdf")
        plt.clf()

        target_list.extend(T[target])
        sensitive_attribute_list.extend(T[sensitive_attribute])
        plot_model_comparison(
            y_preds=prediction_proba_dict,
            y_true=target_list,
            sensitive_features=sensitive_attribute_list,
            x_axis_metric=roc_auc_score,
            y_axis_metric=selection_rate_difference,
            axis_labels=("AUROC", "Selection Rate Difference"),
            point_labels=True,
        )
        plt.savefig(result_path / "auroc_metric_comparison.pdf")
        plt.clf()

        plot_model_comparison(
            y_preds=prediction_proba_dict,
            y_true=target_list,
            sensitive_features=sensitive_attribute_list,
            x_axis_metric=average_precision_score,
            y_axis_metric=selection_rate_difference,
            axis_labels=("AUPRC", "Selection Rate Difference"),
            point_labels=True,
        )
        plt.savefig(result_path / "auprc_metric_comparison.pdf")
        plt.clf()

        plot_model_comparison(
            y_preds=prediction_dict,
            y_true=target_list,
            sensitive_features=sensitive_attribute_list,
            x_axis_metric=accuracy_score,
            y_axis_metric=selection_rate_difference,
            axis_labels=("Accuracy", "Selection Rate Difference"),
            point_labels=True,
        )
        plt.savefig(result_path / "accuracy_metric_comparison.pdf")
        plt.clf()

        plt.close()


def create_sets(df, sensitive_attribute, train_indices, test_indices):
    train = df.iloc[train_indices].copy()
    N = train[train[sensitive_attribute] == 1].copy()
    N["label"] = 1
    R = train[train[sensitive_attribute] == 0].copy()
    R["label"] = 0
    T = df.iloc[test_indices]
    return N, R, T


def selection_rate_difference(y_true, y_pred, sensitive_features):
    y_pred = np.round(y_pred)
    positive_indices = np.where(sensitive_features == 1)
    negative_indices = np.where(sensitive_features == 0)
    positive_selection_rate = selection_rate(
        y_true[positive_indices], y_pred[positive_indices]
    )
    negative_selection_rate = selection_rate(
        y_true[negative_indices], y_pred[negative_indices]
    )
    return abs(positive_selection_rate - negative_selection_rate)
