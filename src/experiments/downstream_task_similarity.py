import json
import numpy as np

from tqdm import trange
from sklearn.discriminant_analysis import StandardScaler

from utils.statistics import create_result_path, write_result_dict
from utils.sampling import sample
from utils.visualization import plot_feature_weights, plot_sample_weights
from utils.metrics import (
    compute_classification_metrics_random_forest,
    compute_metrics,
    calculate_rbf_gamma,
)
from utils.gradient_descent import compute_classification_metrics_gradient_descent

seed = 5


def downstream_experiment(
    df,
    columns,
    sample_weighting_method,
    target: str,
    number_of_repetitions: int = 50,
    bias_type: str = None,
    data_set_name: str = "",
    random_generator=None,
    explicit_weights=True,
    load_previous_results=False,
    method_name=None,
    budget=0.0,
    bias_fraction=0.25,
    validation_method="random_forest",
    drop=1,
    **args
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
    weighted_mmds_list = []
    biases_list = []
    wasserstein_distance_list = []
    rf_auroc_list = []
    rf_auprc_list = []

    tree_auroc_list = []
    tree_auprc_list = []

    gradient_ascent_auroc_list = []
    gradient_ascent_auprc_list = []

    dropped_samples_list = []

    result_path = create_result_path(
        method_name,
        bias_type,
        data_set_name,
        "R_classsification",
        bias_fraction=bias_fraction,
    )
    sample_weights_save_path = result_path / "sample_weights"
    feature_weights_save_path = result_path / "feature_weights"

    sample_weights_save_path.mkdir(exist_ok=True)
    feature_weights_save_path.mkdir(exist_ok=True)

    sample_weight_list = load_weights(sample_weights_save_path)
    feature_weights_list = load_weights(feature_weights_save_path)

    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    sample_df = df.copy()

    if method_name in ("fw-mrs-temperature", "mrs-tree", "mrs-forest"):
        drop_samples = True
    else:
        drop_samples = False

    draw_with_feature_weights = True if method_name == "fw-mrs-temperature" else False
    splitter = (
        "feature_weighted_best"
        if method_name in ("fw-mrs-temperature", "fw-mrs-budget")
        else "best"
    )

    for i in trange(number_of_repetitions):
        N, R = sample(
            bias_type,
            sample_df,
            target,
            train_fraction=0.5,
            bias_fraction=bias_fraction,
            columns=columns,
        )

        gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))

        if len(sample_weight_list) > i and explicit_weights and load_previous_results:
            sample_weights = np.array(sample_weight_list[i])
            feature_weights = np.array(feature_weights_list[i])
        else:
            sample_weights, feature_weights = (
                sample_weighting_method(
                    N=N,
                    R=R,
                    columns=columns,
                    save_path=result_path,
                    bias_variable=target,
                    drop=drop,
                    early_stopping=True,
                    random_generator=random_generator,
                    target=target,
                    budgets=[budget],
                    validation_method=validation_method,
                    method_name=method_name,
                )
            )

            feature_weights_list.append(feature_weights)
            sample_weight_list.append(sample_weights)

            save_weights(sample_weights_save_path, sample_weight_list)
            save_weights(feature_weights_save_path, feature_weights_list)

        dropped_samples = np.count_nonzero(np.array(sample_weights) == 0.0)
        dropped_samples_list.append(dropped_samples)
        if feature_weights is None:
            feature_weights = (np.ones(len(columns)) / len(columns)).tolist()

        weighted_mmd, relative_bias, wasserstein_distances = compute_metrics(
            N,
            R,
            scaler,
            columns,
            columns,
            sample_weights,
            gamma,
        )

        gradient_ascent_auroc, gradient_ascent_auprc = 0, 0

        rf_auroc, rf_auprc, _, _, _, _ = compute_classification_metrics_random_forest(
            N,
            R,
            R,
            columns,
            sample_weights,
            feature_weights,
            target,
            random_state=seed,
            draw_with_feature_weights=draw_with_feature_weights,
            splitter=splitter,
            drop_samples=drop_samples,
        )

        plot_sample_weights(sample_weights, sample_weights_save_path, i)
        if not feature_weights is None:
            plot_feature_weights(feature_weights, feature_weights_save_path, i)

            weighted_mmds_list.append(weighted_mmd)
            biases_list.append(relative_bias)
            wasserstein_distance_list.append(wasserstein_distances)
            rf_auroc_list.append(rf_auroc)
            rf_auprc_list.append(rf_auprc)
            tree_auroc_list.append(0)
            tree_auprc_list.append(0)
            gradient_ascent_auroc_list.append(gradient_ascent_auroc)
            gradient_ascent_auprc_list.append(gradient_ascent_auprc)

    result_dict = write_result_dict(
        N.drop(["label"], axis="columns").columns,
        weighted_mmds_list,
        biases_list,
        rf_auroc_list,
        rf_auprc_list,
        tree_auroc_list,
        tree_auprc_list,
        gradient_ascent_auroc_list,
        gradient_ascent_auprc_list,
        dropped_samples_list,
        len(N),
        explicit_weights=explicit_weights,
    )

    with open(result_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))


def save_weights(path, weights_list):
    with open(path / "weights.json", "w") as file:
        json.dump(weights_list, file, indent=4)


def load_weights(path):
    weight_file = path / "weights.json"
    if weight_file.is_file():
        with open(weight_file, "r") as file:
            weights = json.load(file)
    else:
        weights = []
    return weights
