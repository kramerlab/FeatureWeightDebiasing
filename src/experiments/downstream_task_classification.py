import json
import numpy as np

from tqdm import trange
from sklearn.discriminant_analysis import StandardScaler

from utils.gradient_descent import compute_classification_metrics_gradient_descent
from utils.statistics import create_result_path, write_result_dict_test_set
from utils.sampling import sample_with_test_set
from utils.visualization import plot_sample_weights, plot_feature_weights
from utils.metrics import (
    compute_classification_metrics_random_forest,
    compute_classification_metrics_tree,
)

seed = 5


def downstream_experiment_with_test_set(
    df,
    columns,
    sample_weighting_method,
    target: str,
    number_of_repetitions: int = 50,
    bias_type: str = None,
    data_set_name: str = "",
    random_generator=None,
    explicit_weights=True,
    load_previous_results=True,
    budget="",
    bias_fraction=0.75,
    drop=1,
    validation_method="random_forest",
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
        experiment_name="test_set_classification",
        bias_fraction=bias_fraction,
    )
    sample_weights_save_path = result_path / "sample_weights"
    feature_weights_save_path = result_path / "feature_weights"
    inverse_feature_weights_save_path = result_path / "inverse_feature_weights"
    classificiation_result_path = result_path / "classification_results"

    result_path.mkdir(exist_ok=True)
    classificiation_result_path.mkdir(exist_ok=True)
    sample_weights_save_path.mkdir(exist_ok=True)
    feature_weights_save_path.mkdir(exist_ok=True)
    inverse_feature_weights_save_path.mkdir(exist_ok=True)

    sample_weight_list = load_weights(sample_weights_save_path)
    feature_weight_list = load_weights(feature_weights_save_path)
    inverse_feature_weight_list = load_weights(feature_weights_save_path)

    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    sample_df = df.copy()

    if method_name == "fw-mrs-temperature":
        splitter = "feature_weighted_best"
        draw_with_feature_weights = True
        temperatures = [0.1, 0.05, 0.01]
    else:
        splitter = "best"
        draw_with_feature_weights = False
        temperatures = None

    for i in trange(number_of_repetitions):
        N, R, T = sample_with_test_set(
            bias_type,
            sample_df,
            target,
            train_fraction=0.5,
            bias_fraction=bias_fraction,
            test_fraction=0.2,
            columns=columns,
        )

        if len(sample_weight_list) > i and explicit_weights and load_previous_results:
            sample_weights = sample_weight_list[i]
            feature_weights = feature_weight_list[i]
            inverse_feature_weights = inverse_feature_weight_list[i]

        else:
            sample_weights, feature_weights, inverse_feature_weights = (
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
                    budgets=temperatures,
                    validation_method=validation_method,
                    method_name=method_name,
                )
            )

            feature_weight_list.append(feature_weights)
            inverse_feature_weight_list.append(inverse_feature_weights)
            sample_weight_list.append(sample_weights)
            
            save_weights(sample_weights_save_path, sample_weight_list)
            save_weights(feature_weights_save_path, feature_weight_list)
            save_weights(inverse_feature_weights_save_path, inverse_feature_weight_list)

        dropped_samples = np.count_nonzero(sample_weights == 0.0)
        dropped_samples_list.append(dropped_samples)

        if feature_weights is None:
            feature_weights = (np.ones(len(columns)) / len(columns)).tolist()
            inverse_feature_weights = (np.ones(len(columns)) / len(columns)).tolist()

        gradient_ascent_auroc, gradient_ascent_auprc = (
            compute_classification_metrics_gradient_descent(
                N,
                R,
                T,
                columns,
                sample_weights,
                inverse_feature_weights,
                target,
                random_state=seed,
                n_splits=5,
            )
        )

        rf_auroc, rf_auprc = compute_classification_metrics_random_forest(
            N,
            R,
            T,
            columns,
            sample_weights,
            feature_weights,
            target,
            random_state=seed,
            draw_with_feature_weights=draw_with_feature_weights,
            splitter=splitter,
            n_estimators=500,
            n_splits=5,
        )

        tree_auroc = 0
        tree_auprc = 0

        # plot_sample_weights(sample_weights, sample_weights_save_path, i)
        # plot_feature_weights(feature_weights, feature_weights_save_path, i)

        rf_auroc_list.append(rf_auroc)
        rf_auprc_list.append(rf_auprc)
        tree_auroc_list.append(tree_auroc)
        tree_auprc_list.append(tree_auprc)
        gradient_ascent_auroc_list.append(gradient_ascent_auroc)
        gradient_ascent_auprc_list.append(gradient_ascent_auprc)

    result_dict = write_result_dict_test_set(
        rf_auroc_list,
        rf_auprc_list,
        tree_auroc_list,
        tree_auprc_list,
        gradient_ascent_auroc_list,
        gradient_ascent_auprc_list,
        dropped_samples_list,
        len(N),
    )

    with open(result_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    for result_list, file_name in zip(
        (
            rf_auroc_list,
            rf_auprc_list,
            tree_auroc_list,
            tree_auprc_list,
            gradient_ascent_auroc_list,
            gradient_ascent_auprc_list,
            dropped_samples_list,
        ),
        (
            "rf_auroc",
            "rf_auprc",
            "tree_auroc",
            "tree_auprc",
            "gradient_ascent_auroc",
            "gradient_ascent_auprc",
            "dropped_samples",
        ),
    ):
        with open(
            classificiation_result_path / f"{file_name}.json", "w"
        ) as result_file:
            result_file.write(json.dumps(result_list))


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
