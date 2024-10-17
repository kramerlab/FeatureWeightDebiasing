import json

from experiments.downstream_tasks import load_weights, repeated_train_val_test_split
from utils.data_loader import save_weights
from utils.statistics import create_result_path
from utils.sampling import sample_N
from utils.metrics import scale_df
from utils.visualization_fw_mrs import (
    plot_budget_comparison_auroc,
    plot_budget_comparison_auroc_mean,
)
import numpy as np

seed = 5
sampling_random_generator = np.random.RandomState(seed)


def temperature_comparison(
    df,
    columns,
    sample_weighting_method,
    target: str,
    n_cv_splits: int = 3,
    n_cv_repeats=10,
    bias_type: str = None,
    data_set_name: str = "",
    random_generator=None,
    method_name=None,
    drop=1,
    bias_fraction=0.1,
    load_previous_results=True,
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

    temperatures = [0.0, 0.1, 0.05, 0.01, 0.005]
    hyperparameter_list = [0.0]
    mean = True if method_name == "fw-mrs-temperature-mean" else False
    dropped_samples_dict = {temperature: [] for temperature in temperatures}

    result_path = create_result_path(
        method_name,
        bias_type,
        data_set_name,
        experiment_name="temperature_comparison",
        bias_fraction=bias_fraction,
    )

    sample_weights_save_path = result_path / "sample_weights"
    feature_weights_save_path = result_path / "feature_weights"
    auroc_save_path = result_path / "method_aurocs"

    sample_weights_save_path.mkdir(exist_ok=True)
    feature_weights_save_path.mkdir(exist_ok=True)
    auroc_save_path.mkdir(exist_ok=True)

    feature_weighted_aurocs_list = load_weights(
        auroc_save_path, file_name="method_aurocs"
    )
    sample_weights_list = load_weights(sample_weights_save_path)
    feature_weights_list = load_weights(feature_weights_save_path)

    sample_df, _ = scale_df(df, columns)
    number_of_samples_list = []

    if data_set_name in ("gbs_gesis", "gbs_allensbach"):
        N = sample_df[sample_df["label"] == 1]
        R = sample_df[sample_df["label"] == 0]

    for i, (N, R, _) in enumerate(
        repeated_train_val_test_split(
            n_cv_splits,
            n_cv_repeats,
            sample_df,
            sample_df[target],
            sampling_random_generator,
        )
    ):
        if data_set_name not in ("gbs_gesis", "gbs_allensbach"):
            N = sample_N(
                train=N,
                bias_type=bias_type,
                bias_fraction=bias_fraction,
                columns=columns,
                bias_variable=target,
                random_generator=sampling_random_generator,
            )
            N["label"] = 1
            R["label"] = 0

        if len(sample_weights_list) > i and load_previous_results:
            sample_weights = sample_weights_list[i]
            feature_weights = feature_weights_list[i]
            random_forest_feature_weighted_aurocs = feature_weighted_aurocs_list[i]

        else:
            random_forest_feature_weighted_aurocs, sample_weights, feature_weights = (
                sample_weighting_method(
                    N=N,
                    R=R,
                    target=target,
                    columns=columns,
                    save_path=result_path,
                    bias_variable=target,
                    drop=drop,
                    random_generator=random_generator,
                    budgets=temperatures,
                    hyperparameter_list=hyperparameter_list,
                    method_name=method_name,
                    return_metrics=True,
                    mean=mean,
                )
            )
            random_forest_feature_weighted_aurocs = {
                temperature: random_forest_feature_weighted_aurocs[temperature][
                    hyperparameter_list[0]
                ]
                for temperature in temperatures
            }

            feature_weights_list.append(feature_weights)
            sample_weights_list.append(sample_weights)
            feature_weighted_aurocs_list.append(random_forest_feature_weighted_aurocs)

            save_weights(sample_weights_save_path, sample_weights_list)
            save_weights(feature_weights_save_path, feature_weights_list)
            save_weights(
                auroc_save_path, feature_weighted_aurocs_list, file_name="method_aurocs"
            )

        for temperature, values in sample_weights.items():
            key = next(iter(values.keys()))
            dropped_samples_dict[float(temperature)].append(
                np.count_nonzero(np.array(values[key]) == 0.0)
            )
        number_of_samples_list.append(len(N))

        plot_budget_comparison_auroc(
            random_forest_feature_weighted_aurocs,
            number_of_samples_list,
            drop,
            auroc_save_path / f"auroc_{i}",
        )

    for temperature, values in dropped_samples_dict.items():
        dropped_samples_dict[temperature] = np.mean(dropped_samples_dict[temperature])

    with open(result_path / "dropped_elements.json", "w") as result_file:
        result_file.write(json.dumps(dropped_samples_dict))
        
    plot_budget_comparison_auroc_mean(
        feature_weighted_aurocs_list,
        number_of_samples_list,
        drop,
        result_path / "mean_auroc",
    )


# save_mean_dropped_elements(result_path, dropped_samples_list_dict)


def save_mean_dropped_elements(result_path, dropped_samples_list):
    mean_dropped_samples_dict = {}
    for key, value in dropped_samples_list.items():
        dropped_elements = []
        dropped_elements.append(value)
        mean_dropped_samples_dict[f"{key} mean"] = np.mean(dropped_elements)
        mean_dropped_samples_dict[f"{key} std"] = np.std(dropped_elements)

    with open(result_path / "mean_dropped_samples", "w", encoding="utf-8") as file:
        json.dump(mean_dropped_samples_dict, file, indent=4)
