import json
import random
import numpy as np

from pathlib import Path
from utils.command_line_arguments import (
    get_weighting_function,
    parse_command_line_arguments_statistical_analysis,
)
from utils.data_loader import load_dataset
from utils.metrics import calculate_rbf_gamma, compute_metrics, scale_df
from utils.statistics import logistic_regression
from utils.visualization import plot_statistical_analysis

bins = 25
seed = 5
# Used to draw radom states
max_int = 2**32 - 1


def statistical_analysis(
    method_one,
    method_two,
    **args,
):
    """Analyze GBS corrected with Allensbach with two methods.

    :param method_one: First method
    :param method_two: Second method
    """
    np.random.seed(seed)
    random.seed(seed)
    random_generator = np.random.RandomState(seed)
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../results")
    visualisation_path = (
        result_path / "statistical analysis" / f"{method_one}_{method_two}"
    )
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df, scale_columns, _ = load_dataset("gbs_allensbach")
    scaled_df, scaler = scale_df(df, scale_columns)

    weighting_function_one = get_weighting_function(method_one)
    weighting_function_two = get_weighting_function(method_two)

    gamma = calculate_rbf_gamma(scaled_df[scale_columns])

    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]

    weights_one = weighting_function_one(
        scaled_N,
        scaled_R,
        scale_columns,
        save_path=visualisation_path,
        drop=1,
        early_stopping=True,
        random_generator=random_generator,
        patience=25,
    )

    random_generator = np.random.RandomState(seed)
    weights_two = weighting_function_two(
        scaled_N,
        scaled_R,
        scale_columns,
        save_path=visualisation_path,
        drop=1,
        early_stopping=True,
        random_generator=random_generator,
    )

    weights_uniform = np.ones(len(scaled_N)) / len(scaled_N)

    (
        _,
        sample_biases_one,
        wasserstein_distances_one,
    ) = compute_metrics(
        scaled_N[scale_columns].copy(),
        scaled_R[scale_columns].copy(),
        weights_one,
        scaler,
        scale_columns,
        scale_columns,
        gamma,
    )

    (
        _,
        sample_biases_two,
        wasserstein_distances_two,
    ) = compute_metrics(
        scaled_N[scale_columns].copy(),
        scaled_R[scale_columns].copy(),
        weights_two,
        scaler,
        scale_columns,
        scale_columns,
        gamma,
    )

    (
        _,
        sample_biases_uniform,
        wasserstein_distances_uniform,
    ) = compute_metrics(
        scaled_N[scale_columns].copy(),
        scaled_R[scale_columns].copy(),
        weights_uniform,
        scaler,
        scale_columns,
        scale_columns,
        gamma,
    )

    result_dict_one = {}
    for index, column in enumerate(scale_columns):
        result_dict_one[f"{column}_relative_bias"] = {
            "wasserstein": wasserstein_distances_one[index],
            "relative_bias": sample_biases_one[index],
        }

    result_dict_two = {}
    for index, column in enumerate(scale_columns):
        result_dict_two[f"{column}_relative_bias"] = {
            "wasserstein": wasserstein_distances_two[index],
            "relative_bias": sample_biases_two[index],
        }

    result_dict_uniform = {}
    for index, column in enumerate(scale_columns):
        result_dict_uniform[f"{column}_relative_bias"] = {
            "wasserstein": wasserstein_distances_uniform[index],
            "relative_bias": sample_biases_uniform[index],
        }

    lr_pvalue_gbs, lr_pvalue_weighted_one = logistic_regression(
        scaled_N[scale_columns + ["Wahlteilnahme"]], weights_one
    )

    _, lr_pvalue_weighted_two = logistic_regression(
        scaled_N[scale_columns + ["Wahlteilnahme"]], weights_two
    )

    result_dict = {
        "logistig regression p values": {
            "GBS": lr_pvalue_gbs,
            f"Weighted {method_one}": lr_pvalue_weighted_one,
            f"Weighted {method_two}": lr_pvalue_weighted_two,
        }
    }

    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    with open(visualisation_path / f"results_{method_one}.json", "w") as result_file:
        result_file.write(json.dumps(result_dict_one))

    with open(visualisation_path / f"results_{method_two}.json", "w") as result_file:
        result_file.write(json.dumps(result_dict_two))

    with open(visualisation_path / "results_uniform.json", "w") as result_file:
        result_file.write(json.dumps(result_dict_uniform))

    scaled_N.loc[:, scale_columns] = scaler.inverse_transform(scaled_N[scale_columns])
    scaled_R.loc[:, scale_columns] = scaler.inverse_transform(scaled_R[scale_columns])

    plot_statistical_analysis(
        bins,
        scaled_N[scale_columns],
        scaled_R[scale_columns],
        visualisation_path,
        weights_one,
        weights_two,
        method_one,
        method_two,
    )


if __name__ == "__main__":
    args = parse_command_line_arguments_statistical_analysis()
    statistical_analysis(args.method_one, args.method_two)
