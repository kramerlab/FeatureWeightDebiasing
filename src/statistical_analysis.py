import json
import random
import numpy as np
import scipy.stats
from pathlib import Path
from tqdm import trange

from utils.command_line_arguments import (
    parse_command_line_arguments_statistical_analysis,
)

from utils.data_loader import load_dataset
from utils.metrics import calculate_rbf_gamma, compute_metrics, scale_df
from utils.statistics import logistic_regression
from utils.visualization import plot_statistical_analysis
from weighting_methods import mrs, soft_mrs_weighting


bins = 25
seed = 5
# Used to draw radom states
max_int = 2**32 - 1


def statistical_analysis(
    n_repeats=1000,
    drop=1,
    patience=25,
):
    """Analyze GBS corrected with Allensbach with two methods.

    :param method_one: First method
    :param method_two: Second method
    """
    method_one = "mrs"
    method_two = "soft_mrs"
    np.random.seed(seed)
    random.seed(seed)
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../results/statistical analysis")
    iterations_path = result_path / "iteration"
    mrs_iterations_path = iterations_path / "mrs"
    soft_mrs_iterations_path = iterations_path / "soft_mrs"
    mrs_iterations_path.mkdir(exist_ok=True, parents=True)
    soft_mrs_iterations_path.mkdir(exist_ok=True, parents=True)
    df, scale_columns, _ = load_dataset("gbs_allensbach")
    scaled_df, scaler = scale_df(df, scale_columns)
    first_random_generator = np.random.RandomState(seed)
    second_random_generator = np.random.RandomState(seed)

    gamma = calculate_rbf_gamma(scaled_df[scale_columns])
    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]
    uniform_weights = np.ones(len(scaled_N)) / len(scaled_N)
    mrs_weights_list = []
    soft_mrs_weights_list = []
    mrs_wasserstein_list = []
    soft_mrs_wasserstein_list = []
    mrs_relative_biases_list = []
    soft_mrs_relative_bias_list = []

    (
        _,
        sample_biases_uniform,
        wasserstein_distances_uniform,
    ) = compute_metrics(
        scaled_N[scale_columns].copy(),
        scaled_R[scale_columns].copy(),
        scaler,
        scale_columns,
        scale_columns,
        uniform_weights,
        gamma,
    )

    lr_pvalue_uniform_gbs = logistic_regression(
        scaled_N[scale_columns + ["Wahlteilnahme"]], uniform_weights
    )

    for i in trange(n_repeats):
        weights_mrs = mrs(
            scaled_N,
            scaled_R,
            scale_columns,
            drop=drop,
            early_stopping=True,
            random_generator=first_random_generator,
        )
        mrs_weights_list.append(weights_mrs)

        weights_soft_mrs = soft_mrs_weighting(
            scaled_N,
            scaled_R,
            scale_columns,
            drop=drop,
            early_stopping=True,
            random_generator=second_random_generator,
            patience=patience,
        )
        soft_mrs_weights_list.append(weights_soft_mrs)

        (
            _,
            sample_biases_mrs,
            wasserstein_distances_mrs,
        ) = compute_metrics(
            scaled_N[scale_columns].copy(),
            scaled_R[scale_columns].copy(),
            scaler,
            scale_columns,
            scale_columns,
            weights_mrs,
            gamma,
        )
        mrs_wasserstein_list.append(wasserstein_distances_mrs)
        mrs_relative_biases_list.append(sample_biases_mrs)

        (
            _,
            sample_biases_soft_mrs,
            wasserstein_distances_soft_mrs,
        ) = compute_metrics(
            scaled_N[scale_columns].copy(),
            scaled_R[scale_columns].copy(),
            scaler,
            scale_columns,
            scale_columns,
            weights_soft_mrs,
            gamma,
        )
        soft_mrs_wasserstein_list.append(wasserstein_distances_soft_mrs)
        soft_mrs_relative_bias_list.append(sample_biases_soft_mrs)

        lr_pvalue_weighted_mrs = logistic_regression(
            scaled_N[scale_columns + ["Wahlteilnahme"]], weights_mrs
        )

        lr_pvalue_weighted_soft_mrs = logistic_regression(
            scaled_N[scale_columns + ["Wahlteilnahme"]], weights_soft_mrs
        )

        result_dict_mrs_iteration = {}
        for index, column in enumerate(scale_columns):
            result_dict_mrs_iteration[f"{column}_relative_bias"] = {
                "wasserstein": wasserstein_distances_mrs[index],
                "relative_bias": sample_biases_mrs[index],
            }
        with open(
            mrs_iterations_path / f"results_{method_one}_{i}.json", "w"
        ) as result_file:
            result_file.write(json.dumps(result_dict_mrs_iteration))

        result_dict_soft_mrs_iteration = {}
        for index, column in enumerate(scale_columns):
            result_dict_soft_mrs_iteration[f"{column}_relative_bias"] = {
                "wasserstein": wasserstein_distances_soft_mrs[index],
                "relative_bias": sample_biases_soft_mrs[index],
            }
        with open(
            soft_mrs_iterations_path / f"results_{method_two}_{i}.json", "w"
        ) as result_file:
            result_file.write(json.dumps(result_dict_soft_mrs_iteration))

    mrs_wasserstein_confidence_list = compute_confidence_interval(mrs_wasserstein_list)
    soft_mrs_wasserstein_confidence_list = compute_confidence_interval(
        soft_mrs_wasserstein_list
    )

    mrs_relative_bias_confidence_list = compute_confidence_interval(
        mrs_relative_biases_list
    )
    soft_mrs_relative_bias_confidence_list = compute_confidence_interval(
        soft_mrs_relative_bias_list
    )

    # Save uniform results
    result_dict_uniform = {}
    for index, column in enumerate(scale_columns):
        result_dict_uniform[f"{column}_bias"] = {
            "wasserstein": wasserstein_distances_uniform[index],
            "relative_bias": sample_biases_uniform[index],
        }
        result_dict_uniform["p_value"] = lr_pvalue_uniform_gbs
    with open(result_path / "results_uniform.json", "w") as result_file:
        result_file.write(json.dumps(result_dict_uniform))

    # Save methods mean results
    result_dict_mrs_mean = {}
    for index, column in enumerate(scale_columns):
        result_dict_mrs_mean[f"{column}_bias"] = {
            "wasserstein": mrs_wasserstein_confidence_list[index].tolist(),
            "relative_bias": mrs_relative_bias_confidence_list[index].tolist(),
        }
    with open(result_path / "results_mrs_mean.json", "w") as result_file:
        result_file.write(json.dumps(result_dict_mrs_mean))

    result_dict_soft_mrs_mean = {}
    for index, column in enumerate(scale_columns):
        result_dict_soft_mrs_mean[f"{column}_bias"] = {
            "wasserstein": soft_mrs_wasserstein_confidence_list[index].tolist(),
            "relative_bias": soft_mrs_relative_bias_confidence_list[index].tolist(),
        }
    with open(result_path / "results_soft_mrs_mean.json", "w") as result_file:
        result_file.write(json.dumps(result_dict_soft_mrs_mean))

    scaled_N.loc[:, scale_columns] = scaler.inverse_transform(scaled_N[scale_columns])
    scaled_R.loc[:, scale_columns] = scaler.inverse_transform(scaled_R[scale_columns])

    # Plot exemplary a random iteration
    random_iteration = np.random.randint(0, len(mrs_weights_list))
    plot_statistical_analysis(
        bins,
        scaled_N[scale_columns],
        scaled_R[scale_columns],
        result_path,
        mrs_weights_list[random_iteration],
        soft_mrs_weights_list[random_iteration],
        "MRS",
        "Soft-MRS",
    )


def compute_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    n = data.shape[1]
    mean = np.mean(data, axis=0)
    standard_error = scipy.stats.sem(data, axis=0)
    h = standard_error * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)

    return np.stack([mean - h, mean, mean + h], axis=1)


if __name__ == "__main__":
    args = parse_command_line_arguments_statistical_analysis()
    statistical_analysis(args.n_repeats, args.drop, args.patience)
