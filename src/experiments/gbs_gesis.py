import json
import numpy as np
from pathlib import Path
from utils.metrics import calculate_rbf_gamma, compute_metrics, scale_df
from utils.visualization import (
    plot_statistical_analysis,
    plot_results_with_variance,
    plot_weights,
)


bins = 25
seed = 5


def gbs_gesis_experiment(
    df,
    columns,
    weighting_method,
    method="",
    **args,
):
    """_summary_

    :param df: The data set as pandas.DataFrame
    :param columns: Name of training columns
    :param weighting_method: The weighting function
    :param method: Method name, defaults to ""
    """
    random_generator = np.random.RandomState(seed)
    visualisation_path = create_result_path(method)
    df = df.sample(frac=1)
    mmd_list = []
    mean_list = []
    scale_columns = columns
    scaled_df, scaler = scale_df(df, scale_columns)

    gamma = calculate_rbf_gamma(scaled_df[columns])

    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]

    weights = weighting_method(
        scaled_N,
        scaled_R,
        columns,
        save_path=visualisation_path,
        mean_list=mean_list,
        mmd_list=mmd_list,
        drop=1,
        random_generator=random_generator,
    )

    (
        weighted_mmd,
        weighted_ssmd,
        sample_biases,
        wasserstein_distances,
    ) = compute_metrics(
        scaled_N[columns],
        scaled_R[columns],
        weights,
        scaler,
        scale_columns,
        columns,
        gamma,
    )

    remaining_samples = np.count_nonzero(weights != 0)
    plot_weights(weights, visualisation_path, 0, "GBS")

    if "neural_network_mmd_loss" in method:
        biases_path = visualisation_path / "MMDs"
        biases_path.mkdir(exist_ok=True)
        plot_results_with_variance(
            [],
            [mmd_list[-1]],
            [],
            biases_path,
            "",
        )

    result_dict = {
        "MMDs": weighted_mmd,
        "Remaining Samples": remaining_samples,
        "Number of Samples": len(scaled_N),
    }
    for index, column in enumerate(columns):
        result_dict[f"{column}_relative_bias"] = {
            "bias": sample_biases[index],
            "ssmd": weighted_ssmd[index],
            "wasserstein": wasserstein_distances[index],
        }

    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    plot_statistical_analysis(
        bins,
        scaled_N[columns],
        scaled_R[columns],
        visualisation_path,
        weights,
    )


def create_result_path(method):
    """Creates the result path and makes the directory.

    :param method: Method name
    :return: The result path
    """
    result_path = Path("../results")
    result_path = result_path / method / "gbs_gesis"
    result_path.mkdir(exist_ok=True, parents=True)
    return result_path
