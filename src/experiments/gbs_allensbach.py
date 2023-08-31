import json
import numpy as np

from pathlib import Path

from utils.metrics import calculate_rbf_gamma, compute_metrics, scale_df
from utils.statistics import logistic_regression
from utils.visualization import (
    plot_statistical_analysis,
    plot_weights,
)

bins = 25
seed = 5

def gbs_allensbach_experiment(
    df,
    columns,
    weighting_method,
    method,
    number_of_repetitions,
    **args,
):
    """Performs the experiments with GBS and Allensbach. The weighting method is used to compute the
    sample weights. The results are visualized and saved into a file.

    :param df: The data set as pandas.DataFrame
    :param columns: Name of training columns
    :param weighting_method: The weighting function
    :param method: Method name
    :param number_of_repetitions: Number of repetetions of the experiment
    """
    random_generator = np.random.RandomState(seed)
    result_path = create_result_path(method)
    df = df.sample(frac=1)
    mmd_list = []
    mean_list = []
    weighted_mmds_list = []
    biases_list = []
    wasserstein_parameter_list = []
    remaining_samples_list = []
    scale_columns = columns
    scaled_df, scaler = scale_df(df, scale_columns)

    gamma = calculate_rbf_gamma(scaled_df[columns])

    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]

    for i in range(number_of_repetitions):
        weights = weighting_method(
            scaled_N,
            scaled_R,
            columns,
            save_path=result_path,
            mean_list=mean_list,
            mmd_list=mmd_list,
            drop=1,
            early_stopping=True,
            random_generator=random_generator
        )

        (
            weighted_mmd,
            sample_biases,
            wasserstein_distances,
        ) = compute_metrics(
            scaled_N, scaled_R, weights, scaler, scale_columns, columns, gamma
        )

        plot_weights(weights, result_path / "weights", i)
        remaining_samples = np.count_nonzero(weights != 0)

        weighted_mmds_list.append(weighted_mmd)
        biases_list.append(sample_biases)
        wasserstein_parameter_list.append(wasserstein_distances)
        remaining_samples_list.append(remaining_samples)

        remaining_samples = np.count_nonzero(weights != 0)
        plot_weights(weights, result_path, i, "GBS")

    result_dict = {
        "MMDs": weighted_mmd,
        "Remaining Samples": remaining_samples,
        "Number of Samples": len(scaled_N),
    }
    for index, column in enumerate(columns):
        result_dict[f"{column}_relative_bias"] = {
            "bias": sample_biases[index],
            "wasserstein": wasserstein_distances[index],
        }

    # Statistical test
    lr_pvalue_gbs, lr_pvalue_weighted = logistic_regression(
        scaled_N[columns + ["Wahlteilnahme"]], weights
    )

    result_dict["logistig regression p values"] = {
        "GBS": lr_pvalue_gbs,
        "Weighted": lr_pvalue_weighted,
    }
    with open(result_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    plot_statistical_analysis(
        bins, scaled_N[columns], scaled_R[columns], result_path, weights, method
    )

def create_result_path(method):
    """Creates the result path and makes the directory.

    :param method: Method name
    :return: Result path
    """
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    result_path = result_path / method / "gbs_allensbach"
    result_path.mkdir(exist_ok=True, parents=True)
    return result_path
