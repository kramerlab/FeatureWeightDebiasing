import json
import random
import numpy as np
import scipy.stats
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from utils.parameter import set_parameter
from utils.statistics import logistic_regression
from utils.metrics import (
    calculate_rbf_gamma,
    compute_metrics,
)

seed = 5
sampling_random_generator = np.random.RandomState(seed)


def perform_statistical_analysis_mrs(
    df,
    columns,
    sample_weighting_method,
    method_name,
    n_cv_repeats: int,
    n_cv_splits: int,
    random_generator=None,
    drop=1,
    target=None,
    data_set_name=None,
    **args,
):
    """Analyze GBS corrected with Allensbach with two methods.

    :param method_one: First method
    :param method_two: Second method
    """
    np.random.seed(seed)
    random.seed(seed)
    file_directory = Path(__file__).parent
    result_path = Path(
        file_directory,
        f"../../results/statistical_analysis_mrs/{data_set_name}/{method_name}",
    )
    iterations_path = result_path / "iteration"
    iterations_path.mkdir(exist_ok=True, parents=True)

    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])

    N = df[df["label"] == 1].copy()
    R = df[df["label"] == 0].copy()
    gamma = calculate_rbf_gamma(N[columns])
    sample_weights_list = []
    dropped_samples_list = []
    wasserstein_list = []
    relative_biases_list = []
    mmd_list = []
    pvalue_list = []
    roc_curves_list = []

    (
        _,
        temperatures,
        _,
        _,
        _,
        _,
        hyperparameter_list,
    ) = set_parameter(method_name)

    for i in range(n_cv_repeats):
        sample_weights, _ = sample_weighting_method(
            N=N,
            R=R,
            columns=columns,
            drop=drop,
            early_stopping=True,
            random_generator=random_generator,
            budgets=temperatures,
            hyperparameter_list=hyperparameter_list,
            target=target,
        )
        if method_name == "mrs-forest":
            sample_weights = {0.0: sample_weights}

        dropped_samples = np.count_nonzero(np.array(sample_weights) == 0.0)
        dropped_samples_list.append(dropped_samples)

        weighted_mmd, relative_bias, wasserstein_distances, best_sample_weights = (
            compute_metrics(
                N,
                R,
                scaler,
                columns,
                target,
                sample_weights,
                gamma,
                return_sample_weights=True,
            )
        )

        sample_weights_list.append(best_sample_weights)
        wasserstein_list.append(wasserstein_distances)
        relative_biases_list.append(relative_bias)
        mmd_list.append(weighted_mmd)

        pvalue = logistic_regression(
            N[columns + [target]], best_sample_weights
        )
        pvalue_list.append(pvalue)

    result_dict_mrs_iteration = {}
    for index, column in enumerate(columns):
        result_dict_mrs_iteration[f"{column}_relative_bias"] = {
            "wasserstein": wasserstein_distances[index],
            "relative_bias": relative_bias[index],
        }

    wasserstein_std_list = np.std(wasserstein_list, axis=0)
    relative_bias_std_list = np.std(relative_biases_list, axis=0)
    mmd_std_list = np.std(np.array(mmd_list)[:, np.newaxis], axis=0)

    wasserstein_mean_list = np.mean(wasserstein_list, axis=0)
    relative_bias_mean_list = np.mean(relative_biases_list, axis=0)
    mmd_mean_list = np.mean(np.array(mmd_list)[:, np.newaxis], axis=0)

    pvalue_confidence_list = compute_confidence_interval(
        np.array(pvalue_list)[:, np.newaxis]
    )

    p_values_dict = {
        "logistic regression p values confidence interval": pvalue_confidence_list.tolist(),
        "pvalues": pvalue_list,
    }

    # Save methods mean results
    result_dict_similarity = {}
    for index, column in enumerate(columns):
        result_dict_similarity["MMD Mean"] = mmd_mean_list.tolist()
        result_dict_similarity["MMD Std"] = mmd_std_list.tolist()
        result_dict_similarity[f"{column}_bias"] = {
            "wasserstein mean": wasserstein_mean_list[index].tolist(),
            "wasserstein_std": wasserstein_std_list[index].tolist(),
            "relative_bias mean": relative_bias_mean_list[index].tolist(),
            "relative_bias_std": relative_bias_std_list[index].tolist(),
        }

    for file_name, data in zip(
        (
            "similarity_metrics.json",
            "p_value_results.json",
            "dropped_samples.json",
            "roc_curves.json",
            "sample_weights.json",
        ),
        (
            result_dict_similarity,
            p_values_dict,
            dropped_samples_list,
            roc_curves_list,
            sample_weights_list,
        ),
    ):
        with open(result_path / file_name, "w") as result_file:
            result_file.write(json.dumps(data))


def compute_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    confidence_inveral_list = []
    for i in range(data.shape[1]):
        x = data[:, i]
        lower_bound, upper_bound = scipy.stats.t.interval(
            confidence=confidence,
            df=len(x) - 1,
            loc=np.mean(x),
            scale=scipy.stats.sem(x),
        )
        mean = np.mean(x)
        confidence_inveral_list.append([lower_bound, mean, upper_bound])

    return np.stack(confidence_inveral_list, axis=0)
