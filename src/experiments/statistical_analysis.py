import json
import random
import numpy as np
import scipy.stats
from pathlib import Path
from sklearn.discriminant_analysis import StandardScaler
from tqdm import trange

from utils.statistics import logistic_regression
from utils.visualization import plot_statistical_analysis
from utils.metrics import (
    calculate_rbf_gamma,
    compute_classification_metrics_random_forest_gbs,
    compute_metrics,
)

bins = 25
seed = 5
# Used to draw radom states
max_int = 2**32 - 1


def perform_statistical_analysis(
    df,
    columns,
    sample_weighting_method,
    method_name,
    target: str,
    number_of_repetitions=1000,
    drop=1,
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
        f"../../results/statistical_analysis/{data_set_name}/{method_name}",
    )
    iterations_path = result_path / "iteration"
    iterations_path.mkdir(exist_ok=True, parents=True)

    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    scaled_df = df.copy()
    first_random_generator = np.random.RandomState(seed)

    gamma = calculate_rbf_gamma(scaled_df[columns])
    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]
    sample_weights_list = []
    feature_weights_list = []
    rf_auroc_list = []
    rf_auprc_list = []
    dropped_samples_list = []
    wasserstein_list = []
    relative_biases_list = []
    mmd_list = []
    pvalue_list = []
    abs_feature_importance_list = []
    feature_importance_list = []
    roc_curves_list = []

    for i in trange(number_of_repetitions):
        if method_name == "fw-mrs-temperature":
            splitter = "feature_weighted_best"
            draw_with_feature_weights = True
            temperatures = [None, 0.1, 0.05, 0.01, 0.005]
        else:
            splitter = "best"
            draw_with_feature_weights = False
            temperatures = None

        sample_weights, feature_weights, _ = sample_weighting_method(
            scaled_N,
            scaled_R,
            columns,
            drop=drop,
            early_stopping=True,
            random_generator=first_random_generator,
            budgets=temperatures,
        )

        if feature_weights is None:
            feature_weights = (np.ones(len(columns)) / len(columns)).tolist()

        (
            rf_auroc,
            rf_auprc,
            sample_weights,
            abs_feature_importance,
            feature_importance,
            roc_curve_values,
        ) = compute_classification_metrics_random_forest_gbs(
            scaled_N,
            scaled_N,
            scaled_N,
            columns,
            sample_weights,
            feature_weights,
            target,
            random_state=seed,
            draw_with_feature_weights=draw_with_feature_weights,
            splitter=splitter,
            n_estimators=500,
            n_splits=10,
        )

        sample_weights_list.append(sample_weights)
        feature_weights_list.append(feature_weights)

        rf_auroc_list.append(rf_auroc)
        rf_auprc_list.append(rf_auprc)
        abs_feature_importance_list.append(abs_feature_importance.tolist())
        feature_importance_list.append(feature_importance.tolist())
        roc_curves_list.append(roc_curve_values)

        dropped_samples = np.count_nonzero(np.array(sample_weights) == 0.0)
        dropped_samples_list.append(dropped_samples)

        (
            mmd,
            relative_biases,
            wasserstein_distances_mrs,
        ) = compute_metrics(
            scaled_N[columns].copy(),
            scaled_R[columns].copy(),
            scaler,
            columns,
            columns,
            sample_weights,
            gamma,
        )
        wasserstein_list.append(wasserstein_distances_mrs)
        relative_biases_list.append(relative_biases)
        mmd_list.append(mmd)

        if data_set_name == "gbs_allensbach":
            pvalue = logistic_regression(
                scaled_N[columns + ["Wahlteilnahme"]], sample_weights
            )
            pvalue_list.append(pvalue)

        result_dict_mrs_iteration = {}
        for index, column in enumerate(columns):
            result_dict_mrs_iteration[f"{column}_relative_bias"] = {
                "wasserstein": wasserstein_distances_mrs[index],
                "relative_bias": relative_biases[index],
            }

        with open(
            iterations_path / f"results_{method_name}_{i}.json", "w"
        ) as result_file:
            result_file.write(json.dumps(result_dict_mrs_iteration))

    wasserstein_confidence_list = compute_confidence_interval(wasserstein_list)
    relative_bias_confidence_list = compute_confidence_interval(relative_biases_list)
    mmd_confidence_list = compute_confidence_interval(np.array(mmd_list)[:, np.newaxis])
    pvalue_confidence_list = compute_confidence_interval(
        np.array(pvalue_list)[:, np.newaxis]
    )

    p_values_dict = {
        "logistig regression p values": pvalue_confidence_list.tolist(),
        "pvalues": pvalue_list,
    }

    # Save methods mean results
    result_dict_similarity = {}
    for index, column in enumerate(columns):
        result_dict_similarity["MMD"] = mmd_confidence_list.tolist()
        result_dict_similarity[f"{column}_bias"] = {
            "wasserstein": wasserstein_confidence_list[index].tolist(),
            "relative_bias": relative_bias_confidence_list[index].tolist(),
        }

    for file_name, data in zip(
        (
            "similarity_metrics.json",
            "p_value_results.json",
            "dropped_elements.json",
            "rf_auroc.json",
            "rf_auprc.json",
            "dropped_samples.json",
            "abs_feature_importance.json",
            "feature_importance.json",
            "roc_curves.json",
        ),
        (
            result_dict_similarity,
            p_values_dict,
            dropped_samples_list,
            rf_auroc_list,
            rf_auprc_list,
            dropped_samples_list,
            abs_feature_importance_list,
            feature_importance_list,
            roc_curves_list,
        ),
    ):
        with open(result_path / file_name, "w") as result_file:
            result_file.write(json.dumps(data))

    scaled_N.loc[:, columns] = scaler.inverse_transform(scaled_N[columns])
    scaled_R.loc[:, columns] = scaler.inverse_transform(scaled_R[columns])

    # Plot exemplary a random iteration
    random_iteration = np.random.randint(0, len(sample_weights_list))
    plot_statistical_analysis(
        bins,
        scaled_N[columns],
        scaled_R[columns],
        result_path,
        sample_weights_list[random_iteration],
        method_name,
    )


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
