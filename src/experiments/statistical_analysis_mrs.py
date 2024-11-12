import json
import random
import numpy as np
import scipy.stats
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from utils.parameter import set_parameter
from utils.sampling import repeated_train_val_test_split
from utils.statistics import logistic_regression
from utils.visualization import plot_statistical_analysis
from utils.metrics import (
    calculate_rbf_gamma,
    compute_classification_metrics_random_forest,
    compute_metrics,
)
import pandas as pd

bins = 25
seed = 5
sampling_random_generator = np.random.RandomState(seed)


def perform_statistical_analysis_mrs(
    df,
    columns,
    sample_weighting_method,
    method_name,
    n_cv_repeats: int,
    n_cv_splits: int,
    target: str,
    random_generator=None,
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
        f"../../results/statistical_analysis_mrs/{data_set_name}/{method_name}",
    )
    iterations_path = result_path / "iteration"
    iterations_path.mkdir(exist_ok=True, parents=True)

    gbs = df[df["label"] == 1].copy()
    allensbach = df[df["label"] == 0].copy()
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

    (
        splitter,
        draw_with_feature_weights,
        temperatures,
        _,
        _,
        _,
        hyperparameter_list,
    ) = set_parameter(method_name)

    scaler = StandardScaler()

    for i, (N, R, T) in enumerate(
        repeated_train_val_test_split(
            n_cv_splits,
            n_cv_repeats,
            gbs,
            gbs[target],
            sampling_random_generator,
        )
    ):

        N = pd.concat([N, R])
        N[columns] = scaler.fit_transform(N[columns])
        allensbach[columns] = scaler.transform(allensbach[columns])
        T[columns] = scaler.transform(T[columns])
        gamma = calculate_rbf_gamma(N[columns])
        sample_weights, feature_weights = sample_weighting_method(
            N=N,
            R=allensbach,
            columns=columns,
            drop=drop,
            early_stopping=True,
            random_generator=random_generator,
            budgets=temperatures,
            hyperparameter_list=hyperparameter_list,
            target=target,
        )
        if method_name in ("mrs-forest", "psa"):
            sample_weights = {0.0: sample_weights}
            feature_weights = {0.0: feature_weights}

        if feature_weights is None:
            feature_weights = (np.ones(len(columns)) / len(columns)).tolist()

        (
            rf_auroc,
            rf_auprc,
            sample_weights,
            abs_feature_importance,
            roc_curve_values,
            _,
            _,
            _,
        ) = compute_classification_metrics_random_forest(
            N,
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

        sample_weights_list.append(sample_weights)
        feature_weights_list.append(feature_weights)

        rf_auroc_list.append(rf_auroc)
        rf_auprc_list.append(rf_auprc)
        abs_feature_importance_list.append(abs_feature_importance.tolist())
        roc_curves_list.append(roc_curve_values)

        dropped_samples = np.count_nonzero(np.array(sample_weights) == 0.0)
        dropped_samples_list.append(dropped_samples)

        if method_name not in (
            "fw-mrs-temperature",
        ):
            weighted_mmd, relative_bias, wasserstein_distances = compute_metrics(
                N,
                R,
                scaler,
                columns,
                sample_weights,
                gamma,
            )
            relative_bias = relative_bias.drop(["label"])
        else:
            weighted_mmd = np.ones(len(N.columns))
            relative_bias = np.ones(len(N.columns))
            wasserstein_distances = np.ones(len(N.columns))
        wasserstein_list.append(wasserstein_distances)
        relative_biases_list.append(relative_bias)
        mmd_list.append(weighted_mmd)

        if data_set_name == "gbs_allensbach":
            pvalue = logistic_regression(N[columns + ["Wahlteilnahme"]], sample_weights)
            pvalue_list.append(pvalue)

        result_dict_mrs_iteration = {}
        for index, column in enumerate(columns):
            result_dict_mrs_iteration[f"{column}_relative_bias"] = {
                "wasserstein": wasserstein_distances[index],
                "relative_bias": relative_bias[index],
            }

        with open(iterations_path / f"results_{method_name}_{i}.json", "w") as result_file:
            result_file.write(json.dumps(result_dict_mrs_iteration))

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
        result_dict_similarity["MMD Mean"] = mmd_std_list.tolist()
        result_dict_similarity["MMD Std"] = mmd_mean_list.tolist()
        result_dict_similarity[f"{column}_bias"] = {
            "wasserstein mean": wasserstein_mean_list[index].tolist(),
            "relative_bias mean": relative_bias_mean_list[index].tolist(),
            "wasserstein_std": wasserstein_std_list[index].tolist(),
            "relative_bias_std": relative_bias_std_list[index].tolist(),
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

    N.loc[:, columns] = scaler.inverse_transform(N[columns])
    R.loc[:, columns] = scaler.inverse_transform(R[columns])

    # Plot exemplary last iteration
    plot_statistical_analysis(
        bins,
        N[columns],
        R[columns],
        result_path,
        sample_weights_list[-1],
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
