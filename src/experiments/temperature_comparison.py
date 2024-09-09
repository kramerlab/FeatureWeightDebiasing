import json
from tqdm import trange
from sklearn.discriminant_analysis import StandardScaler

from utils.statistics import create_result_path
from utils.sampling import sample_with_test_set
from utils.metrics import (
    compute_classification_metrics_random_forest,
    compute_classification_metrics_svm,
)
from utils.visualization_fw_mrs import (
    plot_budget_comparison_auroc,
    plot_budget_comparison_auroc_mean,
    plot_feature_weights,
    visualize_boxplot,
)
import numpy as np

seed = 5


def feature_weight_downstream_comparison_experiment(
    df,
    columns,
    sample_weighting_method,
    target: str,
    number_of_repetitions: int = 50,
    bias_type: str = None,
    data_set_name: str = "",
    random_generator=None,
    method_name=None,
    drop=1,
    bias_fraction=0.25,
    validation_method="both",
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

    temperatures = [None, 0.1, 0.05, 0.01, 0.005, 0.001]
    C = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

    result_path = create_result_path(
        method_name,
        bias_type,
        data_set_name,
        experiment_name="drop_auroc_comparison",
        bias_fraction=bias_fraction,
    )
    result_path = result_path / validation_method
    result_path.mkdir(parents=True, exist_ok=True)

    auroc_path = result_path / "aurocs"
    auroc_path.mkdir(exist_ok=True, parents=True)

    boxplots_path = result_path / "boxplots"
    boxplots_path.mkdir(exist_ok=True, parents=True)

    dict_path = result_path / "dictionaries"
    dict_path.mkdir(exist_ok=True, parents=True)

    dropped_samples_list_dict = {}

    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    sample_df = df.copy()
    svm_auroc_dict = {}
    svm_auprc_dict = {}
    rf_auroc_dict = {}
    rf_auprc_dict = {}
    feature_weighted_aurocs_list = []

    if method_name in ("fw-mrs-temperature", "mrs-tree", "mrs-forest"):
        drop_samples = True
    else:
        drop_samples = False

    for temperature in temperatures:
        svm_auroc_dict[temperature] = []
        svm_auprc_dict[temperature] = []
        rf_auroc_dict[temperature] = []
        rf_auprc_dict[temperature] = []
        dropped_samples_list_dict[temperature] = []

    for i in trange(number_of_repetitions):
        if data_set_name in ("gbs_gesis", "gbs_allensbach"):
            N = sample_df[sample_df["label"] == 1]
            R = sample_df[sample_df["label"] == 0]
        else:
            N, R, _ = sample_with_test_set(
                bias_type,
                sample_df,
                target,
                train_fraction=0.5,
                bias_fraction=bias_fraction,
                test_fraction=0.0,
                columns=columns,
            )
        (
            random_forest_feature_weighted_aurocs,
            best_sample_weights_dict,
            dropped_samples_dict,
            best_feature_weights_dict,
            feature_importance_dict,
        ) = sample_weighting_method(
            N=N,
            R=R,
            columns=columns,
            save_path=result_path,
            bias_variable=target,
            drop=drop,
            random_generator=random_generator,
            target=target,
            budgets=temperatures,
            validation_method=validation_method,
            C=C,
            method_name=method_name,
            return_auroc=True,
        )
        feature_weighted_aurocs_list.append(random_forest_feature_weighted_aurocs)

        for temperature in temperatures:
            dropped_samples_list_dict[temperature].append(
                dropped_samples_dict[temperature]
            )
            sample_weights = best_sample_weights_dict[temperature]
            feature_weights = best_feature_weights_dict[temperature]

            (
                svm_auroc,
                svm_auprc,
                _,
                _,
                _,
            ) = compute_classification_metrics_svm(
                N,
                R,
                columns,
                sample_weights,
                np.array(feature_weights),
                target,
                random_state=seed,
                draw_with_feature_weights=True,
                n_splits=10,
                drop_samples=drop_samples,
                compute_feature_importance=False,
            )

            svm_auroc_dict[temperature].append(svm_auroc)
            svm_auprc_dict[temperature].append(svm_auprc)

            rf_auroc, rf_auprc, _, _, _, _ = (
                compute_classification_metrics_random_forest(
                    N,
                    R,
                    columns,
                    sample_weights,
                    feature_weights,
                    target,
                    random_state=seed,
                    draw_with_feature_weights=True,
                    splitter="feature_weighted_best",
                    n_estimators=500,
                    n_splits=10,
                    drop_samples=drop_samples,
                    compute_feature_importance=False,
                )
            )
            rf_auroc_dict[temperature].append(rf_auroc)
            rf_auprc_dict[temperature].append(rf_auprc)

        # Visualize individual run results
        number_of_samples = len(N)
        plot_budget_comparison_auroc(
            random_forest_feature_weighted_aurocs,
            number_of_samples,
            drop,
            auroc_path / f"iteration_{i}",
        )
        feature_weights_path = result_path / f"feature_weights" / str(i)
        feature_weights_path.mkdir(exist_ok=True, parents=True)
        plot_feature_weights(feature_weights, temperature, feature_weights_path)

        feature_importance_path = result_path / f"feature_importance" / str(i)
        feature_importance_path.mkdir(exist_ok=True, parents=True)

        # Visualize dropped samples and auroc comparison results
        for data_dict, y_label, file_name in zip(
            (
                dropped_samples_list_dict,
                svm_auroc_dict,
                svm_auprc_dict,
                rf_auroc_dict,
                rf_auprc_dict,
            ),
            ("Dropped Samples", "AUROC", "AUPRC", "AUROC", "AUPRC"),
            (
                "dropped_samples_comparison",
                "svm_auroc",
                "svm_auprc",
                "rf_auroc",
                "rf_auprc",
            ),
        ):
            visualize_boxplot(data_dict, y_label, file_name=boxplots_path / file_name)

    # Visualize mean results
    plot_budget_comparison_auroc_mean(
        feature_weighted_aurocs_list,
        number_of_samples,
        drop,
        result_path / "mean_auroc_comparison",
    )
    for data, file_name in zip(
        (
            dropped_samples_list_dict,
            svm_auroc_dict,
            svm_auprc_dict,
            rf_auroc_dict,
            rf_auprc_dict,
        ),
        (
            "dropped_samples_dict",
            "svm_auroc_dict",
            "svm_auprc_dict",
            "rf_auroc_dict",
            "rf_auprc_dict",
        ),
    ):
        with open(dict_path / f"{file_name}.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    save_mean_dropped_elements(result_path, dropped_samples_list_dict)


def save_mean_dropped_elements(result_path, dropped_samples_list):
    mean_dropped_samples_dict = {}
    for key, value in dropped_samples_list.items():
        dropped_elements = []
        dropped_elements.append(value)
        mean_dropped_samples_dict[f"{key} mean"] = np.mean(dropped_elements)
        mean_dropped_samples_dict[f"{key} std"] = np.std(dropped_elements)

    with open(result_path / "mean_dropped_samples", "w", encoding="utf-8") as file:
        json.dump(mean_dropped_samples_dict, file, indent=4)
