import json
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import trange
from sklearn.discriminant_analysis import StandardScaler

from utils.statistics import create_result_path
from utils.sampling import sample_N
from utils.metrics import (
    compute_classification_metrics_random_forest,
    compute_classification_metrics_svm,
    scale_df,
)
from utils.visualization_fw_mrs import (
    plot_budget_comparison_auroc,
    plot_budget_comparison_auroc_mean,
    plot_feature_weights,
    visualize_boxplot,
    visualize_heatmap,
)
import numpy as np

seed = 5


def temperature_comparison(
    df,
    columns,
    sample_weighting_method,
    target: str,
    n_cv_splits: int = 2,
    n_cv_repeats=10,
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

    temperatures = [0.0, 0.1, 0.05, 0.01, 0.005]

    if method_name == "fw-mrs-svm-downstream":
        hyperparameter_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    else:
        hyperparameter_list = [0.05, 0.025, 0.0]

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
    svm_auroc_dict = {}
    svm_auprc_dict = {}
    rf_auroc_dict = {}
    rf_auprc_dict = {}
    feature_weighted_aurocs_list = []

    for temperature in temperatures:
        svm_auroc_dict[temperature] = {}
        svm_auprc_dict[temperature] = {}
        rf_auroc_dict[temperature] = {}
        rf_auprc_dict[temperature] = {}
        dropped_samples_list_dict[temperature] = {}
        for hyperparameter in hyperparameter_list:
            svm_auroc_dict[temperature][hyperparameter] = []
            svm_auprc_dict[temperature][hyperparameter] = []
            rf_auroc_dict[temperature][hyperparameter] = []
            rf_auprc_dict[temperature][hyperparameter] = []
            dropped_samples_list_dict[temperature][hyperparameter] = []

    sample_df, _ = scale_df(df, columns)

    if data_set_name in ("gbs_gesis", "gbs_allensbach"):
        N = sample_df[sample_df["label"] == 1]
        R = sample_df[sample_df["label"] == 0]

    skf = RepeatedStratifiedKFold(
        n_splits=n_cv_splits, n_repeats=n_cv_repeats, random_state=seed
    )
    for i, (train_indices, test_indices) in enumerate(
        skf.split(sample_df, sample_df[target])
    ):
        if data_set_name not in ("gbs_gesis", "gbs_allensbach"):
            N = sample_df.iloc[train_indices].copy()
            N = sample_N(
                train=N,
                bias_type=bias_type,
                bias_fraction=bias_fraction,
                columns=columns,
                bias_variable=target,
            )
            R = sample_df.iloc[test_indices].copy()
            N["label"] = 1
            R["label"] = 0
        (
            random_forest_feature_weighted_aurocs,
            best_sample_weights_dict,
            dropped_samples_dict,
            best_feature_weights_dict,
            feature_importance_dict,
        ) = sample_weighting_method(
            N=N,
            R=R,
            target=target,
            columns=columns,
            save_path=result_path,
            bias_variable=target,
            drop=drop,
            random_generator=random_generator,
            budgets=temperatures,
            validation_method=validation_method,
            hyperparameter_list=hyperparameter_list,
            method_name=method_name,
            return_auroc=True,
        )
        feature_weighted_aurocs_list.append(random_forest_feature_weighted_aurocs)

        for temperature in temperatures:
            for hyperparameter in hyperparameter_list:
                dropped_samples_list_dict[temperature][hyperparameter].append(
                    dropped_samples_dict[temperature][hyperparameter]
                )
                sample_weights = best_sample_weights_dict[temperature][hyperparameter]
                feature_weights = best_feature_weights_dict[temperature][hyperparameter]

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
                    compute_feature_importance=False,
                )

                svm_auroc_dict[temperature][hyperparameter].append(svm_auroc)
                svm_auprc_dict[temperature][hyperparameter].append(svm_auprc)

                rf_auroc, rf_auprc, _, _, _, _, _ = (
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
                        compute_feature_importance=False,
                    )
                )
                rf_auroc_dict[temperature][hyperparameter].append(rf_auroc)
                rf_auprc_dict[temperature][hyperparameter].append(rf_auprc)

            # Visualize individual run results
            number_of_samples = len(N)
            plot_budget_comparison_auroc(
                random_forest_feature_weighted_aurocs[temperature],
                number_of_samples,
                drop,
                auroc_path / f"{temperature}_iteration_{i}",
            )
        # feature_weights_path = result_path / f"feature_weights" / str(i)
        # feature_weights_path.mkdir(exist_ok=True, parents=True)
        # plot_feature_weights(feature_weights, temperature, feature_weights_path)
        # feature_importance_path = result_path / f"feature_importance" / str(i)
        # feature_importance_path.mkdir(exist_ok=True, parents=True)

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
            # visualize_boxplot(data_dict, y_label, file_name=boxplots_path / file_name)
            visualize_heatmap(data_dict, y_label, file_name=boxplots_path / file_name)

    # Visualize mean results
    # plot_budget_comparison_auroc_mean(
    #   feature_weighted_aurocs_list,
    #   number_of_samples,
    #   drop,
    #   result_path / "mean_auroc_comparison",
    # )
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
