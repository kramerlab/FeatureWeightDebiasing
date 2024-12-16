import json

from experiments.downstream_tasks import load_saved_results
from utils.data_loader import save_results
from utils.statistics import create_result_path
from utils.sampling import sample_N, repeated_train_val_test_split
from utils.metrics import compute_classification_metrics_random_forest
from utils.visualization_fw_mrs import (
    plot_budget_comparison_auroc,
    plot_temperature_comparison_auroc_mean,
)
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    if method_name in ("fw-mrs-temperature",):
        temperatures = [0.0, 0.1, 0.05, 0.01, 0.005]
        hyperparameter_list = [0.025, 0.01, 0.0]
        fixed_hyperparameter = 0.01
    if method_name == "fw-mrs-temperature-svm":
        temperatures = [0.0, 0.1, 0.05, 0.01, 0.005]
        hyperparameter_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
        fixed_hyperparameter = 1e-1
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
    feature_importances_save_path = result_path / "feature_importances"
    auroc_save_path = result_path / "method_aurocs"

    sample_weights_save_path.mkdir(exist_ok=True)
    feature_weights_save_path.mkdir(exist_ok=True)
    auroc_save_path.mkdir(exist_ok=True)
    feature_importances_save_path.mkdir(exist_ok=True)

    feature_weighted_aurocs_list = load_saved_results(
        auroc_save_path, file_name="optimised_method_aurocs"
    )
    fixed_feature_weighted_aurocs_list = load_saved_results(
        auroc_save_path, file_name="fixed_method_aurocs"
    )
    sample_weights_list = load_saved_results(sample_weights_save_path)
    feature_weights_list = load_saved_results(feature_weights_save_path)
    feature_importances_list = load_saved_results(
        feature_weights_save_path, "feature_importances"
    )

    number_of_samples_list = []
    scaler = StandardScaler()

    if data_set_name in ("gbs_gesis", "gbs_allensbach"):
        N = df[df["label"] == 1]
        R = df[df["label"] == 0]
        split_method = gbs_split
    else:
        split_method = repeated_train_val_test_split

    for i, (N, R, _) in enumerate(
        split_method(
            n_cv_splits,
            n_cv_repeats,
            df,
            df[target],
            sampling_random_generator,
        )
    ):
        N[columns] = scaler.fit_transform(N[columns])
        R[columns] = scaler.transform(R[columns])
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
            (
                random_forest_feature_weighted_aurocs,
                sample_weights,
                feature_weights,
                feature_importances,
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
                hyperparameter_list=hyperparameter_list,
                method_name=method_name,
                return_metrics=True,
            )
            if data_set_name not in ("gbs_gesis", "gbs_allensbach"):
                optimised_random_forest_feature_weighted_aurocs = {}
                fixed_feature_weighted_aurocs = {}
                for temperature, temperature_sample_weights in sample_weights.items():
                    temperature_feature_weights = {"tmp": feature_weights[temperature]}
                    temperature_sample_weights = {"tmp": temperature_sample_weights}

                    (
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        best_hyperparameter,
                        _,
                    ) = compute_classification_metrics_random_forest(
                        N,
                        R,
                        columns,
                        temperature_sample_weights,
                        temperature_feature_weights,
                        target,
                        random_state=seed,
                        draw_with_feature_weights=True,
                        splitter="feature_weighted_best",
                        n_estimators=500,
                        n_splits=5,
                        compute_feature_importance=False,
                    )

                    optimised_random_forest_feature_weighted_aurocs[temperature] = (
                        random_forest_feature_weighted_aurocs[temperature][
                            best_hyperparameter
                        ]
                    )
                    fixed_feature_weighted_aurocs[temperature] = (
                        random_forest_feature_weighted_aurocs[temperature][
                            fixed_hyperparameter
                        ]
                    )

            feature_weights_list.append(feature_weights)
            feature_importances_list.append(feature_importances)
            sample_weights_list.append(sample_weights)
            feature_weighted_aurocs_list.append(
                optimised_random_forest_feature_weighted_aurocs
            )
            fixed_feature_weighted_aurocs_list.append(fixed_feature_weighted_aurocs)

            save_results(sample_weights_save_path, sample_weights_list)
            save_results(feature_weights_save_path, feature_weights_list)
            save_results(
                feature_importances_save_path,
                feature_importances_list,
                "feature_importances",
            )
            save_results(
                auroc_save_path,
                feature_weighted_aurocs_list,
                file_name="optimised_method_aurocs",
            )
            save_results(
                auroc_save_path,
                fixed_feature_weighted_aurocs_list,
                file_name="fixed_method_aurocs",
            )

        for temperature, values in sample_weights.items():
            key = next(iter(values.keys()))
            dropped_samples_dict[float(temperature)].append(
                np.count_nonzero(np.array(values[key]) == 0.0)
            )
        number_of_samples_list.append(len(N))

        plot_temperature_comparison_auroc_mean(
            feature_weighted_aurocs_list,
            number_of_samples_list,
            drop,
            result_path / "optimised_mean_auroc",
        )
        plot_temperature_comparison_auroc_mean(
            fixed_feature_weighted_aurocs_list,
            number_of_samples_list,
            drop,
            result_path / "fixed_mean_auroc",
        )


    for temperature, values in dropped_samples_dict.items():
        dropped_samples_dict[temperature] = np.mean(dropped_samples_dict[temperature])

    with open(result_path / "dropped_elements.json", "w") as result_file:
        result_file.write(json.dumps(dropped_samples_dict))

    plot_temperature_comparison_auroc_mean(
        feature_weighted_aurocs_list,
        number_of_samples_list,
        drop,
        result_path / "optimised_mean_auroc",
    )

    plot_temperature_comparison_auroc_mean(
            fixed_feature_weighted_aurocs_list,
            number_of_samples_list,
            drop,
            result_path / "fixed_mean_auroc",
        )
    meta_data_dict = {"n_dropped": drop, "number_of_samples": number_of_samples_list}
    with open(result_path / 'metadata.json', 'w') as file:
        json.dump(meta_data_dict, file)


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


def gbs_split(n_cv_splits, n_cv_repeats, df, target_values, random_generator):
    for _ in range(n_cv_splits):
        for _ in range(n_cv_repeats):
            N = df[df["label"] == 1]
            R = df[df["label"] == 0]
            yield N, R, _
