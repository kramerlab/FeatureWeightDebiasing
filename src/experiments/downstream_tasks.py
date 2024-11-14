import json
import numpy as np

from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_weights, save_weights
from utils.parameter import set_parameter
from utils.statistics import (
    create_result_path,
    write_result_dict,
    write_result_dict_test_set,
)
from utils.sampling import repeated_train_val_test_split, sample_N
from utils.metrics import (
    calculate_rbf_gamma,
    compute_classification_metrics_random_forest,
    compute_metrics,
)
from utils.visualization_fw_mrs import visualize_boxplot

seed = 5
sampling_random_generator = np.random.RandomState(seed)


def downstream_tasks_experiment(
    df,
    columns,
    sample_weighting_method,
    target: str,
    n_cv_repeats: int,
    n_cv_splits: int,
    bias_type: str = None,
    data_set_name: str = "",
    random_generator=None,
    load_previous_results=True,
    bias_fraction=0.1,
    drop=1,
    method_name=None,
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
    rf_auroc_list = []
    rf_auprc_list = []

    weighted_mmds_list = []
    biases_list = []
    wasserstein_distance_list = []

    abs_feature_importance_list = []
    feature_importance_list = []
    roc_curves_list = []
    best_temperature_list = []
    best_hyperparameter_list = []

    dropped_samples_list = []

    result_path = create_result_path(
        method_name,
        bias_type,
        data_set_name,
        experiment_name="downstream_task",
        bias_fraction=bias_fraction,
    )
    sample_weights_save_path = result_path / "sample_weights"
    feature_weights_save_path = result_path / "feature_weights"
    classificiation_result_path = result_path / "classification_results"
    validation_path = result_path / "validation"
    roc_path = result_path / "rocs"

    result_path.mkdir(exist_ok=True)
    classificiation_result_path.mkdir(exist_ok=True)
    sample_weights_save_path.mkdir(exist_ok=True)
    feature_weights_save_path.mkdir(exist_ok=True)
    roc_path.mkdir(exist_ok=True)
    validation_path.mkdir(exist_ok=True)

    sample_weight_list = load_weights(sample_weights_save_path)
    feature_weight_list = load_weights(feature_weights_save_path)

    scaler = StandardScaler()

    (
        draw_with_feature_weights,
        temperatures,
        dropped_samples_val_dict,
        auroc_val_dict,
        auprc_val_dict,
        hyperparameter_list,
    ) = set_parameter(method_name)


    for i, (N, R, T) in enumerate(
        repeated_train_val_test_split(
            n_cv_splits,
            n_cv_repeats,
            df,
            df[target],
            sampling_random_generator,
        )
    ):
        N[columns] = scaler.fit_transform(N[columns])
        R[columns] = scaler.transform(R[columns])
        T[columns] = scaler.transform(T[columns])
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

        gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))

        if len(sample_weight_list) > i and load_previous_results:
            sample_weights = sample_weight_list[i]
            feature_weights = feature_weight_list[i]

        else:
            sample_weights, feature_weights = sample_weighting_method(
                N=N,
                R=R,
                columns=columns,
                save_path=result_path,
                bias_variable=target,
                drop=drop,
                early_stopping=True,
                random_generator=random_generator,
                target=target,
                budgets=temperatures,
                hyperparameter_list=hyperparameter_list,
                method_name=method_name,
                compute_bias=False,
            )

            if method_name in ("mrs-forest", "psa"):
                sample_weights = {0.0: sample_weights}
                feature_weights = {0.0: feature_weights}

            feature_weight_list.append(feature_weights)
            sample_weight_list.append(sample_weights)

            save_weights(sample_weights_save_path, sample_weight_list)
            save_weights(feature_weights_save_path, feature_weight_list)

        (
            rf_auroc,
            rf_auprc,
            best_sample_weights,
            abs_feature_importance,
            roc_curve_values,
            best_temperature,
            best_hyperparameter,
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
            n_estimators=500,
            n_splits=5,
        )
        dropped_samples = np.count_nonzero(np.array(best_sample_weights) == 0.0)
        dropped_samples_list.append(dropped_samples)
        best_temperature_list.append(best_temperature)
        best_hyperparameter_list.append(best_hyperparameter)

        if method_name not in (
            "fw-mrs-temperature",
            "fw-mrs-temperature-svm",
        ):
            weighted_mmd, relative_bias, wasserstein_distances = compute_metrics(
                N,
                R,
                scaler,
                columns,
                best_sample_weights,
                gamma,
            )

            relative_bias = relative_bias.drop(["label"])
        else:
            weighted_mmd = np.ones(len(N.columns))
            relative_bias = np.ones(len(N.columns))
            wasserstein_distances = np.ones(len(N.columns))

        if method_name in (
            "fw-mrs-temperature",
            "fw-mrs-temperature-svm",
            "mrs-forest",
        ):
            for temperature, temperature_sample_weights in sample_weights.items():
                temperature_feature_weights = {"tmp": feature_weights[temperature]}
                temperature_sample_weights = {"tmp": temperature_sample_weights}

                (
                    rf_auroc_val,
                    rf_auprc_val,
                    best_sample_weights_val,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = compute_classification_metrics_random_forest(
                    N,
                    R,
                    columns,
                    temperature_sample_weights,
                    temperature_feature_weights,
                    target,
                    random_state=seed,
                    draw_with_feature_weights=draw_with_feature_weights,
                    n_estimators=500,
                    n_splits=5,
                    compute_feature_importance=False,
                )

                dropped_samples_val = np.count_nonzero(
                    np.array(best_sample_weights_val) == 0.0
                )
                dropped_samples_val_dict[float(temperature)].append(dropped_samples_val)
                auroc_val_dict[float(temperature)].append(rf_auroc_val)
                auprc_val_dict[float(temperature)].append(rf_auprc_val)

        weighted_mmds_list.append(weighted_mmd)
        biases_list.append(relative_bias)
        wasserstein_distance_list.append(wasserstein_distances)
            rf_auroc_list.append(rf_auroc)
        rf_auprc_list.append(rf_auprc)

        # plot_sample_weights(sample_weights, sample_weights_save_path, i)
        # plot_feature_weights(feature_weights, feature_weights_save_path, i)

        abs_feature_importance_list.append(abs_feature_importance.tolist())
        roc_curves_list.append(roc_curve_values)

        # plot_rocs_downstream(roc_curve_values, roc_path / f"roc_iteration_{i}")
    for result_list, file_name in zip(
        (
            rf_auroc_list,
            rf_auprc_list,
            dropped_samples_list,
            abs_feature_importance_list,
            feature_importance_list,
            roc_curves_list,
            best_temperature_list,
            best_hyperparameter_list,
        ),
        (
            "rf_auroc",
            "rf_auprc",
            "dropped_samples",
            "abs_feature_importance",
            "feature_importance",
            "roc_curves",
            "best_temperature",
            "best_hyperparameter",
        ),
    ):
        with open(
            classificiation_result_path / f"{file_name}.json", "w"
        ) as result_file:
            result_file.write(json.dumps(result_list))

    result_dict = write_result_dict(
        N.drop(["label"], axis="columns").columns,
        weighted_mmds_list,
        biases_list,
        wasserstein_distance_list,
    )

    with open(result_path / "similarity_results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    result_dict = {}
    result_dict = write_result_dict_test_set(
        rf_auroc_list,
        rf_auprc_list,
        dropped_samples_list,
        len(N),
    )

    with open(result_path / "classification_results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    if method_name in (
        "fw-mrs-temperature",
        "fw-mrs-temperature-svm",
        "mrs-forest",
    ):
        visualize_boxplot(auroc_val_dict, "AUROC", validation_path / "auroc_comparison")
        visualize_boxplot(auprc_val_dict, "AUPRC", validation_path / "auprc_comparison")
        visualize_boxplot(
            dropped_samples_val_dict,
            "Dropped Samples",
            validation_path / "dropped_samples_comparison",
        )

        for result_list, file_name in zip(
            (
                auroc_val_dict,
                auprc_val_dict,
                dropped_samples_val_dict,
            ),
            (
                "auroc_val_dict",
                "auprc_val_dict",
                "dropped_samples_val_dict",
            ),
        ):
            with open(validation_path / f"{file_name}.json", "w") as result_file:
                result_file.write(json.dumps(result_list))

    if method_name in (
        "fw-mrs-temperature",
        "mrs-forest",
        "fw-mrs-temperature-svm",
    ):
        dropped_samples_val_results_dict = {}
        for temperature in dropped_samples_val_dict.keys():
            dropped_samples_val_results_dict[f"{temperature}_mean"] = np.mean(
                dropped_samples_val_dict[temperature]
            )
            dropped_samples_val_results_dict[f"{temperature}_std"] = np.std(
                dropped_samples_val_dict[temperature]
            )

        with open(result_path / "dropped_elements.json", "w") as result_file:
            result_file.write(json.dumps(dropped_samples_val_results_dict))
