import json
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_saved_results, save_results
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
    compute_classification_metrics_random_forest_perfect,
    compute_domain_classifier_auroc,
    compute_metrics,
    compute_pad,
)
import copy

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
    pad_list = []
    domain_auroc_list = []

    weighted_mmds_list = []
    biases_list = []
    wasserstein_distance_list = []

    abs_feature_importance_list = []
    feature_importance_list = []
    roc_curves_list = []
    best_temperature_list = []
    best_hyperparameter_list = []

    dropped_samples_list = []
    R_auroc_list = []
    R_auprc_list = []
    R_pad_list = []
    R_domain_auroc_list = []
    validation_score_list = []

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

    sample_weight_list = load_saved_results(sample_weights_save_path)
    feature_weight_list = load_saved_results(feature_weights_save_path)

    scaler = StandardScaler()

    if data_set_name == "gbs_gesis":
        split_method = gbs_gesis_split
    elif data_set_name == "gbs_allensbach":
        split_method = gbs_allensbach_split
    else:
        split_method = repeated_train_val_test_split
    (
        draw_with_feature_weights,
        temperatures,
        dropped_samples_val_dict,
        auroc_val_dict,
        auprc_val_dict,
        accuracy_val_dict,
        hyperparameter_list,
    ) = set_parameter(method_name, bias_type)

    if dropped_samples_val_dict is not None:
        dropped_samples_individual_val_dict = copy.deepcopy(dropped_samples_val_dict)
        auroc_individual_val_dict = copy.deepcopy(auroc_val_dict)
        auprc_individual_val_dict = copy.deepcopy(auprc_val_dict)
        accuracy_individual_val_dict = copy.deepcopy(accuracy_val_dict)
    else:
        dropped_samples_individual_val_dict = None
        auroc_individual_val_dict = None
        auprc_individual_val_dict = None
        accuracy_individual_val_dict = None
    for i, (N, R, T) in enumerate(
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
        T[columns] = scaler.transform(T[columns])

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

            save_results(sample_weights_save_path, sample_weight_list)
            save_results(feature_weights_save_path, feature_weight_list)

        if not method_name == "fw-mrs-temperature-comparison":
            (
                rf_auroc,
                rf_auprc,
                _,
                best_sample_weights,
                best_feature_weights,
                abs_feature_importance,
                roc_curve_values,
                best_temperature,
                best_hyperparameter,
                _,
                validation_score,
            ) = compute_classification_metrics_random_forest(
                N,
                R,
                T,
                columns,
                sample_weights,
                feature_weights,
                target,
                random_state=seed,
                draw_with_feature_weights=draw_with_feature_weights,
                n_estimators=500,
                n_splits=10,
            )

            pad = compute_pad(
                N,
                T,
                columns,
                best_sample_weights,
                best_feature_weights,
                seed=seed,
            )

            domain_auroc = compute_domain_classifier_auroc(
                N,
                T,
                columns,
                best_sample_weights,
                best_feature_weights,
                seed=seed,
            )

            pad_list.append(pad)
            domain_auroc_list.append(domain_auroc)
            dropped_samples = np.count_nonzero(np.array(best_sample_weights) == 0.0)
            dropped_samples_list.append(dropped_samples)
            best_temperature_list.append(best_temperature)
            best_hyperparameter_list.append(best_hyperparameter)
            validation_score_list.append(validation_score)
            rf_auroc_list.append(rf_auroc)
            rf_auprc_list.append(rf_auprc)
            abs_feature_importance_list.append(abs_feature_importance.tolist())
            roc_curves_list.append(roc_curve_values)

        if method_name not in (
            "fw-mrs-temperature",
            "fw-mrs-temperature-svm",
            "fw-mrs-temperature-comparison",
        ):
            weighted_mmd, relative_bias, wasserstein_distances = compute_metrics(
                N,
                R,
                scaler,
                columns,
                target,
                best_sample_weights,
                gamma,
            )

        else:
            weighted_mmd = np.ones(len(N.columns))
            relative_bias = np.ones(len(N.columns))
            wasserstein_distances = np.ones(len(N.columns))

        if method_name in (
            "fw-mrs-temperature",
            "fw-mrs-temperature-svm",
            "fw-mrs-temperature-comparison",
            "mrs-forest",
        ) and data_set_name not in ("gbs_gesis", "gbs_allensbach"):
            compute_validation_results(
                columns,
                target,
                draw_with_feature_weights,
                dropped_samples_val_dict,
                auroc_val_dict,
                auprc_val_dict,
                accuracy_val_dict,
                dropped_samples_individual_val_dict,
                auroc_individual_val_dict,
                auprc_individual_val_dict,
                accuracy_individual_val_dict,
                N,
                R,
                sample_weights,
                feature_weights,
                method_name,
            )

        weighted_mmds_list.append(weighted_mmd)
        biases_list.append(relative_bias.astype(float))
        wasserstein_distance_list.append(wasserstein_distances)

        if method_name == "uniform" and bias_type in ("less_positive_class", "none"):
            (R_auroc, R_auprc) = compute_classification_metrics_random_forest_perfect(
                R,
                T,
                columns,
                target,
                random_state=seed,
                n_estimators=500,
                n_splits=10,
            )

            R_pad = compute_pad(
                R,
                T,
                columns,
                np.ones(len(R)),
                best_feature_weights,
                seed=seed,
                sample=True,
            )

            R_domain_auroc = compute_domain_classifier_auroc(
                R,
                T,
                columns,
                np.ones(len(R)),
                best_feature_weights,
                seed=seed,
            )

            R_domain_auroc_list.append(R_domain_auroc)
            R_auroc_list.append(R_auroc)
            R_auprc_list.append(R_auprc)
            R_pad_list.append(R_pad)

        for result_list, file_name in zip(
            (
                rf_auroc_list,
                rf_auprc_list,
                pad_list,
                domain_auroc_list,
                dropped_samples_list,
                abs_feature_importance_list,
                feature_importance_list,
                roc_curves_list,
                best_temperature_list,
                best_hyperparameter_list,
                R_auroc_list,
                R_auprc_list,
                R_pad_list,
                R_domain_auroc_list,
                validation_score_list,
            ),
            (
                "rf_auroc_list",
                "rf_auprc_list",
                "pad_list",
                "domain_auroc_list",
                "dropped_samples",
                "abs_feature_importance",
                "feature_importance",
                "roc_curves",
                "best_temperature",
                "best_hyperparameter",
                "R_auroc_list",
                "R_auprc_list",
                "R_pad_list",
                "R_domain_auroc_list",
                "validation_score",
            ),
        ):
            with open(
                classificiation_result_path / f"{file_name}.json", "w"
            ) as result_file:
                result_file.write(json.dumps(result_list))

        if data_set_name in ("gbs_gesis", "gbs_allensbach"):
            result_columns = columns
        else:
            result_columns = N.drop(["label"], axis="columns").columns
        result_dict = write_result_dict(
            result_columns,
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
            pad_list,
            domain_auroc_list,
            dropped_samples_list,
            len(N),
        )

        with open(result_path / "classification_results.json", "w") as result_file:
            result_file.write(json.dumps(result_dict))

        if method_name in (
            "fw-mrs-temperature",
            "fw-mrs-temperature-comparison",
            "fw-mrs-temperature-svm",
            "mrs-forest",
        ):
            for result_list, file_name in zip(
                (
                    auroc_val_dict,
                    auprc_val_dict,
                    dropped_samples_val_dict,
                    dropped_samples_individual_val_dict,
                    auroc_individual_val_dict,
                    auprc_individual_val_dict,
                    accuracy_individual_val_dict,
                ),
                (
                    "auroc_val_dict",
                    "auprc_val_dict",
                    "dropped_samples_val_dict",
                    "dropped_samples_individual_val_dict",
                    "auroc_individual_val_dict",
                    "auprc_individual_val_dict",
                    "accuracy_individual_val_dict",
                ),
            ):
                with open(validation_path / f"{file_name}.json", "w") as result_file:
                    result_file.write(json.dumps(result_list))

            dropped_samples_val_results_dict = {}
            if method_name in ("mrs-forest"):
                for temperature, temperature_values in dropped_samples_val_dict.items():
                    for hyperparameter, values in temperature_values.items():
                        dropped_samples_val_results_dict[
                            f"{temperature}_{hyperparameter}_mean"
                        ] = np.mean(values)
                        dropped_samples_val_results_dict[
                            f"{temperature}_{hyperparameter}_std"
                        ] = np.std(values)
            else:
                for temperature, temperature_values in dropped_samples_val_dict.items():
                    dropped_samples_val_results_dict[f"{temperature}_mean"] = np.mean(
                        temperature_values
                    )
                    dropped_samples_val_results_dict[f"{temperature}_std"] = np.std(
                        temperature_values
                    )

            with open(validation_path / "dropped_elements.json", "w") as result_file:
                result_file.write(json.dumps(dropped_samples_val_results_dict))


def compute_validation_results(
    columns,
    target,
    draw_with_feature_weights,
    dropped_samples_val_dict,
    auroc_val_dict,
    auprc_val_dict,
    accuracy_val_dict,
    dropped_samples_individual_val_dict,
    auroc_individual_val_dict,
    auprc_individual_val_dict,
    accuracy_individual_val_dict,
    N,
    R,
    sample_weights,
    feature_weights,
    method_name,
):
    if method_name in ("mrs-forest"):
        for temperature, temperature_sample_weights in sample_weights.items():
            for (
                hyperparameter,
                parameter_sample_weights,
            ) in temperature_sample_weights.items():
                parameter_feature_weights = feature_weights[temperature][hyperparameter]
                parameter_feature_weights = {
                    temperature: {hyperparameter: parameter_feature_weights}
                }
                parameter_sample_weights = {
                    temperature: {hyperparameter: parameter_sample_weights}
                }
                (
                    rf_auroc_val,
                    rf_auprc_val,
                    rf_accuracy_val,
                    best_sample_weights_val,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = compute_classification_metrics_random_forest(
                    N,
                    R,
                    R,
                    columns,
                    parameter_sample_weights,
                    parameter_feature_weights,
                    target,
                    random_state=seed,
                    draw_with_feature_weights=draw_with_feature_weights,
                    n_estimators=500,
                    n_splits=10,
                    compute_feature_importance=False,
                )
                dropped_samples_val = np.count_nonzero(
                    np.array(best_sample_weights_val) == 0.0
                )
                dropped_samples_val_dict[float(temperature)][
                    float(hyperparameter)
                ].append(dropped_samples_val)
                auroc_val_dict[float(temperature)][float(hyperparameter)].append(
                    rf_auroc_val
                )
                auprc_val_dict[float(temperature)][float(hyperparameter)].append(
                    rf_auprc_val
                )
                accuracy_val_dict[float(temperature)][float(hyperparameter)].append(
                    rf_accuracy_val
                )

        for temperature, temperature_sample_weights in sample_weights.items():
            hyperparameter = 0.0

            temperature_feature_weights = feature_weights[temperature]
            temperature_feature_weights = {
                float(k): v for k, v in temperature_feature_weights.items()
            }
            temperature_sample_weights = {
                float(k): v for k, v in temperature_sample_weights.items()
            }

            tmp_sample_weights = temperature_sample_weights[float(hyperparameter)]
            tmp_feature_weights = temperature_feature_weights[float(hyperparameter)]

            tmp_sample_weights = {temperature: {hyperparameter: tmp_sample_weights}}
            tmp_feature_weights = {temperature: {hyperparameter: tmp_feature_weights}}
            (
                rf_auroc_val,
                rf_auprc_val,
                rf_accuracy_val,
                best_sample_weights_val,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = compute_classification_metrics_random_forest(
                N,
                R,
                R,
                columns,
                tmp_sample_weights,
                tmp_feature_weights,
                target,
                random_state=seed,
                draw_with_feature_weights=draw_with_feature_weights,
                n_estimators=500,
                n_splits=10,
                compute_feature_importance=False,
            )

            dropped_samples_individual_val = np.count_nonzero(
                np.array(best_sample_weights_val) == 0.0
            )
            dropped_samples_individual_val_dict[float(temperature)][
                float(hyperparameter)
            ].append(dropped_samples_individual_val)
            auroc_individual_val_dict[float(temperature)][float(hyperparameter)].append(
                rf_auroc_val
            )
            auprc_individual_val_dict[float(temperature)][float(hyperparameter)].append(
                rf_auprc_val
            )
            accuracy_individual_val_dict[float(temperature)][
                float(hyperparameter)
            ].append(rf_accuracy_val)
    else:
        for temperature, temperature_sample_weights in sample_weights.items():
            temperature_feature_weights = feature_weights[temperature]
            tmp_sample_weights = {temperature: temperature_sample_weights}
            tmp_feature_weights = {temperature: temperature_feature_weights}
            (
                rf_auroc_val,
                rf_auprc_val,
                rf_accuracy_val,
                best_sample_weights_val,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = compute_classification_metrics_random_forest(
                N,
                R,
                R,
                columns,
                tmp_sample_weights,
                tmp_feature_weights,
                target,
                random_state=seed,
                draw_with_feature_weights=draw_with_feature_weights,
                n_estimators=500,
                n_splits=10,
                compute_feature_importance=False,
            )

            dropped_samples_val = np.count_nonzero(
                np.array(best_sample_weights_val) == 0.0
            )
            dropped_samples_val_dict[float(temperature)].append(dropped_samples_val)
            auroc_val_dict[float(temperature)].append(rf_auroc_val)
            auprc_val_dict[float(temperature)].append(rf_auprc_val)
            accuracy_val_dict[float(temperature)].append(rf_accuracy_val)

        for temperature, temperature_sample_weights in sample_weights.items():
            hyperparameter = 1.0 if method_name == "fw-mrs-temperature-svm" else 0.0
            temperature_feature_weights = feature_weights[temperature]
            temperature_feature_weights = {
                float(k): v for k, v in temperature_feature_weights.items()
            }
            temperature_sample_weights = {
                float(k): v for k, v in temperature_sample_weights.items()
            }

            tmp_sample_weights = temperature_sample_weights[hyperparameter]
            tmp_sample_weights = {temperature: {hyperparameter: tmp_sample_weights}}
            tmp_feature_weights = temperature_feature_weights[hyperparameter]
            tmp_feature_weights = {temperature: {hyperparameter: tmp_feature_weights}}
            (
                rf_auroc_val,
                rf_auprc_val,
                rf_accuracy_val,
                best_sample_weights_val,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = compute_classification_metrics_random_forest(
                N,
                R,
                R,
                columns,
                tmp_sample_weights,
                tmp_feature_weights,
                target,
                random_state=seed,
                draw_with_feature_weights=draw_with_feature_weights,
                n_estimators=500,
                n_splits=10,
                compute_feature_importance=False,
            )

            dropped_samples_individual_val = np.count_nonzero(
                np.array(best_sample_weights_val) == 0.0
            )
            dropped_samples_individual_val_dict[float(temperature)].append(
                dropped_samples_individual_val
            )
            auroc_individual_val_dict[float(temperature)].append(rf_auroc_val)
            auprc_individual_val_dict[float(temperature)].append(rf_auprc_val)
            accuracy_individual_val_dict[float(temperature)].append(rf_accuracy_val)


def gbs_gesis_split(
    n_cv_splits, n_cv_repeats, df, target_values, random_generator
):
    # Is used to draw radom states
    max_int = 2**32 - 1
    N = df[df["label"] == 1]
    R = df[df["label"] == 0]
    for _ in range(n_cv_repeats):
        skf = StratifiedKFold(
            n_splits=n_cv_splits,
            shuffle=True,
            random_state=random_generator.randint(max_int),
        )
        for train_val_index, test_index in skf.split(R, R["Wahlteilnahme"]):
            R_train = R.iloc[train_val_index]
            R_test = R.iloc[test_index]
            yield N.copy(), R_train.copy(), R_test.copy()


def gbs_allensbach_split(
    n_cv_splits, n_cv_repeats, df, target_values, random_generator
):
    # Is used to draw radom states
    max_int = 2**32 - 1
    N = df[df["label"] == 1]
    R = df[df["label"] == 0]
    for _ in range(n_cv_repeats):
        skf = KFold(
            n_splits=n_cv_splits,
            shuffle=True,
            random_state=random_generator.randint(max_int),
        )
        for train_val_index, test_index in skf.split(R):
            R_train = R.iloc[train_val_index]
            R_test = R.iloc[test_index]
            yield N.copy(), R_train.copy(), R_test.copy()
