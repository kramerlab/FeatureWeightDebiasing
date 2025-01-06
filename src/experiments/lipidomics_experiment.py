import json
import numpy as np

from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_saved_results, save_results
from utils.parameter import set_parameter
from utils.statistics import (
    create_result_path,
    write_result_dict,
    write_result_dict_test_set_lipidomics,
)
from utils.metrics import (
    calculate_rbf_gamma,
    compute_classification_metrics_random_forest_lipidomics,
    compute_metrics,
)
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold

seed = 5
sampling_random_generator = np.random.RandomState(seed)


def lipidomics_quantification_experiment(
    df,
    columns,
    sample_weighting_method,
    target: str,
    n_cv_repeats: int,
    n_cv_splits: int,
    random_generator=None,
    load_previous_results=True,
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
    rf_auroc_cal_list = []
    rf_auprc_cal_list = []

    rf_auroc_set2_list = []
    rf_auprc_set2_list = []

    weighted_mmds_list = []
    biases_list = []
    wasserstein_distance_list = []

    best_temperature_list = []
    best_hyperparameter_list = []

    dropped_samples_list = []

    result_path = create_result_path(
        method_name,
        "",
        "",
        experiment_name="lipidomics_task",
        bias_fraction="",
        prefix="../../lipid_results",
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

    cal_scaler = StandardScaler()
    cal_imputer = KNNImputer(n_neighbors=10, weights="uniform")
    set2_scaler = StandardScaler()
    set2_imputer = KNNImputer(n_neighbors=10, weights="uniform")

    (
        draw_with_feature_weights,
        temperatures,
        _,
        _,
        _,
        _,
        hyperparameter_list,
    ) = set_parameter(method_name)

    for i, (N_train, N_test, R_train, R_test) in enumerate(
        repeated_train_val_test_split_for_lipidomics(
            n_cv_splits,
            n_cv_repeats,
            df,
            target,
            sampling_random_generator,
        )
    ):
        N_train[columns] = cal_scaler.fit_transform(N_train[columns])
        N_train[columns] = cal_imputer.fit_transform(N_train[columns])
        N_train["label"] = 1
        N_test[columns] = cal_scaler.transform(N_test[columns])
        N_test[columns] = cal_imputer.transform(N_test[columns])

        R_train[columns] = set2_scaler.fit_transform(R_train[columns])
        R_train[columns] = set2_imputer.fit_transform(R_train[columns])
        R_train["label"] = 0
        R_test[columns] = set2_scaler.transform(R_test[columns])
        R_test[columns] = set2_imputer.transform(R_test[columns])

        gamma = calculate_rbf_gamma(
            np.append(N_train[columns], R_train[columns], axis=0)
        )

        if len(sample_weight_list) > i and load_previous_results:
            sample_weights = sample_weight_list[i]
            feature_weights = feature_weight_list[i]

        else:
            sample_weights, feature_weights = sample_weighting_method(
                N=N_train,
                R=R_train,
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

        (
            rf_auroc_cal,
            rf_auprc_cal,
            rf_auroc_set2,
            rf_auprc_set2,
            best_sample_weights,
            best_temperature,
            best_hyperparameter,
        ) = compute_classification_metrics_random_forest_lipidomics(
            N_train,
            N_test,
            R_test,
            columns,
            sample_weights,
            feature_weights,
            target,
            random_state=seed,
            draw_with_feature_weights=draw_with_feature_weights,
            n_estimators=200,
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
                N_train,
                R_train,
                cal_scaler,
                columns.values,
                target,
                best_sample_weights,
                gamma,
            )

        else:
            weighted_mmd = np.ones(len(N_train.columns))
            relative_bias = np.ones(len(N_train.columns))
            wasserstein_distances = np.ones(len(N_train.columns))

        weighted_mmds_list.append(weighted_mmd)
        biases_list.append(relative_bias)
        wasserstein_distance_list.append(wasserstein_distances)
        rf_auroc_cal_list.append(rf_auroc_cal)
        rf_auprc_cal_list.append(rf_auprc_cal)
        rf_auroc_set2_list.append(rf_auroc_set2)
        rf_auprc_set2_list.append(rf_auprc_set2)

    for result_list, file_name in zip(
        (
            rf_auroc_cal_list,
            rf_auprc_cal_list,
            rf_auroc_set2_list,
            rf_auprc_set2_list,
            dropped_samples_list,
            best_temperature_list,
            best_hyperparameter_list,
        ),
        (
            "rf_auroc_cal",
            "rf_auprc_cal",
            "rf_auroc_set2",
            "rf_auprc_set2",
            "dropped_samples",
            "best_temperature",
            "best_hyperparameter",
        ),
    ):
        with open(
            classificiation_result_path / f"{file_name}.json", "w"
        ) as result_file:
            result_file.write(json.dumps(result_list))

    result_dict = write_result_dict(
        N_train.drop(["label", "Method", "ID"], axis="columns").columns,
        weighted_mmds_list,
        biases_list,
        wasserstein_distance_list,
    )

    with open(result_path / "similarity_results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    result_dict = {}
    result_dict = write_result_dict_test_set_lipidomics(
        rf_auroc_cal_list,
        rf_auprc_cal_list,
        rf_auroc_set2_list,
        rf_auprc_set2_list,
        dropped_samples_list,
        len(N_train),
    )

    with open(result_path / "classification_results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))


def repeated_train_val_test_split_for_lipidomics(
    n_cv_splits, n_cv_repeats, df, target, random_generator
):
    # Is used to draw radom states
    max_int = 2**32 - 1

    cal_data = df[df["Method"] == "cal"]
    set2_data = df[df["Method"] == "set2"]
    for _ in range(n_cv_repeats):
        skf = StratifiedKFold(
            n_splits=n_cv_splits,
            shuffle=True,
            random_state=random_generator.randint(max_int),
        )
        for train_val_index, test_index in skf.split(cal_data, cal_data[target]):
            train_samples_cal = cal_data.iloc[train_val_index].copy()
            test_samples_cal = cal_data.iloc[test_index].copy()

            train_samples_set2 = set2_data.iloc[train_val_index].copy()
            test_samples_set2 = set2_data.iloc[test_index].copy()

            yield train_samples_cal, test_samples_cal, train_samples_set2, test_samples_set2
