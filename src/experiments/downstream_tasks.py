import json
import numpy as np

from tqdm import trange
from sklearn.discriminant_analysis import StandardScaler

from utils.statistics import (
    create_result_path,
    write_result_dict,
    write_result_dict_test_set,
)
from utils.sampling import sample_with_test_set
from utils.visualization import plot_feature_weights, plot_rocs_downstream
from utils.metrics import (
    calculate_rbf_gamma,
    compute_classification_metrics_random_forest,
    compute_classification_metrics_svm,
    compute_classification_metrics_pseudo_labels,
    compute_metrics,
)

seed = 5


def downstream_experiment_with_test_set(
    df,
    columns,
    sample_weighting_method,
    target: str,
    number_of_repetitions: int = 50,
    bias_type: str = None,
    data_set_name: str = "",
    random_generator=None,
    load_previous_results=True,
    bias_fraction=0.75,
    drop=1,
    validation_method="random_forest",
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

    svc_auroc_list = []
    svc_auprc_list = []

    svc_auroc_list = []
    svc_auprc_list = []

    pseudo_targets_auroc_list = []
    pseudo_targets_auprc_list = []

    weighted_mmds_list = []
    biases_list = []
    wasserstein_distance_list = []

    abs_feature_importance_list = []
    feature_importance_list = []
    roc_curves_list = []
    best_temperature_list = []

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
    roc_path = result_path / "rocs"

    result_path.mkdir(exist_ok=True)
    classificiation_result_path.mkdir(exist_ok=True)
    sample_weights_save_path.mkdir(exist_ok=True)
    feature_weights_save_path.mkdir(exist_ok=True)
    roc_path.mkdir(exist_ok=True)

    sample_weight_list = load_weights(sample_weights_save_path)
    feature_weight_list = load_weights(feature_weights_save_path)

    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    sample_df = df.copy()

    if method_name == "fw-mrs-temperature":
        splitter = "feature_weighted_best"
        draw_with_feature_weights = True
        temperatures = [0.1, 0.05, 0.01, 0.005]
        explicit_weights = False
    elif method_name == "fw-mrs-svm":
        temperatures = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
        # temperatures = [1e-1, 1e0,]
        splitter = "feature_weighted_best"
        draw_with_feature_weights = True
        explicit_weights = False
    else:
        splitter = "best"
        draw_with_feature_weights = False
        temperatures = None
        explicit_weights = True

    for i in trange(number_of_repetitions):
        N, R, T = sample_with_test_set(
            bias_type,
            sample_df,
            target,
            train_fraction=0.5,
            bias_fraction=bias_fraction,
            test_fraction=0.1,
            columns=columns,
        )
        gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))

        if len(sample_weight_list) > i and explicit_weights and load_previous_results:
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
                validation_method=validation_method,
                method_name=method_name,
            )

            feature_weight_list.append(feature_weights)
            sample_weight_list.append(sample_weights)

            save_weights(sample_weights_save_path, sample_weight_list)
            save_weights(feature_weights_save_path, feature_weight_list)

        if feature_weights is None:
            feature_weights = (np.ones(len(columns)) / len(columns)).tolist()

        if method_name not in ("fw-mrs-svm", "fw-mrs-temperature"):
            weighted_mmd, relative_bias, wasserstein_distances = compute_metrics(
                N,
                R,
                scaler,
                columns,
                columns,
                sample_weights,
                gamma,
            )
        else:
            weighted_mmd = 0
            relative_bias = 0
            wasserstein_distances = 0

        (
            rf_auroc,
            rf_auprc,
            best_sample_weights,
            abs_feature_importance,
            roc_curve_values,
            best_temperature,
            best_clf,
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
            n_splits=10,
        )
        dropped_samples = np.count_nonzero(np.array(best_sample_weights) == 0.0)
        dropped_samples_list.append(dropped_samples)
        best_temperature_list.append(best_temperature)

        (
            svc_auroc,
            svc_auprc,
            _,
            _,
            _,
        ) = compute_classification_metrics_svm(
            N,
            T,
            columns,
            sample_weights,
            feature_weights,
            target,
            random_state=seed,
            draw_with_feature_weights=draw_with_feature_weights,
            n_splits=10,
            compute_feature_importance=False,
        )

        (
            pseudo_targets_auroc,
            pseudo_targets_auprc,
        ) = compute_classification_metrics_pseudo_labels(
            R,
            T,
            columns,
            best_clf,
            target,
            random_state=seed,
        )

        pseudo_targets_auroc_list.append(pseudo_targets_auroc)
        pseudo_targets_auprc_list.append(pseudo_targets_auprc)

        if not feature_weights is None:
            plot_feature_weights(feature_weights, feature_weights_save_path, i)
            weighted_mmds_list.append(weighted_mmd)
            biases_list.append(relative_bias)
            wasserstein_distance_list.append(wasserstein_distances)
            rf_auroc_list.append(rf_auroc)
            rf_auprc_list.append(rf_auprc)
            svc_auroc_list.append(svc_auroc)
            svc_auprc_list.append(svc_auprc)

        # plot_sample_weights(sample_weights, sample_weights_save_path, i)
        # plot_feature_weights(feature_weights, feature_weights_save_path, i)

        abs_feature_importance_list.append(abs_feature_importance.tolist())
        roc_curves_list.append(roc_curve_values)

        plot_rocs_downstream(roc_curve_values, roc_path / f"roc_iteration_{i}")
        for result_list, file_name in zip(
            (
                rf_auroc_list,
                rf_auprc_list,
                svc_auroc_list,
                svc_auprc_list,
                pseudo_targets_auroc_list,
                pseudo_targets_auprc_list,
                dropped_samples_list,
                abs_feature_importance_list,
                feature_importance_list,
                roc_curves_list,
                best_temperature_list,
            ),
            (
                "rf_auroc",
                "rf_auprc",
                "svm_auroc",
                "svm_auprc",
                "pseudo_target_auroc",
                "pseudo_target_auprc",
                "dropped_samples",
                "abs_feature_importance",
                "feature_importance",
                "roc_curves",
                "best_temperature",
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
        explicit_weights=explicit_weights,
    )

    with open(result_path / "similarity_results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    result_dict = {}
    result_dict = write_result_dict_test_set(
        rf_auroc_list,
        rf_auprc_list,
        svc_auroc_list,
        svc_auprc_list,
        dropped_samples_list,
        len(N),
    )

    with open(result_path / "classification_results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))


def save_weights(path, weights_list):
    with open(path / "weights.json", "w") as file:
        json.dump(weights_list, file, indent=4)


def load_weights(path):
    weight_file = path / "weights.json"
    if weight_file.is_file():
        with open(weight_file, "r") as file:
            weights = json.load(file)
    else:
        weights = []
    return weights
