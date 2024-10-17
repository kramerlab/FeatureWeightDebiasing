import json
import numpy as np

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.discriminant_analysis import StandardScaler

from utils.data_loader import load_weights, save_weights
from utils.statistics import (
    create_result_path,
    write_result_dict,
    write_result_dict_test_set,
)
from utils.sampling import sample_N
from utils.visualization import plot_rocs_downstream
from utils.metrics import (
    calculate_rbf_gamma,
    compute_classification_metrics_random_forest,
    compute_metrics,
)

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

    if method_name in ("fw-mrs-temperature", "fw-mrs-temperature-mean"):
        splitter = "feature_weighted_best"
        draw_with_feature_weights = True
        temperatures = [0.1, 0.05, 0.01, 0.005]
        hyperparameter_list = [0.05, 0.025, 0.0]
    elif method_name == "fw-mrs-svm":
        temperatures = [0.1, 0.05, 0.01, 0.005]
        hyperparameter_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        splitter = "feature_weighted_best"
        draw_with_feature_weights = True
    elif method_name == "mrs-forest":
        hyperparameter_list = [0.05, 0.025, 0.0]
        splitter = "best"
        draw_with_feature_weights = False
        temperatures = None
    else:
        splitter = "best"
        draw_with_feature_weights = False
        temperatures = None
        hyperparameter_list = []

    mean = True if method_name == "fw-mrs-temperature-mean" else False

    for i, (N, R, T) in enumerate(
        repeated_train_val_test_split(
            n_cv_splits,
            n_cv_repeats,
            sample_df,
            sample_df[target],
            sampling_random_generator,
        )
    ):
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
                mean=mean,
                compute_bias=False,
            )

            feature_weight_list.append(feature_weights)
            sample_weight_list.append(sample_weights)

            save_weights(sample_weights_save_path, sample_weight_list)
            save_weights(feature_weights_save_path, feature_weight_list)

        if method_name == "mrs-forest":
            sample_weights = {0.0: sample_weights}
            feature_weights = {0.0: feature_weights}

        if method_name not in (
            "fw-mrs-svm",
            "fw-mrs-temperature",
            "fw-mrs-temperature-mean",
        ):
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
            weighted_mmd = np.ones(len(N.columns))
            relative_bias = np.ones(len(N.columns))
            wasserstein_distances = np.ones(len(N.columns))

        (
            rf_auroc,
            rf_auprc,
            best_sample_weights,
            abs_feature_importance,
            roc_curve_values,
            best_temperature,
            best_hyperparameter,
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
            n_splits=5,
        )
        dropped_samples = np.count_nonzero(np.array(best_sample_weights) == 0.0)
        dropped_samples_list.append(dropped_samples)
        best_temperature_list.append(best_temperature)
        best_hyperparameter_list.append(best_hyperparameter)

        # (
        #     svc_auroc,
        #     svc_auprc,
        # ) = compute_classification_metrics_svm(
        #     N,
        #     T,
        #     columns,
        #     sample_weights,
        #     feature_weights,
        #     target,
        #     random_state=seed,
        #     draw_with_feature_weights=draw_with_feature_weights,
        #     n_splits=10,
        # )

        svc_auroc = 0
        svc_auprc = 0

        # (
        #     pseudo_targets_auroc,
        #     pseudo_targets_auprc,
        # ) = compute_classification_metrics_pseudo_labels(
        #     R,
        #     T,
        #     columns,
        #     best_clf,
        #     target,
        #     random_state=seed,
        # )
        pseudo_targets_auroc = 0
        pseudo_targets_auprc = 0

        pseudo_targets_auroc_list.append(pseudo_targets_auroc)
        pseudo_targets_auprc_list.append(pseudo_targets_auprc)
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
                best_hyperparameter_list,
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
    )

    with open(result_path / "similarity_results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    result_dict = {}
    result_dict = write_result_dict_test_set(
        rf_auroc_list,
        rf_auprc_list,
        svc_auroc_list,
        svc_auprc_list,
        pseudo_targets_auroc_list,
        pseudo_targets_auprc_list,
        dropped_samples_list,
        len(N),
    )

    with open(result_path / "classification_results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))


# Used to draw radom states
max_int = 2**32 - 1


def repeated_train_val_test_split(
    n_cv_splits, n_cv_repeats, df, target, random_generator
):
    for _ in range(n_cv_repeats):
        train_val_samples, test_samples, train_val_y, _ = train_test_split(
            df,
            target,
            random_state=random_generator.randint(max_int),
            stratify=target,
            test_size=(1 / 3),
        )
        train_samples, val_samples, _, _ = train_test_split(
            train_val_samples,
            train_val_y,
            stratify=train_val_y,
            random_state=random_generator.randint(max_int),
            test_size=0.5,
        )

        min_samples = np.min(
            [
                len(sample_list)
                for sample_list in (
                    train_samples.values,
                    val_samples.values,
                    test_samples.values,
                )
            ]
        )

        train_samples = train_samples.iloc[:min_samples].copy()
        val_samples = val_samples.iloc[:min_samples].copy()
        test_samples = test_samples.iloc[:min_samples].copy()
        splits_list = [
            train_samples,
            val_samples,
            test_samples,
        ]
        train_index = -1
        val_index = 0
        test_index = 1
        for _ in range(n_cv_splits):
            train_index = (train_index + 1) % n_cv_splits
            val_index = (val_index + 1) % n_cv_splits
            test_index = (test_index + 1) % n_cv_splits

            yield splits_list[train_index], splits_list[val_index], splits_list[
                test_index
            ]
