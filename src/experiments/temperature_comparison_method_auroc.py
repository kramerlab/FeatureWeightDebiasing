import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.downstream_tasks import load_saved_results
from utils.data_loader import save_results
from utils.statistics import create_result_path
from utils.sampling import sample_N
from utils.metrics import (
    calculate_rbf_gamma,
    compute_metrics,
)
from utils.visualization_fw_mrs import plot_temperature_comparison_auroc_mean
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
    bias_type: str = "",
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
        temperatures = [0.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]
        hyperparameter_list = [0.0, 0.001, 0.01, 0.025]
        fixed_hyperparameter = 0.01
    if method_name in ("fw-mrs-temperature-svm",):
        temperatures = [0.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]
        hyperparameter_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
        fixed_hyperparameter = 1e0
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
        auroc_save_path, file_name="aurocs"
    )
    optimised_feature_weighted_aurocs_list = []
    fixed_feature_weighted_aurocs_list = []
    all_feature_weighted_aurocs_list = []
    sample_weights_list = load_saved_results(sample_weights_save_path)
    feature_weights_list = load_saved_results(feature_weights_save_path)
    feature_importances_list = load_saved_results(
        feature_weights_save_path, "feature_importances"
    )
    optimized_mmd_dict = {temperature: [] for temperature in temperatures}
    all_mmd_list = []

    number_of_samples_list = []
    scaler = StandardScaler()

    if data_set_name in ("gbs_gesis", "gbs_allensbach"):
        split_method = gbs_split
    else:
        split_method = repeated_train_val_test_split

    for i, (N, R) in enumerate(
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

        gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))
        if len(sample_weights_list) > i and load_previous_results:
            sample_weights = sample_weights_list[i]
            feature_weights = feature_weights_list[i]
            random_forest_feature_weighted_aurocs = feature_weighted_aurocs_list[i]
            fixed_hyperparameter = str(fixed_hyperparameter)

        else:
            fixed_hyperparameter = float(fixed_hyperparameter)
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

            feature_weights_list.append(feature_weights)
            feature_importances_list.append(feature_importances)
            sample_weights_list.append(sample_weights)

            save_results(sample_weights_save_path, sample_weights_list)
            save_results(feature_weights_save_path, feature_weights_list)
            save_results(
                feature_importances_save_path,
                feature_importances_list,
                "feature_importances",
            )

            feature_weighted_aurocs_list.append(random_forest_feature_weighted_aurocs)
            save_results(
                auroc_save_path,
                feature_weighted_aurocs_list,
                file_name="aurocs",
            )

        optimised_random_forest_feature_weighted_aurocs = {}
        fixed_feature_weighted_aurocs = {}
        best_feature_weighted_aurocs = []

        best_mmd_all = np.inf
        for temperature, temperature_sample_weights in sample_weights.items():
            temperature_feature_weights = feature_weights[temperature]
            best_mmd_temperature = np.inf
            for (
                hyperparameter,
                one_sample_weights,
            ) in temperature_sample_weights.items():
                one_feature_weights = temperature_feature_weights[hyperparameter]
                if len(one_sample_weights) == 0:
                    continue

                key = list(one_sample_weights.keys())[0]
                feature_weighted_mmd, _, _ = compute_metrics(
                    N,
                    R,
                    scaler,
                    columns,
                    target,
                    one_sample_weights[key],
                    one_feature_weights,
                    gamma,
                )
                if feature_weighted_mmd < best_mmd_temperature:
                    best_hyperparameter_temperature = hyperparameter
                    best_mmd_temperature = feature_weighted_mmd

                if feature_weighted_mmd < best_mmd_all:
                    best_hyperparameter_all = hyperparameter
                    best_temperature_all = temperature
                    best_mmd_all = feature_weighted_mmd

            optimised_random_forest_feature_weighted_aurocs[temperature] = (
                random_forest_feature_weighted_aurocs[temperature][
                    best_hyperparameter_temperature
                ]
            )
            optimized_mmd_dict[temperature].append(best_mmd_temperature)
            fixed_feature_weighted_aurocs[temperature] = (
                random_forest_feature_weighted_aurocs[temperature][fixed_hyperparameter]
            )

            key = list(
                temperature_sample_weights[best_hyperparameter_temperature].keys()
            )[0]
            dropped_samples_dict[temperature].append(
                np.count_nonzero(
                    np.array(
                        temperature_sample_weights[best_hyperparameter_temperature][key]
                    )
                    == 0.0
                )
            )

        optimised_feature_weighted_aurocs_list.append(
            optimised_random_forest_feature_weighted_aurocs
        )

        fixed_feature_weighted_aurocs_list.append(fixed_feature_weighted_aurocs)

        best_feature_weighted_aurocs = random_forest_feature_weighted_aurocs[
            best_temperature_all
        ][best_hyperparameter_all]
        all_feature_weighted_aurocs_list.append(best_feature_weighted_aurocs)
        all_mmd_list.append(best_mmd_all)

        save_results(
            auroc_save_path,
            optimised_feature_weighted_aurocs_list,
            file_name="optimised_method_aurocs",
        )
        save_results(
            auroc_save_path,
            fixed_feature_weighted_aurocs_list,
            file_name="fixed_method_aurocs",
        )
        save_results(
            auroc_save_path,
            all_feature_weighted_aurocs_list,
            file_name="all_method_aurocs",
        )

        meta_data_dict = {
            "n_dropped": drop,
            "number_of_samples": number_of_samples_list,
        }
        with open(result_path / "metadata.json", "w") as file:
            json.dump(meta_data_dict, file)

        number_of_samples_list.append(len(N))

        plot_temperature_comparison_auroc_mean(
            optimised_feature_weighted_aurocs_list,
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

        with open(result_path / "optimised_mmd.json", "w") as result_file:
            result_file.write(json.dumps(optimized_mmd_dict))

        with open(result_path / "all_mmds.json", "w") as result_file:
            result_file.write(json.dumps(all_mmd_list))

        with open(result_path / "dropped_elements.json", "w") as result_file:
            result_file.write(json.dumps(dropped_samples_dict))


def gbs_split(n_cv_splits, n_cv_repeats, df, target_values, random_generator):
    # Is used to draw radom states
    N = df[df["label"] == 1]
    R = df[df["label"] == 0]
    for _ in range(n_cv_repeats * n_cv_splits):
        yield N.copy(), R.copy()


def repeated_train_val_test_split(
    n_cv_splits, n_cv_repeats, df, target_values, random_generator
):
    # Is used to draw radom states
    max_int = 2**32 - 1
    for _ in range(n_cv_repeats):
        skf = StratifiedKFold(
            n_splits=n_cv_splits,
            shuffle=True,
            random_state=random_generator.randint(max_int),
        )
        for train_val_index, test_index in skf.split(df, target_values):
            train_val_samples = df.iloc[train_val_index].copy()
            train_samples = df.iloc[test_index].copy()

            yield train_val_samples, train_samples
