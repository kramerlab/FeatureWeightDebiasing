import json

from experiments.downstream_tasks import load_weights, repeated_train_val_test_split
from utils.data_loader import save_weights
from utils.statistics import create_result_path
from utils.sampling import sample_N
from utils.metrics import compute_classification_metrics_random_forest, scale_df
from utils.visualization_fw_mrs import visualize_boxplot
import numpy as np

seed = 5
sampling_random_generator = np.random.RandomState(seed)


def downstream_task_validation_comparison(
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

    if method_name in ("fw-mrs-temperature", "fw-mrs-temperature-mean"):
        splitter = "feature_weighted_best"
        draw_with_feature_weights = True
        temperatures = [0.1, 0.05, 0.01, 0.005]
        hyperparameter_list = [0.05, 0.025, 0.0]

    elif method_name == "mrs-forest":
        hyperparameter_list = [0.05, 0.025, 0.0]
        splitter = "best"
        draw_with_feature_weights = False
        temperatures = None

    mean = True if method_name == "fw-mrs-temperature-mean" else False

    result_path = create_result_path(
        method_name,
        bias_type,
        data_set_name,
        experiment_name="downstrea_task_validation",
        bias_fraction=bias_fraction,
    )

    sample_weights_save_path = result_path / "sample_weights"
    feature_weights_save_path = result_path / "feature_weights"

    sample_weights_save_path.mkdir(exist_ok=True)
    feature_weights_save_path.mkdir(exist_ok=True)

    sample_weights_list = load_weights(sample_weights_save_path)
    feature_weights_list = load_weights(feature_weights_save_path)

    sample_df, _ = scale_df(df, columns)
    dropped_samples_dict = {temperature: [] for temperature in temperatures}
    auroc_dict = {temperature: [] for temperature in temperatures}
    auprc_dict = {temperature: [] for temperature in temperatures}

    if data_set_name in ("gbs_gesis", "gbs_allensbach"):
        N = sample_df[sample_df["label"] == 1]
        R = sample_df[sample_df["label"] == 0]

    for i, (N, R, _) in enumerate(
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

        if len(sample_weights_list) > i and load_previous_results:
            sample_weights = sample_weights_list[i]
            feature_weights = feature_weights_list[i]

        else:
            sample_weights, feature_weights = sample_weighting_method(
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
                return_metrics=False,
                mean=mean,
            )

            feature_weights_list.append(feature_weights)
            sample_weights_list.append(sample_weights)

            save_weights(sample_weights_save_path, sample_weights_list)
            save_weights(feature_weights_save_path, feature_weights_list)

        for temperature, temperature_sample_weights in sample_weights.items():
            temperature_feature_weights = {"tmp": feature_weights[temperature]}
            temperature_sample_weights = {"tmp": temperature_sample_weights}

            (
                rf_auroc,
                rf_auprc,
                best_sample_weights,
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
                splitter=splitter,
                n_estimators=500,
                n_splits=5,
            )

            dropped_samples = np.count_nonzero(np.array(best_sample_weights) == 0.0)
            dropped_samples_dict[temperature].append(dropped_samples)
            auroc_dict[temperature].append(rf_auroc)
            auprc_dict[temperature].append(rf_auprc)
        visualize_boxplot(auroc_dict, "AUROC", result_path / "auroc_comparison")
        visualize_boxplot(auprc_dict, "AUPRC", result_path / "auprc_comparison")
        visualize_boxplot(
            dropped_samples_dict, "Dropped Samples", result_path / "dropped_samples_comparison"
        )


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
