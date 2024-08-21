import json
from tqdm import trange
from sklearn.discriminant_analysis import StandardScaler
from utils.statistics import create_result_path
from utils.sampling import sample_with_test_set
from weighting_methods.feature_weighted_maximum_representative_subsampling import (
    feature_weighted_repeated_MRS,
)
import numpy as np

from utils.visualization_fw_mrs import (
    plot_budget_comparison_auroc,
    plot_feature_importance,
    plot_feature_weights,
    plot_budget_comparison_auroc_mean,
)

seed = 5


def feature_weight_budget_comparison_experiment(
    df,
    columns,
    target: str,
    number_of_repetitions: int = 50,
    bias_type: str = None,
    data_set_name: str = "",
    random_generator=None,
    method_name=None,
    drop=1,
    bias_fraction=0.25,
    validation_method="",
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

    # temperatures = [None, 0.1, 0.05, 0.01, 0.005]
    temperatures = [None, 0.05, 0.01, 0.005]

    result_path = create_result_path(
        method_name,
        bias_type,
        data_set_name,
        experiment_name="budget_comparison",
        bias_fraction=bias_fraction,
    )
    result_path = result_path / method_name / validation_method
    result_path.mkdir(parents=True, exist_ok=True)

    saved_weights_path = result_path / "saved_weights"
    saved_weights_path.mkdir(parents=True, exist_ok=True)

    auroc_path = result_path / "aurocs"
    auroc_path.mkdir(exist_ok=True, parents=True)

    feature_weights_list = []
    dropped_samples_list = []

    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    sample_df = df.copy()
    feature_weighted_aurocs_list = []
    abs_feature_importances_list = []
    feature_importances_list = []

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
            test_fraction=0.2,
            columns=columns,
        )

    for i in trange(number_of_repetitions):

        (
            random_forest_feature_weighted_aurocs,
            abs_feature_importances,
            feature_weights,
            dropped_samples,
        ) = feature_weighted_repeated_MRS(
            N=N,
            R=R,
            columns=columns,
            save_path=result_path,
            bias_variable=target,
            drop=drop,
            early_stopping=False,
            random_generator=random_generator,
            max_patience=len(N),
            target=target,
            budgets=temperatures,
            return_auroc=True,
            validation_method=validation_method,
            method_name=method_name,
            validate_iteration=1,
        )

        number_of_samples = len(N)
        feature_weighted_aurocs_list.append(random_forest_feature_weighted_aurocs)
        abs_feature_importances_list.append(abs_feature_importances)
        #feature_importances_list.append(feature_importances)
        feature_weights_list.append(feature_weights)
        dropped_samples_list.append(dropped_samples)

        # Visualize individual run results
        plot_budget_comparison_auroc(
            random_forest_feature_weighted_aurocs,
            number_of_samples,
            drop,
            auroc_path / f"iteration_{i}",
        )
        feature_weights_path = result_path / f"feature_weights" / str(i)
        feature_weights_path.mkdir(exist_ok=True, parents=True)
        plot_feature_weights(feature_weights, feature_weights_path)

        feature_importance_path = result_path / f"feature_importance" / str(i)
        feature_importance_path.mkdir(exist_ok=True, parents=True)
       # plot_feature_importance(abs_feature_importances, feature_importance_path)

    # Visualize mean results
    plot_budget_comparison_auroc_mean(
        feature_weighted_aurocs_list,
        number_of_samples,
        drop,
        result_path / "mean_auroc_comparison",
    )

    for data, file_name in zip(
        (
            feature_weighted_aurocs_list,
            abs_feature_importances_list,
            feature_importances_list,
            feature_weights_list,
            dropped_samples_list,
        ),
        (
            "feature_weighted_aurocs.json",
            "abs_feature_importances.json",
            "feature_importances.json",
            "feature_weights.json",
            "dropped_samples.json",
        ),
    ):
        save_list_to_json(result_path, data, file_name)

    save_mean_dropped_elements(result_path, dropped_samples_list)


def save_list_to_json(result_path, data, file_name):
    with open(result_path / file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_mean_dropped_elements(result_path, dropped_samples_list):
    mean_dropped_samples_dict = {}
    for _, budget in enumerate(dropped_samples_list[0].keys()):
        dropped_elements = []
        for dictionary in dropped_samples_list:
            dropped_elements.append(dictionary[budget])
        mean_dropped_samples_dict[f"{budget} mean"] = np.mean(dropped_elements)
        mean_dropped_samples_dict[f"{budget} std"] = np.std(dropped_elements)

    with open(result_path / "mean_dropped_samples", "w", encoding="utf-8") as file:
        json.dump(mean_dropped_samples_dict, file, indent=4)
