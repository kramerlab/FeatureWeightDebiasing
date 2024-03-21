import json
from tqdm import trange
from sklearn.discriminant_analysis import StandardScaler
from utils.statistics import create_result_path
from utils.sampling import sample
from weighting_methods.feature_weighted_maximum_representative_subsampling import (
    feature_weighted_repeated_MRS,
)
from utils.visualization import (
    plot_budget_comparison_auroc,
    plot_feature_importance,
    plot_feature_weights,
)

seed = 5
budgets = [1.0, 0.1, 0.01, 0.001]


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

    result_path = create_result_path(
        method_name, bias_type, data_set_name, experiment_name="budget_comparison"
    )
    auroc_path = result_path / "aurocs"
    auroc_path.mkdir(exist_ok=True, parents=True)

    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    sample_df = df.copy()
    feature_weighted_aurocs_list = []
    feature_importances_list = []
    feature_weights_list = []

    for i in trange(number_of_repetitions):
        N, R = sample(
            bias_type,
            sample_df,
            target,
            train_fraction=0.5,
            bias_fraction=0.1,
            columns=columns,
        )

        feature_weighted_aurocs, feature_importances, feature_weights, mrs_iteration = (
            feature_weighted_repeated_MRS(
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
                budgets=budgets,
                return_auroc=True,
            )
        )

        number_of_samples = len(N)
        feature_weighted_aurocs_list.append(feature_weighted_aurocs)
        feature_importances_list.append(feature_importances)
        feature_weights_list.append(feature_weights)

        # Visualize individual run results

        plot_budget_comparison_auroc(
            feature_weighted_aurocs,
            number_of_samples,
            drop,
            auroc_path / f"iteration_{i}",
        )
        feature_weights_path = result_path / f"feature_weights_{i}/"
        feature_weights_path.mkdir(exist_ok=True, parents=True)
        plot_feature_weights(feature_weights, result_path / f"feature_weights_{i}/")

        feature_importance_path = result_path / f"feature_importance_{i}/"
        feature_importance_path.mkdir(exist_ok=True, parents=True)
        plot_feature_importance(feature_importances, feature_importance_path)

    # Visualize mean results
    # plot_budget_comparison_auroc(feature_weighted_aurocs_list)
    # plot_feature_weights(feature_weights_list, result_path / "mean_feature_weights")
    # plot_feature_importance(
    #    feature_importances_list, result_path / "mean_feature_importance"
    # )

    with open(
        result_path / f"feature_weighted_aurocs.json", "w", encoding="utf-8"
    ) as file:
        json.dump(feature_weighted_aurocs, file, indent=4)

    with open(result_path / f"feature_importances.json", "w", encoding="utf-8") as file:
        json.dump(feature_importances_list, file, indent=4)

    with open(result_path / f"feature_weights.json", "w", encoding="utf-8") as file:
        json.dump(feature_weights_list, file, indent=4)
