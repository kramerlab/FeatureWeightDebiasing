import json
from tqdm import trange
from sklearn.discriminant_analysis import StandardScaler

from utils.statistics import create_result_path
from utils.sampling import sample_with_test_set
from weighting_methods.fw_mrs_for_downstream_comparison import (
    feature_weighted_repeated_MRS,
)
from utils.metrics import (
    compute_classification_metrics_random_forest,
    compute_classification_metrics_tree,
)
from utils.visualization_fw_mrs import (
    visualize_boxplot,
)

seed = 5


def feature_weight_downstream_comparison_experiment(
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
    validation_method="both",
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

    temperatures = [None, 0.1, 0.05, 0.01, 0.005, 0.001]

    result_path = create_result_path(
        method_name,
        bias_type,
        data_set_name,
        experiment_name="drop_auroc_comparison",
        bias_fraction=bias_fraction,
    )
    result_path = result_path / validation_method
    result_path.mkdir(parents=True, exist_ok=True)

    dropped_samples_list_dict = {}

    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    sample_df = df.copy()
    tree_auroc_dict = {}
    tree_auprc_dict = {}
    rf_auroc_dict = {}
    rf_auprc_dict = {}

    if method_name in ("fw-mrs-temperature", "mrs-tree", "mrs-forest"):
        drop_samples = True
    else:
        drop_samples = False

    for temperature in temperatures:
        tree_auroc_dict[temperature] = []
        tree_auprc_dict[temperature] = []
        rf_auroc_dict[temperature] = []
        rf_auprc_dict[temperature] = []
        dropped_samples_list_dict[temperature] = []

    _, _, T = sample_with_test_set(
        bias_type,
        sample_df,
        target,
        train_fraction=0.5,
        bias_fraction=bias_fraction,
        test_fraction=0.2,
        columns=columns,
    )

    for _ in trange(number_of_repetitions):
        N, R, _ = sample_with_test_set(
            bias_type,
            sample_df,
            target,
            train_fraction=0.5,
            bias_fraction=bias_fraction,
            test_fraction=0.2,
            columns=columns,
        )
        (
            best_sample_weights_dict,
            dropped_samples_dict,
            best_feature_weights_dict,
            best_inverse_feature_weights_dict,
        ) = feature_weighted_repeated_MRS(
            N=N,
            R=R,
            columns=columns,
            save_path=result_path,
            bias_variable=target,
            drop=drop,
            random_generator=random_generator,
            target=target,
            budgets=temperatures,
            validation_method=validation_method,
            method_name=method_name,
        )

        for temperature in temperatures:
            dropped_samples_list_dict[temperature].append(
                dropped_samples_dict[temperature]
            )
            sample_weights = best_sample_weights_dict[temperature]
            feature_weights = best_feature_weights_dict[temperature]
            inverse_feature_weights = best_inverse_feature_weights_dict[temperature]

            best_feature_weights_dict,
            best_inverse_feature_weights_dict,
            tree_auroc, tree_auprc = compute_classification_metrics_tree(
                N,
                R,
                T,
                columns,
                sample_weights,
                inverse_feature_weights,
                target,
                random_state=seed,
                n_splits=5,
            )

            tree_auroc_dict[temperature].append(tree_auroc)
            tree_auprc_dict[temperature].append(tree_auprc)

            rf_auroc, rf_auprc, _, _, _, _ = compute_classification_metrics_random_forest(
                N,
                R,
                T,
                columns,
                sample_weights,
                feature_weights,
                target,
                random_state=seed,
                draw_with_feature_weights=True,
                splitter="feature_weighted_best",
                n_estimators=500,
                n_splits=5,
                drop_samples=drop_samples,
            )
            rf_auroc_dict[temperature].append(rf_auroc)
            rf_auprc_dict[temperature].append(rf_auprc)

        # Visualize dropped samples and auroc comparison results
        for data_dict, y_label, file_name in zip(
            (
                dropped_samples_list_dict,
                tree_auroc_dict,
                tree_auprc_dict,
                rf_auroc_dict,
                rf_auprc_dict,
            ),
            ("Dropped Samples", "AUROC", "AUPRC", "AUROC", "AUPRC"),
            (
                "dropped_samples_comparison",
                "tree_auroc",
                "tree_auprc",
                "rf_auroc",
                "rf_auprc",
            ),
        ):
            visualize_boxplot(data_dict, y_label, file_name=result_path / file_name)

    for data, file_name in zip(
        (
            dropped_samples_list_dict,
            tree_auroc_dict,
            tree_auprc_dict,
            rf_auroc_dict,
            rf_auprc_dict,
        ),
        (
            "dropped_samples_dict",
            "tree_auroc_dict",
            "tree_auprc_dict",
            "rf_auroc_dict",
            "rf_auprc_dict",
        ),
    ):
        with open(result_path / f"{file_name}.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
