import json
import random
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from tqdm import trange

from utils.data_loader import load_weights, save_weights
from utils.metrics import compute_classification_metrics_random_forest
from utils.statistics import create_result_path

seed = 5


def perform_statistical_analysis_fw_mrs(
    df,
    columns,
    sample_weighting_method,
    method_name,
    target: str,
    n_cv_repeats: int,
    drop=1,
    load_previous_results=True,
    data_set_name=None,
    **args,
):
    """Analyze GBS corrected with Allensbach with two methods.

    :param method_one: First method
    :param method_two: Second method
    """
    np.random.seed(seed)
    random.seed(seed)
    result_path = create_result_path(
        method_name,
        "",
        data_set_name,
        experiment_name="statistical_analysis_fw_mrs",
        bias_fraction="",
    )
    iterations_path = result_path / "iterations"
    iterations_path.mkdir(exist_ok=True, parents=True)

    sample_weights_save_path = result_path / "sample_weights"
    feature_weights_save_path = result_path / "feature_weights"
    sample_weights_save_path.mkdir(exist_ok=True)
    feature_weights_save_path.mkdir(exist_ok=True)

    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    scaled_df = df.copy()
    first_random_generator = np.random.RandomState(seed)

    sample_weight_list = load_weights(sample_weights_save_path)
    feature_weight_list = load_weights(feature_weights_save_path)

    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]
    rf_auroc_list = []
    rf_auprc_list = []
    dropped_samples_list = []
    abs_feature_importance_list = []
    roc_curves_list = []

    for i in trange(n_cv_repeats):
        if method_name in ("fw-mrs-temperature", "fw-mrs-temperature-mean"):
            splitter = "feature_weighted_best"
            draw_with_feature_weights = True
            temperatures = [0.05, 0.01, 0.005]
            hyperparameter_list = [0.05, 0.025, 0.0]
        else:
            splitter = "best"
            draw_with_feature_weights = False
            temperatures = None
            hyperparameter_list = []

        mean = True if method_name == "fw-mrs-temperature-mean" else False
        if len(sample_weight_list) > i and load_previous_results:
            sample_weights = sample_weight_list[i]
            feature_weights = feature_weight_list[i]

        else:
            sample_weights, feature_weights = sample_weighting_method(
                N=scaled_N,
                R=scaled_R,
                target=target,
                columns=columns,
                drop=drop,
                early_stopping=True,
                random_generator=first_random_generator,
                budgets=temperatures,
                mean=mean,
                hyperparameter_list=hyperparameter_list,
                return_metrics=False,
            )
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
            _,
            _,
            _,
        ) = compute_classification_metrics_random_forest(
            scaled_N,
            scaled_N,
            columns,
            sample_weights,
            feature_weights,
            target,
            random_state=seed,
            draw_with_feature_weights=draw_with_feature_weights,
            splitter=splitter,
            n_estimators=500,
            n_splits=5,
            compute_feature_importance=True,
        )

        rf_auroc_list.append(rf_auroc)
        rf_auprc_list.append(rf_auprc)
        abs_feature_importance_list.append(abs_feature_importance.tolist())
        roc_curves_list.append(roc_curve_values)

        dropped_samples = np.count_nonzero(np.array(best_sample_weights) == 0.0)
        dropped_samples_list.append(dropped_samples)

    for file_name, data in zip(
        (
            "rf_auroc.json",
            "rf_auprc.json",
            "dropped_samples.json",
            "abs_feature_importance.json",
            "roc_curves.json",
        ),
        (
            rf_auroc_list,
            rf_auprc_list,
            dropped_samples_list,
            abs_feature_importance_list,
            roc_curves_list,
        ),
    ):
        with open(result_path / file_name, "w") as result_file:
            result_file.write(json.dumps(data))

    mean_feature_importance = np.mean(abs_feature_importance_list, axis=0)
    mean_feature_importance = {feature: importance for feature, importance in zip(columns, mean_feature_importance)}
    with open(result_path / "mean_feature_importance.json", "w") as result_file:
        result_file.write(json.dumps(mean_feature_importance))