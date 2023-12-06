import json
import numpy as np

from pathlib import Path
from tqdm import trange
from sklearn.discriminant_analysis import StandardScaler

from utils.statistics import write_result_dict
from utils.sampling import sample
from utils.visualization import plot_weights
from utils.metrics import (
    compute_classification_metrics,
    compute_classification_metrics_svm,
    compute_metrics,
    calculate_rbf_gamma,
)

seed = 5


def downstream_experiment(
    df,
    columns,
    sample_weighting_method,
    feature_weighting_method,
    target: str,
    number_of_repetitions: int = 100,
    bias_type: str = None,
    data_set_name: str = "",
    random_generator=None,
    **args
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
    weighted_mmds_list = []
    biases_list = []
    mmd_list = []
    mean_list = []
    auroc_list = []
    auprc_list = []

    auroc_list_svm = []
    auprc_list_svm = []

    result_path = create_result_path(
        sample_weighting_method, feature_weighting_method, bias_type, data_set_name
    )
    sample_weights_save_path = result_path.parent / "sample_weights"
    feature_weights_save_path = result_path / "feature_weights"

    sample_weights_save_path.mkdir(exist_ok=True)
    feature_weights_save_path.mkdir(exist_ok=True)

    sample_weight_list = load_weights(sample_weights_save_path)
    feature_weight_list = load_weights(feature_weights_save_path)

    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    sample_df = df.copy()

    for i in trange(number_of_repetitions):
        N, R = sample(
            bias_type,
            sample_df,
            target,
            train_fraction=0.5,
            bias_fraction=0.75,
            columns=columns,
        )
        gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))

        if len(sample_weight_list) > i:
            sample_weights = np.array(sample_weight_list[i])
        else:
            sample_weights = sample_weighting_method(
                N,
                R,
                columns,
                save_path=result_path,
                bias_variable=target,
                mean_list=mean_list,
                mmd_list=mmd_list,
                drop=1,
                early_stopping=True,
                random_generator=random_generator,
                patience=25,
            )
            sample_weight_list.append(sample_weights.tolist())
            save_weights(sample_weights_save_path, sample_weight_list)

        if len(feature_weight_list) > i:
            feature_weights = np.array(feature_weight_list[i])
        else:
            feature_weights = feature_weighting_method(
                N=N,
                R=R,
                sample_weights=sample_weights,
                columns=columns,
                bias_variable=target,
                early_stopping=True,
                random_generator=random_generator,
            )
            feature_weight_list.append(feature_weights.tolist())
            save_weights(feature_weights_save_path, feature_weight_list)

        (
            weighted_mmd,
            relative_bias,
        ) = compute_metrics(
            N,
            R,
            scaler,
            columns,
            columns,
            sample_weights,
            feature_weights,
            gamma,
        )

        # auroc, auprc = compute_classification_metrics(
        #     N, R, columns, target, random_state=seed
        # )
        auroc = auprc = 0
        auroc_svm, auprc_svm = compute_classification_metrics_svm(
            N,
            R,
            sample_weights,
            feature_weights,
            gamma,
            columns,
            target,
            random_state=seed,
        )

        plot_weights(sample_weights, sample_weights_save_path, i)
        plot_weights(feature_weights, feature_weights_save_path, i)

        weighted_mmds_list.append(weighted_mmd)
        biases_list.append(relative_bias)
        auroc_list.append(auroc)
        auprc_list.append(auprc)

        auroc_list_svm.append(auroc_svm)
        auprc_list_svm.append(auprc_svm)

    result_dict = write_result_dict(
        N.drop(["label"], axis="columns").columns,
        weighted_mmds_list,
        biases_list,
        auroc_list,
        auprc_list,
        auroc_list_svm,
        auprc_list_svm,
        len(N),
    )

    with open(result_path / "results.json", "w") as result_file:
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


def create_result_path(
    sample_weighting_method,
    feature_weighting_method,
    bias_type,
    data_set_name,
):
    """The function creates the save path and makes the directory.

    :param method: Method name
    :param bias_type: Bias type name
    :param data_set_name: Data set name
    :return: The result path
    """
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    result_path = (
        result_path
        / "downstream"
        / data_set_name
        / bias_type
        / sample_weighting_method.__name__
        / feature_weighting_method.__name__
    )
    result_path.mkdir(exist_ok=True, parents=True)
    return result_path
