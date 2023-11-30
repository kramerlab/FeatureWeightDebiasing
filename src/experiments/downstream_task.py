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


def downstream_experiment(
    df,
    columns,
    weighting_method,
    target: str,
    method: str = "",
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
    wasserstein_parameter_list = []
    mmd_list = []
    mean_list = []
    auroc_list = []
    auprc_list = []

    auroc_list_svm = []
    auprc_list_svm = []

    result_path = create_result_path(method, bias_type, data_set_name)
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

        feature_weights = weighting_method(
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
        N[columns] = N[columns] * feature_weights
        R[columns] = R[columns] * feature_weights

        unique_list_n = []
        unique_list_r = []
        for column in N.columns:
            unique_list_n.append(N[column].sort_values().unique())
            unique_list_r.append(R[column].sort_values().unique())

        # tmp = [unique_r - unique_n for unique_r, unique_n in zip(unique_list_r, unique_list_n)]
        # auroc, auprc= compute_classification_metrics(
        #    N, R, columns, target, random_state=5
        # )
        auroc = auprc = auroc_svm = auprc_svm = 0

        # auroc_svm, auprc_svm = compute_classification_metrics_svm(
        #     N, R, columns, target, random_state=5
        # )

        (
            weighted_mmd,
            relative_bias,
            wasserstein_distances,
        ) = compute_metrics(
            N,
            R,
            scaler,
            columns,
            columns,
            gamma,
            feature_weights,
        )

        plot_weights(feature_weights, result_path / "weights", i)

        weighted_mmds_list.append(weighted_mmd)
        biases_list.append(relative_bias)
        wasserstein_parameter_list.append(wasserstein_distances)
        auroc_list.append(auroc)
        auprc_list.append(auprc)

        auroc_list_svm.append(auroc_svm)
        auprc_list_svm.append(auprc_svm)

    result_dict = write_result_dict(
        N.drop(["label"], axis="columns").columns,
        weighted_mmds_list,
        biases_list,
        wasserstein_parameter_list,
        auroc_list,
        auprc_list,
        auroc_list_svm,
        auprc_list_svm,
        len(N),
    )

    with open(result_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))


def trunc(values, decs=0):
    return np.trunc(values * 10**decs) / (10**decs)


def create_result_path(method_name, bias_type, data_set_name):
    """The function creates the save path and makes the directory.

    :param method: Method name
    :param bias_type: Bias type name
    :param data_set_name: Data set name
    :return: The result path
    """
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    result_path = result_path / method_name / "downstream" / data_set_name / bias_type
    result_path.mkdir(exist_ok=True, parents=True)
    return result_path
