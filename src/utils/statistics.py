from pathlib import Path
import numpy as np


def write_result_dict(
    columns,
    weighted_mmds_list,
    biases_list,
    wasserstein_distance_list,
):
    """Creates the result dictionary

    :param columns: Column names
    :param weighted_mmds_list: List of result mmds
    :param biases_list: List of result relative biases
    :param wasserstein_parameter_list: List of wasserstein distances
    :return: The result dictionary
    """
    result_dict = {
        "MMDs": {
            "mean": np.nanmean(weighted_mmds_list),
            "sd": np.nanstd(weighted_mmds_list),
        },
    }

    mean_biases = np.nanmean(biases_list, axis=0)
    sd_biases = np.nanstd(biases_list, axis=0)

    mean_wasserstein_distances = np.mean(wasserstein_distance_list, axis=0)
    mean_wasserstein_distances = np.std(wasserstein_distance_list, axis=0)

    for index, column in enumerate(columns):
        result_dict[f"{column}_relative_bias"] = {
            "bias mean": mean_biases[index],
            "bias sd": sd_biases[index],
        }
        result_dict[f"{column}_wasserstein_distance"] = {
            "bias mean": mean_wasserstein_distances[index],
            "bias sd": mean_wasserstein_distances[index],
        }

    return result_dict


def write_result_dict_test_set(
    rf_auroc_list,
    rf_auprc_list,
    svm_pad_list,
    rf_domain_auroc_list,
    dropped_samples_list,
    number_of_samples,
):
    """Creates the result dictionary

    :param rf_auroc_list: List with AUROC values
    :param rf_auprc_list: List with AUPRC values
    :param svm_pad_list: List with pad values
    :param rf_domain_auroc_list: List with domain AUROC values
    :param dropped_samples_list: List with dropped samples values
    :param number_of_samples: List with number of samples in non-representative data set
    :return: Filled dictionary
    """
    result_dict = {
        "random forest auroc": {
            "mean": np.mean(rf_auroc_list),
            "sd": np.std(rf_auroc_list),
        },
        "random forest auprc": {
            "mean": np.mean(rf_auprc_list),
            "sd": np.std(rf_auprc_list),
        },
        "svm pad": {
            "mean": np.mean(svm_pad_list),
            "sd": np.std(svm_pad_list),
        },
        "rf domain auroc": {
            "mean": np.mean(rf_domain_auroc_list),
            "sd": np.std(rf_domain_auroc_list),
        },
        "dropped_samples": {
            "mean": np.mean(dropped_samples_list),
            "std": np.std(dropped_samples_list),
        },
        "all_samples": number_of_samples,
    }

    return result_dict


def create_result_path(
    method_name,
    bias_type,
    data_set_name,
    experiment_name="downstream",
    bias_fraction="",
    prefix="../..",
):
    """The function creates the save path and makes the directory.

    :param method: Method name
    :param bias_type: Bias type name
    :param data_set_name: Data set name
    :param experiment: Experiment name
    :param bias_fraction: Bias strength
    :param prefix: additional prefix for save path
    :return: The result path
    """
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, f"{prefix}/results")
    result_path = (
        result_path
        / experiment_name
        / data_set_name
        / bias_type
        / str(bias_fraction)
        / method_name
    )
    result_path.mkdir(exist_ok=True, parents=True)
    return result_path
