from pathlib import Path
import numpy as np
import statsmodels.api as sm


def logistic_regression(allensbach_gbs, sample_weights):
    """Performs a logisti regression

    :param allensbach_gbs: Allensbach and gbs data set
    :param weights: Sample weights
    :return: p values
    """
    sample_weights = (np.array(sample_weights) / np.sum(sample_weights)) * len(
        sample_weights
    )
    y = allensbach_gbs["Wahlteilnahme"]
    X = allensbach_gbs["Resilienz"]
    X = sm.add_constant(X)

    model_all = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(sm.families.links.Logit()),
        freq_weights=sample_weights,
    )
    results_weighted = model_all.fit()
    lr_pvalue_weighted = results_weighted.pvalues["Resilienz"]

    return lr_pvalue_weighted


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
    :param remaining_samples_list: List of remaining samples
    :param auroc_list: List of auroc values
    :param auprc_list: List of auprc values
    :param number_of_samples: Number of samples in the original data set
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

    :param columns: Column names
    :param weighted_mmds_list: List of result mmds
    :param biases_list: List of result relative biases
    :param wasserstein_parameter_list: List of wasserstein distances
    :param remaining_samples_list: List of remaining samples
    :param auroc_list: List of auroc values
    :param auprc_list: List of auprc values
    :param number_of_samples: Number of samples in the original data set
    :return: The result dictionary
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
    bias_fraction=0.0,
    prefix="../..",
):
    """The function creates the save path and makes the directory.

    :param method: Method name
    :param bias_type: Bias type name
    :param data_set_name: Data set name
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


def write_result_dict_test_set_lipidomics(
    rf_auroc_cal_list,
    rf_auprc_cal_list,
    rf_auroc_set2_list,
    rf_auprc_set2_list,
    dropped_samples_list,
    number_of_samples,
):
    """Creates the result dictionary

    :param columns: Column names
    :param weighted_mmds_list: List of result mmds
    :param biases_list: List of result relative biases
    :param wasserstein_parameter_list: List of wasserstein distances
    :param remaining_samples_list: List of remaining samples
    :param auroc_list: List of auroc values
    :param auprc_list: List of auprc values
    :param number_of_samples: Number of samples in the original data set
    :return: The result dictionary
    """
    result_dict = {
        "random forest auroc cal": {
            "mean": np.mean(rf_auroc_cal_list),
            "sd": np.std(rf_auroc_cal_list),
        },
        "random forest auprc cal": {
            "mean": np.mean(rf_auprc_cal_list),
            "sd": np.std(rf_auprc_cal_list),
        },
        "random forest auroc set2": {
            "mean": np.mean(rf_auroc_set2_list),
            "sd": np.std(rf_auroc_set2_list),
        },
        "random forest auprc set2": {
            "mean": np.mean(rf_auprc_set2_list),
            "sd": np.std(rf_auprc_set2_list),
        },
        "dropped_samples": {
            "mean": np.mean(dropped_samples_list),
            "std": np.std(dropped_samples_list),
        },
        "all_samples": number_of_samples,
    }

    return result_dict