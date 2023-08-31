import numpy as np
import statsmodels.api as sm


def logistic_regression(allensbach_gbs, weights):
    """Performs a logisti regression

    :param allensbach_gbs: Allensbach and gbs data set
    :param weights: Sample weights
    :return: p values
    """
    weights = weights * len(weights)
    y = allensbach_gbs["Wahlteilnahme"]
    X = allensbach_gbs["Resilienz"]
    X = sm.add_constant(X)
    model_gbs = sm.GLM(y, X, family=sm.families.Binomial(sm.families.links.Logit()))
    results_gbs = model_gbs.fit()
    lr_pvalue_gbs = results_gbs.pvalues[1]

    model_all = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(sm.families.links.Logit()),
        freq_weights=weights,
    )
    results_weighted = model_all.fit()
    lr_pvalue_weighted = results_weighted.pvalues[1]

    return lr_pvalue_gbs, lr_pvalue_weighted


def write_result_dict(
    columns,
    weighted_mmds_list,
    biases_list,
    wasserstein_parameter_list,
    remaining_samples_list,
    auroc_list,
    auprc_list,
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
        "MMDs": {
            "mean": np.nanmean(weighted_mmds_list),
            "sd": np.nanstd(weighted_mmds_list),
        },
        "auroc": {
            "mean": np.nanmean(auroc_list),
            "sd": np.nanstd(auroc_list),
        },
        "auprc": {
            "mean": np.nanmean(auprc_list),
            "sd": np.nanstd(auprc_list),
        },
        "remaining samples": {
            "mean": np.nanmean(remaining_samples_list),
            "sd": np.nanstd(remaining_samples_list),
        },
        "all_samples": number_of_samples,
    }

    mean_biases = np.nanmean(biases_list, axis=0)
    sd_biases = np.nanstd(biases_list, axis=0)
    mean_wasserstein = np.nanmean(wasserstein_parameter_list, axis=0)
    sd_wasserstein = np.nanstd(wasserstein_parameter_list, axis=0)
    for index, column in enumerate(columns):
        result_dict[f"{column}_relative_bias"] = {
            "bias mean": mean_biases[index],
            "bias sd": sd_biases[index],
            "wasserstein mean": mean_wasserstein[index],
            "wasserstein sd": sd_wasserstein[index],
        }

    return result_dict
