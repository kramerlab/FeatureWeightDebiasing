import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path
from cycler import cycler

sns.set_theme(style="ticks")


def plot_cumulative_distribution_function(
    N, R, file_name: str, weights_one, weights_two, method_one, method_two, wide=True
):
    """Plots the cumulative distribution functions of two methods.

    :param N: Values of the first method
    :param R: Values of the second method
    :param file_name: File name for the plot
    :param weights_one: Weights for the first method
    :param weights_two: Weights for the second method
    :param method_one: Name of the first method
    :param method_two: Name of the second method
    :param wide: If true, plots the data in a wide format, defaults to True
    """
    if wide:
        plt.figure(figsize=(10, 5))
    plot_directory = file_name / "cumulative_distributions"
    plot_directory.mkdir(exist_ok=True)
    for column_name in N.columns:
        sns.ecdfplot(N, x=column_name, label="GBS")
        sns.ecdfplot(R, x=column_name, label="Allensbach", linestyle="dashed")
        sns.ecdfplot(
            N, x=column_name, weights=weights_one, label=method_one, linestyle="dotted"
        )
        sns.ecdfplot(
            N, x=column_name, weights=weights_two, label=method_two, linestyle="dashdot"
        )
        plt.legend()
        plt.savefig(plot_directory / f"{column_name}.pdf")
        plt.clf()


def plot_feature_histograms(
    N, R, file_name, bins, weights_one, weights_two, method_one, method_two
):
    """Plot and saves the feature histograms of two methods

    :param N: Features of the first data set
    :param R: Features of the second data set
    :param file_name: File name of the plot
    :param bins: How many bins are used
    :param weights_one: Weights for the first method
    :param weights_two: _Weights for the second method
    :param method_one: Name of the first method
    :param method_two: Name of the second method
    """
    plot_directory = file_name / "histograms"
    plot_directory.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(10, 5))
    for column_name in N.columns:
        sns.histplot(
            N, x=column_name, ax=ax[0], bins=bins, stat="probability", kde=True
        ).set_title("GBS")
        sns.histplot(
            R, x=column_name, ax=ax[1], bins=bins, stat="probability", kde=True
        ).set_title("Allensbach")
        sns.histplot(
            N,
            x=column_name,
            weights=weights_one,
            ax=ax[2],
            bins=bins,
            stat="probability",
            kde=True,
        ).set_title(method_one)
        sns.histplot(
            N,
            x=column_name,
            weights=weights_two,
            ax=ax[3],
            bins=bins,
            stat="probability",
            kde=True,
        ).set_title(method_two)
        fig.savefig(plot_directory / f"{column_name}.pdf")
        [axis.clear() for axis in ax]
    plt.clf()


def plot_weights(weights, path, iteration, title="", bins=25):
    """Plot the weights for a method

    :param weights: Weights
    :param path: Save path
    :param iteration: From which iteration are the weights
    :param title: Title for the plot, defaults to ""
    :param bins: How many bin are used, defaults to 25
    """
    path.mkdir(exist_ok=True)
    weights = weights / sum(weights)
    sns.histplot(x=weights, bins=bins).set_title(title)
    same_weights_positition = 1 / len(weights)
    plt.axvline(same_weights_positition, color="k")
    plt.savefig(f"{path}/weights_{iteration}.pdf", bbox_inches="tight")
    plt.clf()


def plot_statistical_analysis(
    bins: int,
    N: np.ndarray,
    R: np.ndarray,
    visualisation_path: Path,
    weights_one: list[float],
    weights_two: list[float],
    method_one: str = "",
    method_two: str = "",
):
    """Plots the statistical analysis for two methods

    :param bins: How many bins are used
    :param N: Features of the first method
    :param R: Features of the second method
    :param visualisation_path: Save path
    :param weights_one: Weights for the first method
    :param weights_two: Weights for the second method
    :param method_one: Name of the first method, defaults to ""
    :param method_two: Name of the second method, defaults to ""
    """
    plot_cumulative_distribution_function(
        N, R, visualisation_path, weights_one, weights_two, method_one, method_two
    )
    plot_feature_histograms(
        N, R, visualisation_path, bins, weights_one, weights_two, method_one, method_two
    )


def plot_results_with_variance(
    metric_list: list[float], visualisation_path: Path, suffix: str = "", metric="MMD"
):
    """Plots a mean value line with variance

    :param metric_list: Mean values
    :param visualisation_path: Save path
    :param suffix: Suffix for the file name, defaults to ""
    :param metric: Metric name, defaults to "MMD"
    """
    mean_metric = np.nanmean(metric_list, axis=0)
    sd_metric = np.nanstd(metric_list, axis=0)
    plt.plot(range(len(mean_metric)), mean_metric, color="blue")
    plt.fill_between(
        x=range(len(mean_metric)),
        y1=mean_metric - sd_metric,
        y2=mean_metric + sd_metric,
        color="blue",
        alpha=0.5,
    )
    plt.ylabel(f"Weighted {metric}")
    plt.xlabel("Pass")
    plt.savefig(Path(visualisation_path) / f"weighted_{metric}_{suffix}.pdf")
    plt.clf()


def plot_auc_average(
    auc_score,
    std_aucs,
    drop,
    file_name,
    number_of_samples,
    mrs_iterations,
    wide=True,
):
    """Plots average aurocs with variance

    :param auc_score: Mean auroc values
    :param std_aucs: Standard deviation for the aurocs
    :param drop: How many elements were dropped each iteration
    :param file_name: File name for the plot
    :param number_of_samples: Number of samples in the original data set
    :param mrs_iterations: In which iteration where the mrs returned
    :param wide: If true, plot the data in wide format, defaults to True
    """
    if wide:
        plt.figure(figsize=(12.8, 4.8))

    aucs_upper = np.minimum(auc_score + std_aucs, 1)
    aucs_lower = np.maximum(auc_score - std_aucs, 0)
    stop = number_of_samples - ((auc_score.size) * drop)
    x_labels = list(range(number_of_samples, stop, -drop))
    plt.fill_between(x_labels, aucs_lower, aucs_upper, color="blue", alpha=0.2)
    plt.plot(x_labels, auc_score, color="blue", linestyle="-")
    random_line = len(x_labels) * [0.5]
    plt.plot(
        x_labels,
        random_line,
        color="black",
        linestyle="--",
        label="Random",
    )
    plt.ylabel("AUROC")
    mrs_iterations = number_of_samples - (np.array(mrs_iterations))
    minimum = min(0.5, np.min(aucs_lower)) - 0.01
    maximum = plt.gca().get_ylim()[1] + 0.01
    plt.margins(0.05, 0)
    plt.vlines(
        mrs_iterations,
        minimum,
        maximum,
        colors="black",
        linestyles="solid",
    )
    x_ticks = list(range(number_of_samples, stop, -(number_of_samples // 5))) + [4]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()
    plt.xlabel("Number of Remaining Samples")
    xlim = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xlim(xlim)
    plt.xticks(list(mrs_iterations))
    [tick.set_color("blue") for tick in plt.gca().get_xticklabels()]
    plt.savefig(f"{file_name}.pdf")
    plt.close()


def plot_mmds_average(
    mmds, std, drop, mmd_iteration, file_name, mrs_iterations, number_of_samples
):
    """Plots the mean mmds with variance

    :param mmds: Mean mmd values
    :param std: Standard deviation for the mmd values
    :param drop: How many elements were dropped in each iteration
    :param mmd_iteration: In which iteration where the mmd computed
    :param file_name: Save file name
    :param mrs_iterations: In whih iterations were the mrs' returned
    :param number_of_samples: How many samples were in the original data set
    """
    mmds_upper = mmds + std
    mmds_lower = np.maximum(mmds - std, 0)
    stop = number_of_samples - ((mmds.size) * drop)
    x_labels = range(number_of_samples, stop, -(drop * mmd_iteration))
    plt.fill_between(x_labels, mmds_lower, mmds_upper, color="blue", alpha=0.2)
    plt.plot(x_labels, mmds, linestyle="-")
    minimum = np.min(mmds_lower) - 0.001
    maximum = plt.gca().get_ylim()[1] + 0.001
    plt.margins(0.05, 0)
    mrs_iterations = number_of_samples - np.array(mrs_iterations)
    plt.vlines(mrs_iterations, minimum, maximum, colors="black", linestyles="solid")
    plt.ylabel("Maximum Mean Discrepancy")
    plt.xlabel("Number of Remaining Samples")
    x_ticks = list(range(number_of_samples, stop, -(number_of_samples // 5))) + [4]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()
    xlim = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xlim(xlim)
    plt.xticks(mrs_iterations)
    [tick.set_color("blue") for tick in plt.gca().get_xticklabels()]
    plt.savefig(f"{file_name}.pdf")
    plt.close()


def plot_experiment_comparison_auc(
    auc_score_mrs,
    std_aucs_mrs,
    auc_score_experiment,
    std_aucs_experiment,
    experiment_label,
    drop,
    file_name,
    number_of_samples,
):
    """Plots the mean auroc with variance

    :param auc_score_mrs: Mean auroc values
    :param std_aucs_mrs: Standard deviation of auroc values
    :param auc_score_experiment: Mean auroc values for the mrs variant
    :param std_aucs_experiment: Standard deviation of auroc values for the mrs variant
    :param experiment_label: Name of the mrs variant
    :param drop: Number of dropped samples per iteration
    :param file_name: File name for the plot
    :param number_of_samples: Number of sample sin the original data set
    """
    aucs_upper = np.minimum(auc_score_mrs + std_aucs_mrs, 1)
    aucs_lower = np.maximum(auc_score_mrs - std_aucs_mrs, 0)

    aucs_upper_experiment = np.minimum(auc_score_experiment + std_aucs_experiment, 1)
    aucs_lower_experiment = np.maximum(auc_score_experiment - std_aucs_experiment, 0)

    stop = number_of_samples - ((auc_score_mrs.size) * drop)
    x_labels = list(range(number_of_samples, stop, -drop))

    plt.fill_between(x_labels, aucs_lower, aucs_upper, color="blue", alpha=0.2)
    plt.plot(x_labels, auc_score_mrs, color="blue", linestyle="-", label="MRS")

    plt.fill_between(
        x_labels,
        aucs_lower_experiment,
        aucs_upper_experiment,
        color="orange",
        alpha=0.2,
    )
    plt.plot(
        x_labels,
        auc_score_experiment,
        linestyle=":",
        color="orange",
        label=experiment_label,
    )

    plt.plot(len(auc_score_mrs) * drop * [0.5], color="black", linestyle="--")

    plt.ylabel("AUROC")
    plt.xlabel("Number of Remaining Samples")
    plt.xticks(list(range(number_of_samples, 0, -100)) + [4])
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig(f"{file_name}.pdf")


def plot_experiment_comparison_mmd(
    mean_mmd,
    std_mmd,
    mean_mmd_experiment,
    std_mmd_experiment,
    experiment_label,
    drop,
    mmd_iteration,
    file_name,
    number_of_samples,
):
    """Plots the mmd of a comparison with a variant

    :param mean_mmd: Mean mmds of mrs
    :param std_mmd: Standard deviation for the mmds of mrs
    :param mean_mmd_experiment: Mean mmds of the mrs variant
    :param std_mmd_experiment: Standard deviation for the mmds of the mrs variant
    :param experiment_label: Name of the mrs variant
    :param drop: Number of dropped samples per iteration
    :param mmd_iteration: Number of iteration the mmd was computed
    :param file_name: File name for the plot
    :param number_of_samples: Number of sample sin the original data set
    """
    mmd_upper = np.minimum(mean_mmd + std_mmd, 1)
    mmd_lower = np.maximum(mean_mmd - std_mmd, 0)

    mmd_upper_experiment = np.minimum(mean_mmd_experiment + std_mmd_experiment, 1)
    mmd_lower_experiment = np.maximum(mean_mmd_experiment - std_mmd_experiment, 0)

    stop = number_of_samples - ((mean_mmd.size) * drop)
    x_labels = range(number_of_samples, stop, -(drop * mmd_iteration))

    plt.fill_between(x_labels, mmd_lower, mmd_upper, color="blue", alpha=0.2)
    plt.plot(x_labels, mean_mmd, color="blue", linestyle="-", label="MRS")

    plt.fill_between(
        x_labels, mmd_lower_experiment, mmd_upper_experiment, color="orange", alpha=0.2
    )
    plt.plot(
        x_labels,
        mean_mmd_experiment,
        linestyle=":",
        color="orange",
        label=experiment_label,
    )

    plt.ylabel("Maximum Mean Discrepancy")
    plt.xlabel("Number of Remaining Samples")
    plt.xticks(list(range(number_of_samples, 0, -100)) + [4])
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig(f"{file_name}.pdf")
    plt.close()


# Create a custom line style and color cycler
default_cycle = cycler(
    "linestyle",
    [
        ":",
        "-.",
        (0, (3, 5, 1, 5, 1, 5)),
        "-",
    ],
) + cycler(color=["blue", "orange", "orangered", "cyan"])


def plot_rocs(roc_list, file_name):
    """Plots rocs

    :param roc_list: Roc list
    :param file_name: File name for the plot
    """
    plt.rc("")
    plt.rc("axes", prop_cycle=default_cycle)
    for fper, tper, std, deleted_elements in roc_list:
        tpfrs_higher = np.minimum(tper + std, 1)
        tpfrs_lower = np.maximum(tper - std, 0)
        plt.plot(fper, tper, label=f"{deleted_elements} samples removed")
        plt.fill_between(fper, tpfrs_lower, tpfrs_higher, alpha=0.2)
    plt.plot(
        [0, 1], [0, 1], color="black", linestyle="--", label="Random", linewidth=0.8
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"{file_name}.pdf")
    plt.close()


def plot_relative_bias(
    mean_relative_bias_list,
    std_relative_bias_list,
    file_name,
    mrs_iterations,
    number_of_samples,
    drop,
):
    """Plots relative biases

    :param mean_relative_bias_list: Mean relative biases
    :param std_relative_bias_list: Standard deviation for the relative biases
    :param file_name: File name for the plot
    :param mrs_iterations: Iteration in which mrs stopped
    :param number_of_samples: Number of samples in the original data set
    :param drop: Number of dropped elements per iteration
    """
    plt.xlabel("Number of Remaining Samples")
    plt.ylabel("Relative Bias")
    ratio_upper = mean_relative_bias_list + std_relative_bias_list
    ratio_lower = (mean_relative_bias_list - std_relative_bias_list).clip(min=0)
    stop = number_of_samples - (mean_relative_bias_list.size * drop)
    x_labels = list(range(number_of_samples, stop, -drop))

    plt.plot(
        x_labels,
        mean_relative_bias_list,
        linestyle="-",
        color="blue",
    )
    plt.fill_between(x_labels, ratio_lower, ratio_upper, color="blue", alpha=0.2)
    minimum = plt.gca().get_ylim()[0]
    maximum = plt.gca().get_ylim()[1]
    plt.margins(0.05, 0)
    mrs_iterations = number_of_samples - np.array(mrs_iterations)
    plt.vlines(mrs_iterations, minimum, maximum, colors="black", linestyles="solid")
    x_ticks = list(range(number_of_samples, stop, -(number_of_samples // 5))) + [
        stop + drop
    ]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()
    xlim = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xlim(xlim)
    plt.xticks(mrs_iterations)
    [tick.set_color("blue") for tick in plt.gca().get_xticklabels()]
    plt.savefig(f"{file_name}.pdf")
    plt.close()
