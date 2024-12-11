import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path
from cycler import cycler

sns.set_theme(style="ticks")
# Create a custom line style and color cycler
default_cycle = cycler(
    "linestyle",
    [
        "solid",
        "dotted",
        "dashdot",
        "dashed",
        (5, (10, 3)),
        (0, (3, 1, 1, 1, 1, 1)),
        (5, (10, 3)),
    ],
) + cycler(color=sns.color_palette()[:7])

line_styles = [
    "solid",
    "dotted",
    "dashdot",
    "dashed",
    (5, (10, 3)),
    (0, (3, 1, 1, 1, 1, 1)),
    (5, (10, 3)),
]


def plot_cumulative_distribution_function(
    N,
    R,
    file_name: str,
    sample_weights,
    method_name,
    wide=True,
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
            N,
            x=column_name,
            weights=sample_weights,
            label=method_name,
            linestyle="dotted",
        )
        plt.legend()
        plt.savefig(plot_directory / f"{column_name}.pdf")
        plt.clf()


def plot_feature_histograms(N, R, file_name, bins, sample_weights, method_name):
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
    fig, ax = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(10, 5))
    for column_name in N.columns:
        sns.histplot(
            N, x=column_name, bins=bins, stat="probability", kde=True
        ).set_title("GBS")
        fig.savefig(plot_directory / f"{column_name}_gbs.pdf")
        ax.clear()
        sns.histplot(
            R, x=column_name, bins=bins, stat="probability", kde=True
        ).set_title("Allensbach")
        fig.savefig(plot_directory / f"{column_name}_allensbach.pdf")
        ax.clear()
        sns.histplot(
            N,
            x=column_name,
            weights=sample_weights,
            bins=bins,
            stat="probability",
            kde=True,
        ).set_title(method_name)
        fig.savefig(plot_directory / f"{column_name}_{method_name}.pdf")
        ax.clear()
    plt.clf()


def plot_sample_weights(weights, path, iteration, title="", bins=100):
    """Plot the weights for a method

    :param weights: Weights
    :param path: Save path
    :param iteration: From which iteration are the weights
    :param title: Title for the plot, defaults to ""
    :param bins: How many bin are used, defaults to 25
    """
    path.mkdir(exist_ok=True)
    sns.histplot(x=weights, bins=bins).set_title(title)
    plt.savefig(f"{path}/weights_{iteration}.pdf", bbox_inches="tight")
    plt.clf()


def plot_statistical_analysis(
    bins: int,
    N: np.ndarray,
    R: np.ndarray,
    visualisation_path: Path,
    sample_weights: list[float],
    method_name: str = "",
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
        N,
        R,
        visualisation_path,
        sample_weights,
        method_name,
    )
    plot_feature_histograms(
        N,
        R,
        visualisation_path,
        bins,
        sample_weights,
        method_name,
    )


def plot_auc_average(
    auc_dicts,
    drop,
    file_name,
    number_of_samples,
    mrs_iterations,
    wide=True,
    n_ticks=5,
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

    min_length = np.min(
        [
            [len(auroc_list) for auroc_list in auroc_lists.values()]
            for auroc_lists in auc_dicts
        ]
    )
    min_number_of_samples = np.min(number_of_samples)
    x_labels = list(range(min_number_of_samples, 0, -drop))[:min_length]
    for i, hyperparameter in enumerate(auc_dicts[0].keys()):
        mmd_list = []
        for dictionary in auc_dicts:
            dictionary = {float(k): v for k, v in dictionary.items()}
            mmd_list.append(dictionary[float(hyperparameter)][:min_length])

        mean = np.mean(mmd_list, axis=0)
        std = np.std(mmd_list, axis=0)
        aucs_upper = np.minimum(mean + std, 1)
        aucs_lower = np.maximum(mean - std, 0)

        sns.lineplot(
            x=x_labels, y=mean, label=str(hyperparameter), linestyle=line_styles[i]
        )
        plt.fill_between(x_labels, aucs_lower, aucs_upper, alpha=0.3)

    random_line = len(x_labels) * [0.5]
    plt.plot(
        x_labels,
        random_line,
        color="black",
        linestyle="--",
        label="Random",
    )
    plt.ylabel("AUROC")
    plt.xlabel("Number of Remaining Samples")
    step_size = len(x_labels) // n_ticks
    x_ticks = x_labels[::-step_size]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()

    plt.savefig(f"{file_name}_wide_{wide}.pdf")
    plt.close()


def plot_value_average(
    mmds_dicts,
    drop,
    file_name,
    mrs_iterations,
    number_of_samples,
    n_ticks=5,
    ylabel="Maximum Mean Discrepancy",
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
    min_length = np.min(
        [
            [len(auroc_list) for auroc_list in auroc_lists.values()]
            for auroc_lists in mmds_dicts
        ]
    )
    min_number_of_samples = np.min(number_of_samples)
    x_labels = list(range(min_number_of_samples, 0, -drop))[:min_length]
    for i, hyperparameter in enumerate(mmds_dicts[0].keys()):
        mmd_list = []
        for dictionary in mmds_dicts:
            dictionary = {float(k): v for k, v in dictionary.items()}
            mmd_list.append(dictionary[float(hyperparameter)][:min_length])

        mean = np.mean(mmd_list, axis=0)
        std = np.std(mmd_list, axis=0)
        mmds_upper = mean + std
        mmds_lower = np.maximum(mean - std, 0)
        sns.lineplot(
            x=x_labels, y=mean, label=str(hyperparameter), linestyle=line_styles[i]
        )
        plt.fill_between(x_labels, mmds_lower, mmds_upper, alpha=0.3)

    plt.ylabel(ylabel)
    plt.xlabel("Number of Remaining Samples")
    step_size = len(x_labels) // n_ticks
    x_ticks = x_labels[::-step_size]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()

    plt.savefig(f"{file_name}.pdf")
    plt.close()


def plot_rocs_mrs(mean_roc_dict, file_name):
    """Plots rocs

    :param roc_list: Roc list
    :param file_name: File name for the plot
    """
    plt.rc("")
    plt.rc("axes", prop_cycle=default_cycle)
    for hyperparameter, mean_roc in mean_roc_dict.items():
        for i, (fper, tper, std, deleted_elements) in enumerate(mean_roc):
            tpfrs_higher = np.minimum(tper + std, 1)
            tpfrs_lower = np.maximum(tper - std, 0)
            sns.lineplot(x=fper, y=tper, label=f"{deleted_elements} samples removed", linestyle=line_styles[i])
            plt.fill_between(fper, tpfrs_lower, tpfrs_higher, alpha=0.3)
        sns.lineplot(
            x=[0, 1],
            y=[0, 1],
            color="black",
            linestyle="--",
            linewidth=1,
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(f"{file_name}_{hyperparameter}.pdf")
        plt.close()


def plot_relative_bias(
    relative_bias_dict,
    file_name,
    mrs_iterations,
    number_of_samples,
    drop,
    n_ticks=5,
):
    """Plots relative biases

    :param mean_relative_bias_list: Mean relative biases
    :param std_relative_bias_list: Standard deviation for the relative biases
    :param file_name: File name for the plot
    :param mrs_iterations: Iteration in which mrs stopped
    :param number_of_samples: Number of samples in the original data set
    :param drop: Number of dropped elements per iteration
    """
    min_length = np.min(
        [
            [len(auroc_list) for auroc_list in auroc_lists.values()]
            for auroc_lists in relative_bias_dict
        ]
    )
    min_number_of_samples = np.min(number_of_samples)
    x_labels = list(range(min_number_of_samples, 0, -drop))[:min_length]
    for i, hyperparameter in enumerate(relative_bias_dict[0].keys()):
        mmd_list = []
        for dictionary in relative_bias_dict:
            dictionary = {float(k): v for k, v in dictionary.items()}
            mmd_list.append(dictionary[float(hyperparameter)][:min_length])

        mean = np.mean(mmd_list, axis=0)
        std = np.std(mmd_list, axis=0)
        mmds_upper = mean + std
        mmds_lower = np.maximum(mean - std, 0)
        sns.lineplot(
            x=x_labels, y=mean, label=str(hyperparameter), linestyle=line_styles[i]
        )
        plt.fill_between(x_labels, mmds_lower, mmds_upper, alpha=0.3)

    plt.xlabel("Number of Remaining Samples")
    plt.ylabel("Relative Bias")
    step_size = len(x_labels) // n_ticks
    x_ticks = x_labels[::-step_size]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()

    plt.savefig(f"{file_name}.pdf")
    plt.close()
