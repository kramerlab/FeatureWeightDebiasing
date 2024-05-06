import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="ticks")
line_styles = [
    "solid",
    "dotted",
    "dashdot",
    "dashed",
    (5, (10, 3)),
]


def plot_budget_comparison_auroc(
    auroc_dictionary,
    number_of_samples,
    drop,
    file_name,
):
    n_auroc_values = len(auroc_dictionary[list(auroc_dictionary.keys())[0]])
    stop = number_of_samples - (n_auroc_values * drop)
    x_labels = list(range(number_of_samples, stop, -drop))
    for i, (budget, aurocs) in enumerate(auroc_dictionary.items()):
        sns.lineplot(x=x_labels, y=aurocs, label=str(budget), linestyle=line_styles[i])
    plt.plot(n_auroc_values * drop * [0.5], color="black", linestyle="--")
    plt.ylabel("Feature Weighted AUROC")
    plt.xlabel("Number of Remaining Samples")
    x_ticks = list(
        range(number_of_samples, stop, -((number_of_samples - (stop + drop)) // 4))
    ) + [stop + drop]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()

    plt.savefig(f"{file_name}.pdf")
    plt.close()


def plot_feature_weights(feature_weights_list, save_path):
    for budget, feature_weights in feature_weights_list.items():
        budget_path = save_path / str(budget)
        budget_path.mkdir(parents=True, exist_ok=True)
        for i, feature_weight in enumerate(feature_weights):
            sns.barplot(feature_weight)
            plt.savefig(budget_path / f"feature_weights_{i}.pdf")
            plt.close()


def plot_feature_importance(feature_importance_list, save_path):
    for i, feature_importance in enumerate(feature_importance_list):
        sns.barplot(feature_importance)
        plt.savefig(save_path / f"feature_importance_{i}.pdf")
        plt.close()


def plot_budget_comparison_auroc_mean(
    auroc_list_of_dictionaries,
    number_of_samples,
    drop,
    file_name,
):
    n_auroc_values = len(
        auroc_list_of_dictionaries[0][list(auroc_list_of_dictionaries[0].keys())[0]]
    )
    stop = number_of_samples - (n_auroc_values * drop)
    x_labels = list(range(number_of_samples, stop, -drop))
    for i, budget in enumerate(auroc_list_of_dictionaries[0].keys()):
        auroc_list = []
        for dictionary in auroc_list_of_dictionaries:
            auroc_list.append(dictionary[budget])
        mean_aurocs = np.mean(auroc_list, axis=0)
        std_aurocs = np.std(auroc_list, axis=0)
        ratio_upper = mean_aurocs + std_aurocs
        ratio_lower = (mean_aurocs - std_aurocs).clip(min=0)
        sns.lineplot(
            x=x_labels, y=mean_aurocs, label=str(budget), linestyle=line_styles[i]
        )
        plt.fill_between(x_labels, ratio_lower, ratio_upper, alpha=0.2)
    plt.plot(n_auroc_values * drop * [0.5], color="black", linestyle="--")
    plt.ylabel("Feature Weighted AUROC")
    plt.xlabel("Number of Remaining Samples")
    x_ticks = list(
        range(number_of_samples, stop, -((number_of_samples - (stop + drop)) // 4))
    ) + [stop + drop]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()

    plt.savefig(f"{file_name}.pdf")
    plt.close()
