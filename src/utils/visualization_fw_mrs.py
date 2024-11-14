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
    (0, (3, 1, 1, 1, 1, 1)),
    (5, (10, 3)),
]


def plot_feature_weights(feature_weights_list, budget, save_path):
    budget_path = save_path / str(budget)
    budget_path.mkdir(parents=True, exist_ok=True)
    sns.barplot(feature_weights_list)
    plt.savefig(budget_path / f"feature_weights.pdf")
    plt.close()


def plot_feature_importance(feature_importance_list, save_path):
    for i, feature_importance in enumerate(feature_importance_list):
        sns.barplot(feature_importance)
        plt.savefig(save_path / f"feature_importance_{i}.pdf")
        plt.close()


def plot_budget_comparison_auroc(
    auroc_dictionaries,
    number_of_samples,
    drop,
    file_name,
    wide=True,
    n_ticks=5,
):
    if wide:
        plt.figure(figsize=(10, 5))
    stop = 0
    max_number_of_samples = np.max(number_of_samples)
    x_labels = list(range(max_number_of_samples, stop, -drop))[:-1]
    for i, (budget, aurocs) in enumerate(auroc_dictionaries.items()):
        sns.lineplot(
            x=x_labels[: len(aurocs)],
            y=aurocs,
            label=str(budget),
            linestyle=line_styles[i],
        )
    plt.plot([max_number_of_samples, x_labels[-1]], [0.5, 0.5], color="black", linestyle="--")
    plt.ylabel("Feature Weighted AUROC")
    plt.xlabel("Number of Remaining Samples")
    step_size = len(x_labels) // n_ticks
    x_ticks = x_labels[::-step_size]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()

    plt.savefig(f"{file_name}.pdf")
    plt.close()


def plot_budget_comparison_auroc_mean(
    auroc_list_of_dictionaries,
    number_of_samples,
    drop,
    file_name,
    wide=True,
    n_ticks=5,
):
    if wide:
        plt.figure(figsize=(10, 5))
    min_length = np.min(
        [
            [len(auroc_list) for auroc_list in auroc_lists.values()]
            for auroc_lists in auroc_list_of_dictionaries
        ]
    )
    min_number_of_samples = np.min(number_of_samples)
    x_labels = list(range(min_number_of_samples, 0, -drop))[:min_length]
    for i, budget in enumerate(auroc_list_of_dictionaries[0].keys()):
        auroc_list = []
        for dictionary in auroc_list_of_dictionaries:
            dictionary = {float(k): v for k, v in dictionary.items()}
            auroc_list.append(dictionary[float(budget)][:min_length])
        mean_aurocs = np.mean(auroc_list, axis=0)
        std_aurocs = np.std(auroc_list, axis=0)
        ratio_upper = mean_aurocs + std_aurocs
        ratio_lower = (mean_aurocs - std_aurocs).clip(min=0)
        sns.lineplot(
            x=x_labels, y=mean_aurocs, label=str(budget), linestyle=line_styles[i]
        )
        plt.fill_between(x_labels, ratio_lower, ratio_upper, alpha=0.2)
    plt.plot([min_number_of_samples, x_labels[-1]], [0.5, 0.5], color="black", linestyle="--")
    plt.ylabel("Feature Weighted AUROC")
    plt.xlabel("Number of Remaining Samples")
    step_size = len(x_labels) // n_ticks
    x_ticks = x_labels[::-step_size]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()

    plt.savefig(f"{file_name}.pdf")
    plt.close()


def visualize_boxplot(
    values_dict,
    y_label,
    file_name="",
):
    tmp_dict = {}
    tmp_dict.update(values_dict)
    ax = sns.boxplot(data=tmp_dict)
    ax.set_ylabel(y_label)

    plt.savefig(f"{file_name}.pdf", bbox_inches="tight")
    plt.close()


def visualize_heatmap(
    values_dict,
    y_label,
    file_name="",
):
    input_values = [
        [np.mean(values) for values in dicts.values()] for dicts in values_dict.values()
    ]
    x_ticks = [values.keys() for values in values_dict.values()][0]
    y_ticks = values_dict.keys()
    ax = sns.heatmap(
        data=input_values,
        xticklabels=x_ticks,
        yticklabels=y_ticks,
    )
    ax.set_ylabel(y_label)

    plt.savefig(f"{file_name}.pdf", bbox_inches="tight")
    plt.close()
