import random
import numpy as np
from pathlib import Path
from tqdm import trange

from methods import maximum_representative_subsampling
from utils.command_line_arguments import parse_mrs_ablation_command_line_arguments
from utils.data_loader import load_dataset
from utils.metrics import scale_df
from utils.visualization import (
    plot_experiment_comparison_auc,
    plot_experiment_comparison_mmd,
)


seed = 5


def compare_mrs_variants(number_of_repetitions, ablation_experiment, drop):
    """Compare different mrs variants

    :param number_of_repetitions: Number of repetitions
    :param ablation_experiment: Name of the MRS variatn
    :param drop: Defines how many samples are dropped each iteration
    """
    np.random.seed(seed)
    random.seed(seed)
    random_generator = np.random.RandomState(seed)
    aucs_complete = []
    mmds_complete = []
    aucs_comparison = []
    mmds_comparison = []
    mmd_list = []

    file_directory = Path(__file__).parent
    result_path = create_result_path(ablation_experiment, file_directory)
    data, columns, _ = load_dataset("gbs_gesis")
    scaled_N, scaled_R = preprocess_data(data, columns)
    number_of_samples = len(scaled_N)

    class_weights, mrs_function, sampling, experiment_label = choose_hyperparameter(
        ablation_experiment
    )

    for _ in trange(number_of_repetitions):
        # First the normal variant
        (
            auc_list,
            mmd_list,
            _,
            _,
            _,
        ) = maximum_representative_subsampling.repeated_MRS(
            scaled_N,
            scaled_R,
            columns,
            drop=drop,
            save_path=result_path,
            return_metrics=True,
            sampling="max",
            random_generator=random_generator,
        )
        aucs_complete.append(auc_list)
        mmds_complete.append(mmd_list)
        # Than comparison variant
        (
            comparison_auc_list,
            comparison_mmd_list,
            _,
            _,
            _,
        ) = maximum_representative_subsampling.repeated_MRS(
            scaled_N,
            scaled_R,
            columns,
            drop=drop,
            sampling=sampling,
            mrs_function=mrs_function,
            save_path=result_path,
            return_metrics=True,
            class_weights=class_weights,
            random_generator=random_generator,
        )
        aucs_comparison.append(comparison_auc_list)
        mmds_comparison.append(comparison_mmd_list)

    comparison_mean_mmds = np.mean(mmds_comparison, axis=0)
    comparison_std_mmds = np.std(mmds_comparison, axis=0)
    comparison_mean_aucs = np.mean(aucs_comparison, axis=0)
    comparison_std_aucs = np.std(aucs_comparison, axis=0)

    mean_mmds = np.mean(mmds_complete, axis=0)
    std_mmds = np.std(mmds_complete, axis=0)
    mean_aucs = np.mean(aucs_complete, axis=0)
    std_aucs = np.std(aucs_complete, axis=0)

    plot_result_graphs(
        drop,
        result_path,
        number_of_samples,
        experiment_label,
        comparison_mean_mmds,
        comparison_std_mmds,
        comparison_mean_aucs,
        comparison_std_aucs,
        mean_mmds,
        std_mmds,
        mean_aucs,
        std_aucs,
    )


def preprocess_data(data, columns):
    """Preprocess the data

    :param data: Data set as pandas.DataFrame
    :param columns: Defines the columns that will be preprocessed
    :return: Preprocessed data
    """
    data = data.sample(frac=1)
    scaled_df, _ = scale_df(data, columns)
    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]
    return scaled_N, scaled_R


def create_result_path(ablation_experiment, file_directory):
    """Creates the result path

    :param ablation_experiment: Experiment name
    :param file_directory: Path to the file directory
    :return: The created path
    """
    result_path = Path(file_directory, "../results")
    result_path = result_path / "ablation_study" / ablation_experiment
    result_path.mkdir(exist_ok=True, parents=True)
    return result_path


def plot_result_graphs(
    drop,
    result_path,
    number_of_samples,
    experiment_label,
    comparison_mean_mmds,
    comparison_std_mmds,
    comparison_mean_aucs,
    comparison_std_aucs,
    mean_mmds,
    std_mmds,
    mean_aucs,
    std_aucs,
):
    """Plots the results

    :param drop: How many samples were dropped each iteration
    :param result_path: The path were the results should be saved
    :param number_of_samples: Number of samples in the original data set
    :param experiment_label: Name of the experiment
    :param comparison_mean_mmds: Mean MMDs of the MRS variant
    :param comparison_std_mmds: Standard deviation of the MMDs of the MRS variant
    :param comparison_mean_aucs: Mean AUROCs of the MRS variant
    :param comparison_std_aucs: Standard deviation AUROCs of the MMDs of the MRS variant
    :param mean_mmds: Mean MMDs of MRS
    :param std_mmds: Standard deviation of the MMDs of MRS
    :param mean_aucs: Mean AUROCs of MRS
    :param std_aucs: Standard deviation AUROCs of the MMDs of MRS
    """
    plot_experiment_comparison_mmd(
        mean_mmds,
        std_mmds,
        comparison_mean_mmds,
        comparison_std_mmds,
        experiment_label,
        drop,
        1,
        result_path / "mmd_comparison",
        number_of_samples,
    )
    plot_experiment_comparison_auc(
        mean_aucs,
        std_aucs,
        comparison_mean_aucs,
        comparison_std_aucs,
        experiment_label,
        drop,
        result_path / "auroc_comparison",
        number_of_samples,
    )


def choose_hyperparameter(ablation_experiment):
    """Choose the hyperparameter for a given MRS variant

    :param ablation_experiment: Name of the MRS variant
    :return: The corresponding hyperparameter
    """
    class_weights = "balanced"
    if ablation_experiment == "random":
        mrs_function = maximum_representative_subsampling.random_drops
        sampling = "max"
        experiment_label = "Random Drop"
    elif ablation_experiment == "cross-validation":
        sampling = "max"
        experiment_label = "MRS without cross-validation"
        mrs_function = maximum_representative_subsampling.mrs_without_cv
    return class_weights, mrs_function, sampling, experiment_label


if __name__ == "__main__":
    args = parse_mrs_ablation_command_line_arguments()
    compare_mrs_variants(
        args.number_of_repetitions, args.ablation_experiment, args.drop
    )
