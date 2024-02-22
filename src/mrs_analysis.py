import random
import numpy as np
from pathlib import Path
from tqdm import trange

from methods import maximum_representative_subsampling
from utils.command_line_arguments import parse_mrs_analysis_command_line_arguments
from utils.data_loader import load_dataset
from utils.sampling import sample
from utils.metrics import calculate_mean_rocs, scale_df
from utils.visualization import (
    plot_auc_average,
    plot_relative_bias,
    plot_mmds_average,
    plot_rocs,
)

seed = 5


def analyse_mrs(number_of_repetitions, data_set_name, bias_type, drop):
    """Run mrs on different data sets

    :param number_of_repetitions: Number of repetitions
    :param data_set_name: Data set name
    :param bias_type: Bias type
    :param drop: Defines how many samples are dropped in each iteration
    """
    np.random.seed(seed)
    random.seed(seed)
    random_generator = np.random.RandomState(seed)
    aucs_complete = []
    mmds_complete = []
    mrs_iteration_list = []
    rocs_list_list = []
    relative_bias_list_list = []
    data, columns, _ = load_dataset(data_set_name)
    data = data.sample(frac=1)
    result_path = create_save_path(data_set_name, bias_type)
    mmd_list = []
    bias_variable = ""
    use_bias_mean = False

    if data_set_name == "folktables_income":
        scaled_df, _ = scale_df(data, columns)
        bias_variable = "Binary Income"
        scaled_N, scaled_R = sample(
            bias_type,
            scaled_df,
            "Binary Income",
            train_fraction=0.5,
            bias_fraction=0.1,
            columns=columns,
        )
        use_bias_mean = True
    else:
        scaled_df, _ = scale_df(data, columns)
        scaled_N = scaled_df[scaled_df["label"] == 1]
        scaled_R = scaled_df[scaled_df["label"] == 0]
    number_of_samples = len(scaled_N)

    for _ in trange(number_of_repetitions):
        (
            auc_list,
            mmd_list,
            relative_bias_list,
            mrs_iteration,
            roc_list,
        ) = maximum_representative_subsampling.repeated_MRS(
            scaled_N,
            scaled_R,
            columns,
            drop=drop,
            save_path=result_path,
            return_metrics=True,
            use_bias_mean=use_bias_mean,
            bias_variable=bias_variable,
            random_generator=random_generator,
        )
        aucs_complete.append(auc_list)
        mmds_complete.append(mmd_list)
        mrs_iteration_list.append(mrs_iteration)
        rocs_list_list.append(roc_list)
        relative_bias_list_list.append(relative_bias_list)

    mean_mmds = np.mean(mmds_complete, axis=0)
    std_mmds = np.std(mmds_complete, axis=0)

    mean_aucs = np.mean(aucs_complete, axis=0)
    std_aucs = np.std(aucs_complete, axis=0)

    mean_relative_bias = np.mean(relative_bias_list_list, axis=0)
    std_relative_bias = np.std(relative_bias_list_list, axis=0)

    mean_rocs = calculate_mean_rocs(rocs_list_list)

    plot_mmds_average(
        mean_mmds,
        std_mmds,
        drop,
        1,
        result_path / "mmd",
        mrs_iteration_list,
        number_of_samples,
    )
    plot_auc_average(
        mean_aucs,
        std_aucs,
        drop,
        result_path / "auroc",
        number_of_samples,
        mrs_iteration_list,
    )

    plot_rocs(mean_rocs, result_path / "rocs")
    plot_relative_bias(
        mean_relative_bias,
        std_relative_bias,
        result_path / "relative_bias",
        mrs_iteration_list,
        number_of_samples,
        drop,
    )


def create_save_path(data_set_name, bias_type):
    """Creates the path for result files

    :param data_set_name: Data set name
    :param bias_type: Bias type name
    :return: File path
    """
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../results")
    result_path = result_path / "mrs_analysis" / data_set_name / bias_type
    result_path.mkdir(exist_ok=True, parents=True)
    return result_path


if __name__ == "__main__":
    args = parse_mrs_analysis_command_line_arguments()
    analyse_mrs(
        args.number_of_repetitions, args.data_set_name, args.bias_type, args.drop
    )
