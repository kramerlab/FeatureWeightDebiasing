import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold

from weighting_methods import maximum_representative_subsampling
from utils.command_line_arguments import parse_mrs_analysis_command_line_arguments
from utils.data_loader import load_dataset
from utils.sampling import sample_N
from utils.metrics import calculate_mean_rocs, scale_df
from utils.visualization import (
    plot_auc_average,
    plot_relative_bias,
    plot_mmds_average,
    plot_rocs_mrs,
)

seed = 5


def analyse_mrs(
    n_cv_splits, n_cv_repeats, data_set_name, bias_type, drop, bias_fraction
):
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
    result_path = create_save_path(
        data_set_name, bias_type, bias_fraction=bias_fraction
    )
    mmd_list = []

    df, columns, target = load_dataset(data_set_name)
    sample_df, _ = scale_df(df, columns)

    if data_set_name in ("gbs_gesis", "gbs_allensbach"):
        N = sample_df[sample_df["label"] == 1]
        R = sample_df[sample_df["label"] == 0]
        use_bias_mean = False
    else:
        use_bias_mean = True

    skf = RepeatedStratifiedKFold(
        n_splits=n_cv_splits, n_repeats=n_cv_repeats, random_state=seed
    )
    for train_indices, test_indices in skf.split(sample_df, sample_df[target]):
        if data_set_name not in ("gbs_gesis", "gbs_allensbach"):
            N = sample_df.iloc[train_indices].copy()
            N = sample_N(
                train=N,
                bias_type=bias_type,
                bias_fraction=bias_fraction,
                columns=columns,
                bias_variable=target,
            )
            number_of_samples = len(N)
            R = sample_df.iloc[test_indices].copy()
            N["label"] = 1
            R["label"] = 0
        (
            auc_list,
            mmd_list,
            relative_bias_list,
            mrs_iteration,
            roc_list,
        ) = maximum_representative_subsampling.mrs(
            N,
            R,
            columns,
            drop=drop,
            save_path=result_path,
            return_metrics=True,
            compute_bias=use_bias_mean,
            target=target,
            random_generator=random_generator,
            early_stopping=False,
        )
        aucs_complete.append(auc_list)

        mmds_complete.append(mmd_list)
        mrs_iteration_list.append(mrs_iteration)
        rocs_list_list.append(roc_list)
        relative_bias_list_list.append(relative_bias_list)

        min_length = np.min([len(mmd_values) for mmd_values in mmds_complete])
        mean_mmds = np.mean(np.array(mmds_complete)[:, :min_length], axis=0)
        std_mmds = np.std(np.array(mmds_complete)[:, :min_length], axis=0)

        mean_aucs = np.mean(np.array(aucs_complete)[:, :min_length], axis=0)
        std_aucs = np.std(np.array(aucs_complete)[:, :min_length], axis=0)

        mean_relative_bias = np.mean(
            np.array(relative_bias_list_list)[:, :min_length], axis=0
        )
        std_relative_bias = np.std(
            np.array(relative_bias_list_list)[:, :min_length], axis=0
        )

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

        plot_rocs_mrs(mean_rocs, result_path / "rocs")
        plot_relative_bias(
            mean_relative_bias,
            std_relative_bias,
            result_path / "relative_bias",
            mrs_iteration_list,
            number_of_samples,
            drop,
        )


def create_save_path(data_set_name, bias_type, bias_fraction):
    """Creates the path for result files

    :param data_set_name: Data set name
    :param bias_type: Bias type name
    :return: File path
    """
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../results")
    result_path = (
        result_path / "mrs_analysis" / data_set_name / bias_type / str(bias_fraction)
    )
    result_path.mkdir(exist_ok=True, parents=True)
    return result_path


if __name__ == "__main__":
    args = parse_mrs_analysis_command_line_arguments()
    analyse_mrs(
        args.n_cv_splits,
        args.n_cv_repeats,
        args.data_set_name,
        args.bias_type,
        args.drop,
        args.bias_fraction,
    )
