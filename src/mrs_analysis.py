import random
import numpy as np
from pathlib import Path

from experiments.downstream_tasks import repeated_train_val_test_split
from utils.parameter import set_parameter
from utils.statistics import create_result_path
from weighting_methods import maximum_representative_subsampling
from utils.command_line_arguments import parse_mrs_analysis_command_line_arguments
from utils.data_loader import load_dataset, load_weights
from utils.sampling import sample_N
from utils.metrics import calculate_mean_rocs, scale_df
from utils.visualization import (
    plot_auc_average,
    plot_relative_bias,
    plot_mmds_average,
    plot_rocs_mrs,
)

seed = 5
sampling_random_generator = np.random.RandomState(seed)


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
    result_path = create_result_path(
        "mrs",
        bias_type,
        data_set_name,
        bias_fraction=bias_fraction,
        experiment_name="mrs_analysis",
    )
    load_previous_results = True
    mmd_list = []
    (
        _,
        _,
        _,
        _,
        _,
        _,
        hyperparameter_list,
    ) = set_parameter("mrs-forest")

    df, columns, target = load_dataset(data_set_name)
    sample_df, _ = scale_df(df, columns)

    if data_set_name in ("gbs_gesis", "gbs_allensbach"):
        N = sample_df[sample_df["label"] == 1]
        R = sample_df[sample_df["label"] == 0]
        use_bias_mean = False
    else:
        use_bias_mean = True

    sample_weights_save_path = result_path / "sample_weights"
    feature_weights_save_path = result_path / "feature_weights"

    sample_weights_save_path.mkdir(exist_ok=True)
    feature_weights_save_path.mkdir(exist_ok=True)

    sample_weights_list = load_weights(sample_weights_save_path)
    feature_weights_list = load_weights(feature_weights_save_path)

    for i, (N, R, _) in enumerate(
        repeated_train_val_test_split(
            n_cv_splits,
            n_cv_repeats,
            sample_df,
            sample_df[target],
            sampling_random_generator,
        )
    ):
        if data_set_name not in ("gbs_gesis", "gbs_allensbach"):
            N = sample_N(
                train=N,
                bias_type=bias_type,
                bias_fraction=bias_fraction,
                columns=columns,
                bias_variable=target,
                random_generator=sampling_random_generator,
            )
            number_of_samples = len(N)
            N["label"] = 1
            R["label"] = 0

        if len(sample_weights_list) > i and load_previous_results:
            sample_weights = sample_weights_list[i]
            feature_weights = feature_weights_list[i]
        else:
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
                hyperparameter_list=hyperparameter_list,
                early_stopping=False,
            )

        aucs_complete.append(auc_list)
        mmds_complete.append(mmd_list)
        mrs_iteration_list.append(mrs_iteration)
        rocs_list_list.append(roc_list)
        relative_bias_list_list.append(relative_bias_list)

        # mean_rocs = calculate_mean_rocs(rocs_list_list)

        plot_mmds_average(
            mmds_complete,
            drop,
            result_path / "mmd",
            mrs_iteration_list,
            number_of_samples,
        )
        plot_auc_average(
            aucs_complete,
            drop,
            result_path / "auroc",
            number_of_samples,
            mrs_iteration_list,
        )

        # plot_rocs_mrs(mean_rocs, result_path / "rocs")
        plot_relative_bias(
            relative_bias_list_list,
            result_path / "relative_bias",
            mrs_iteration_list,
            number_of_samples,
            drop,
        )


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
