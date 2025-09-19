import random
import numpy as np
import json

from experiments.downstream_tasks import repeated_train_val_test_split
from utils.parameter import set_parameter
from utils.statistics import create_result_path
from weighting_methods import maximum_representative_subsampling
from utils.command_line_arguments import parse_mrs_analysis_command_line_arguments
from utils.data_loader import load_dataset, load_saved_results, save_results
from utils.sampling import sample_N
from utils.metrics import scale_df
from utils.visualization import (
    plot_auc_average,
    plot_relative_bias,
    plot_value_average,
)

seed = 5
sampling_random_generator = np.random.RandomState(seed)


def analyse_mrs(
    n_cv_splits,
    n_cv_repeats,
    data_set_name,
    bias_type,
    drop,
    bias_fraction,
    mrs_function,
    load_previous_results,
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
    wasserstein_target = "Resilienz" if data_set_name == "gbs_allensbach" else None

    result_path = create_result_path(
        "mrs",
        bias_type,
        data_set_name,
        bias_fraction=bias_fraction,
        experiment_name=f"mrs_analysis/{mrs_function}",
    )
    mmd_dict = []
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
        split_method = gbs_split
        use_bias_mean = True
    else:
        use_bias_mean = True
        split_method = repeated_train_val_test_split

    aurocs_save_path = result_path / "aurocs"
    mmds_save_path = result_path / "mmds"
    relative_bias_save_path = result_path / "relative_bias"
    mrs_iterations_save_path = result_path / "mrs_iterations"
    wassersteins_save_path = result_path / "wassersteins"

    aurocs_save_path.mkdir(exist_ok=True)
    mmds_save_path.mkdir(exist_ok=True)
    relative_bias_save_path.mkdir(exist_ok=True)
    mrs_iterations_save_path.mkdir(exist_ok=True)
    wassersteins_save_path.mkdir(exist_ok=True)

    aucs_complete = load_saved_results(aurocs_save_path, "aurocs")
    wassersteins_complete = load_saved_results(wassersteins_save_path, "wassersteins")
    mmds_complete = load_saved_results(mmds_save_path, "mmds")
    mrs_iteration_dict_list = load_saved_results(
        mrs_iterations_save_path, "mrs_iterations"
    )
    relative_bias_dict_list = load_saved_results(
        relative_bias_save_path, "relative_biases"
    )

    for i, (N, R, _) in enumerate(
        split_method(
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
            N["label"] = 1
            R["label"] = 0

        number_of_samples = len(N)

        if len(aucs_complete) <= i and load_previous_results:
            (
                auc_dict,
                mmd_dict,
                relative_bias_dict,
                mrs_iteration_dict,
                wasserstein_dict,
            ) = maximum_representative_subsampling.mrs(
                N,
                R,
                columns,
                drop=drop,
                return_metrics=True,
                compute_bias=use_bias_mean,
                target=target,
                random_generator=random_generator,
                hyperparameter_list=hyperparameter_list,
                early_stopping=False,
                mrs_function=mrs_function,
                wasserstein_target=wasserstein_target,
            )

            aucs_complete.append(auc_dict)
            mmds_complete.append(mmd_dict)
            wassersteins_complete.append(wasserstein_dict)
            mrs_iteration_dict_list.append(mrs_iteration_dict)
            relative_bias_dict_list.append(relative_bias_dict)

            save_results(mmds_save_path, mmds_complete, "mmds")
            save_results(
                relative_bias_save_path, relative_bias_dict_list, "relative_biases"
            )
            save_results(
                mrs_iterations_save_path, mrs_iteration_dict_list, "mrs_iterations"
            )
            save_results(wassersteins_save_path, wassersteins_complete, "wassersteins")


        plot_value_average(
            mmds_complete,
            drop,
            result_path / "mmd",
            mrs_iteration_dict_list,
            number_of_samples,
        )
        plot_auc_average(
            aucs_complete,
            drop,
            result_path / "auroc",
            number_of_samples,
            mrs_iteration_dict_list,
        )
        plot_auc_average(
            aucs_complete,
            drop,
            result_path / "auroc",
            number_of_samples,
            mrs_iteration_dict_list,
            wide=False,
        )

        plot_value_average(
            wassersteins_complete,
            drop,
            result_path / "wasserstein",
            mrs_iteration_dict_list,
            number_of_samples,
            ylabel="Wasserstein Distance",
        )


        if data_set_name not in ("gbs_gesis", "gbs_allensbach"):
            plot_relative_bias(
                relative_bias_dict_list,
                result_path / "relative_bias",
                mrs_iteration_dict_list,
                number_of_samples,
                drop,
            )

        meta_data_dict = {"n_dropped": drop, "number_of_samples": number_of_samples}
        with open(result_path / 'metadata.json', 'w') as file:
            json.dump(meta_data_dict, file)


def gbs_split(n_cv_splits, n_cv_repeats, df, target_values, random_generator):
    for _ in range(n_cv_splits):
        for _ in range(n_cv_repeats):
            N = df[df["label"] == 1]
            R = df[df["label"] == 0]
            yield N, R, _


if __name__ == "__main__":
    args = parse_mrs_analysis_command_line_arguments()
    analyse_mrs(
        args.n_cv_splits,
        args.n_cv_repeats,
        args.data_set_name,
        args.bias_type,
        args.drop,
        args.bias_fraction,
        args.mrs_function,
        args.load_previous_results,
    )
