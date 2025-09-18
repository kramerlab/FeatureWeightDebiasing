import random
import numpy as np

from experiments.downstream_tasks import repeated_train_val_test_split
from utils.parameter import set_parameter
from utils.statistics import create_result_path
from weighting_methods import soft_mrs_weighting
from utils.command_line_arguments import parse_mrs_analysis_command_line_arguments
from utils.data_loader import load_dataset, load_saved_results, save_results
from utils.sampling import sample_N
from utils.metrics import scale_df

seed = 5
sampling_random_generator = np.random.RandomState(seed)


def analyse_soft_mrs(
    n_cv_splits,
    n_cv_repeats,
    data_set_name,
    bias_type,
    bias_fraction,
    soft_mrs_function,
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
    mmds_complete = []
    relative_bias_complete = []
    sample_weights_list_list = []
    wasserstein_target = "Resilienz" if data_set_name == "gbs_allensbach" else None

    result_path = create_result_path(
        "soft_mrs",
        bias_type,
        data_set_name,
        bias_fraction=bias_fraction,
        experiment_name=f"soft_mrs_analysis/{soft_mrs_function}",
    )
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

    mmds_save_path = result_path / "mmds"
    wasserstein_save_path = result_path / "wassersteins"
    aurocs_save_path = result_path / "aurocs"
    relative_bias_save_path = result_path / "relative_bias"
    sample_weights_save_path = result_path / "sample_weights"

    mmds_save_path.mkdir(exist_ok=True)
    wasserstein_save_path.mkdir(exist_ok=True)
    aurocs_save_path.mkdir(exist_ok=True)
    relative_bias_save_path.mkdir(exist_ok=True)
    sample_weights_save_path.mkdir(exist_ok=True)

    mmds_complete = load_saved_results(mmds_save_path, "mmds")
    wassersteins_complete = load_saved_results(wasserstein_save_path, "wassersteins")
    auroc_complete = load_saved_results(aurocs_save_path, "aurocs")
    relative_bias_complete = load_saved_results(
        relative_bias_save_path, "relative_biases"
    )
    sample_weights_list = load_saved_results(sample_weights_save_path, "sample_weights")

    exponential = True if soft_mrs_function == "exponential" else False

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

        if not (len(mmds_complete) > i and load_previous_results):
            (
                mmd_list,
                relative_bias_list,
                sample_weights_list,
                auroc_list,
                wasserstein_list,
            ) = soft_mrs_weighting(
                N,
                R,
                columns,
                return_metrics=True,
                compute_bias=use_bias_mean,
                target=target,
                random_generator=random_generator,
                hyperparameter_list=hyperparameter_list,
                early_stopping=False,
                exponential=exponential,
                n_iterations=2000,
                wasserstein_target=wasserstein_target,
            )

            mmds_complete.append(mmd_list)
            auroc_complete.append(auroc_list)
            wassersteins_complete.append(wasserstein_list)
            relative_bias_complete.append(relative_bias_list)
            sample_weights_list_list.append(sample_weights_list)

            save_results(mmds_save_path, mmds_complete, "mmds")
            save_results(aurocs_save_path, auroc_complete, "aurocs")
            save_results(wasserstein_save_path, wassersteins_complete, "wassersteins")
            save_results(
                relative_bias_save_path, relative_bias_complete, "relative_biases"
            )
            save_results(
                sample_weights_save_path, sample_weights_list_list, "sample_weights"
            )


def gbs_split(n_cv_splits, n_cv_repeats, df, target_values, random_generator):
    N = df[df["label"] == 1]
    R = df[df["label"] == 0]
    for _ in range(n_cv_splits * n_cv_repeats):
        yield N, R, _


if __name__ == "__main__":
    args = parse_mrs_analysis_command_line_arguments()
    analyse_soft_mrs(
        args.n_cv_splits,
        args.n_cv_repeats,
        args.data_set_name,
        args.bias_type,
        args.bias_fraction,
        args.mrs_function,
        args.load_previous_results,
    )
