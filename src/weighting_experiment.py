import random
import numpy as np

from utils.data_loader import load_dataset
from utils.command_line_arguments import (
    parse_command_line_arguments,
    get_sample_weighting_function,
    get_experiment_function,
)

seed = 5


def weighting_experiment(
    data_set_name: str,
    sample_weighting_method_name: str,
    bias_type: str,
    n_cv_splits: int,
    n_cv_repeats,
    load_previous_results: bool,
    experiment_name: str,
    drop: int,
    bias_fraction,
) -> None:
    """Main method that starts all experiments

    :param data_set_name: Data set name
    :param sample_weighting_method_name: Used weighting method
    :param bias_type: Bias Type
    :param n_cv_splits: Number of cross-validation splits
    :param n_cv_repeats: Number of repetetions of the cross-validation
    :param load_previous_results: Load weights from previous runs
    :param experiment_name: Save name of th experiment
    :param drop: Number of dropped samples per iteration
    :param bias_fraction: Bias strength
    """
    # Set random seeds for reproducibility.
    np.random.seed(seed)
    random.seed(seed)
    random_generator = np.random.RandomState(seed)
    data, columns, target = load_dataset(data_set_name)
    compute_sample_weights_function = get_sample_weighting_function(
        sample_weighting_method_name
    )
    experiment_function = get_experiment_function(experiment_name)

    experiment_function(
        df=data,
        columns=columns,
        sample_weighting_method=compute_sample_weights_function,
        bias_type=bias_type,
        n_cv_splits=n_cv_splits,
        n_cv_repeats=n_cv_repeats,
        data_set_name=data_set_name,
        target=target,
        random_generator=random_generator,
        load_previous_results=load_previous_results,
        method_name=sample_weighting_method_name,
        drop=drop,
        bias_fraction=bias_fraction,
    )


if __name__ == "__main__":
    """Parse arguments and call main method"""
    args = parse_command_line_arguments()
    weighting_experiment(
        args.dataset,
        args.sample_weighting_method,
        args.bias_type,
        args.n_cv_splits,
        args.n_cv_repeats,
        args.load_previous_results,
        args.experiment_name,
        args.drop,
        args.bias_fraction,
    )
