import random
import numpy as np

from utils.data_loader import load_dataset
from utils.command_line_arguments import (
    parse_command_line_arguments,
    get_feature_weighting_function,
    get_sample_weighting_function,
    get_experiment_function,
)

seed = 5
no_weights_function = [
    "train_domain_adversarial_network",
    "train_correction_weights_network",
]


def weighting_experiment(
    data_set_name: str,
    feature_weighting_method_name: str,
    sample_weighting_method_name: str,
    bias_type: str,
    number_of_repetitions: int,
) -> None:
    """_summary_

    :param data_set_name: Data set name
    :param method_name: Method name
    :param bias_type: Bias Type
    :param number_of_repetitions: Number of repetitions
    """
    # Set random seeds for reproducibility.
    np.random.seed(seed)
    random.seed(seed)
    random_generator = np.random.RandomState(seed)
    data, columns, target = load_dataset(data_set_name)
    compute_feature_weights_function = get_feature_weighting_function(
        feature_weighting_method_name
    )
    compute_sample_weights_function = get_sample_weighting_function(
        sample_weighting_method_name
    )
    experiment_function = get_experiment_function()

    explicit_weights = (
        False
        if compute_sample_weights_function.__name__ in no_weights_function
        else True
    )

    experiment_function(
        data,
        columns,
        compute_sample_weights_function,
        compute_feature_weights_function,
        bias_type=bias_type,
        number_of_repetitions=number_of_repetitions,
        data_set_name=data_set_name,
        target=target,
        random_generator=random_generator,
        explicit_weights=explicit_weights,
    )


if __name__ == "__main__":
    args = parse_command_line_arguments()
    weighting_experiment(
        args.dataset,
        args.feature_weighting_method,
        args.sample_weighting_method,
        args.bias_type,
        args.number_of_repetitions,
    )
