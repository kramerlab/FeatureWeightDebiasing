import random
import numpy as np

from utils.data_loader import load_dataset
from utils.command_line_arguments import (
    parse_command_line_arguments,
    get_sample_weighting_function,
    get_experiment_function,
)

seed = 5
no_weights_function = [
    "train_domain_adversarial_network",
]


def weighting_experiment(
    data_set_name: str,
    sample_weighting_method_name: str,
    bias_type: str,
    number_of_repetitions: int,
    load_previous_results: bool,
    experiment_name: str,
    budget: float,
    drop: int,
    transformation_method,
    validation_method,
    bias_fraction,
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
    compute_sample_weights_function = get_sample_weighting_function(
        sample_weighting_method_name
    )
    experiment_function = get_experiment_function(experiment_name)

    explicit_weights = (
        False
        if compute_sample_weights_function.__name__ in no_weights_function
        else True
    )

    experiment_function(
        df=data,
        columns=columns,
        sample_weighting_method=compute_sample_weights_function,
        bias_type=bias_type,
        number_of_repetitions=number_of_repetitions,
        data_set_name=data_set_name,
        target=target,
        random_generator=random_generator,
        explicit_weights=explicit_weights,
        load_previous_results=load_previous_results,
        method_name=sample_weighting_method_name,
        budget=budget,
        drop=drop,
        transformation_method=transformation_method,
        validation_method=validation_method,
        bias_fraction=bias_fraction,
    )


if __name__ == "__main__":
    args = parse_command_line_arguments()
    weighting_experiment(
        args.dataset,
        args.sample_weighting_method,
        args.bias_type,
        args.number_of_repetitions,
        args.load_previous_results,
        args.experiment_name,
        args.budget,
        args.drop,
        args.transformation_method,
        args.validation_method,
        args.bias_fraction,
    )
