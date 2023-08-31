import argparse
import functools


from experiments import (
    gbs_allensbach_experiment,
    gbs_gesis_experiment,
    downstream_experiment,
)

from methods import (
    soft_mrs_weighting,
    kernel_mean_matching,
    propensity_score_adjustmen,
    neural_network_mmd_loss_weighting,
    repeated_MRS,
    uniform_weighting,
)

# Possible weighting methods
method_list = [
    "logistic_regression",
    "neural_network_mmd_loss",
    "uniform",
    "soft-mrs",
    "soft-mrs-exponential",
    "mrs",
    "kmm",
]

# Possible bias types
bias_choice = [
    "none",
    "less_negative_class",
    "less_positive_class",
    "mean_difference",
]

# Possible data sets
dataset_list = [
    "gbs_allensbach",
    "gbs_gesis",
    "folktables_income",
    "folktables_employment",
    "breast_cancer",
    "hr_analytics",
    "loan_prediction",
]

mrs_ablation_experiments = [
    "random",
    "cross-validation",
]

down_stream_data_sets = [
    "breast_cancer",
    "folktables_employment",
    "folktables_income",
    "hr_analytics",
    "loan_prediction",
]


def parse_command_line_arguments():
    """Parses the command line arguments.

    :return: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=dataset_list, required=True)
    parser.add_argument("--method", choices=method_list, required=True)
    parser.add_argument("--bias_type", choices=bias_choice, default="none")
    parser.add_argument("--number_of_repetitions", default=100, type=int)
    return parser.parse_args()


def parse_mrs_ablation_command_line_arguments():
    """Parses the command line arguments for the ablation study.

    :return: Parsed command line arguments for the ablation study.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation_experiment", choices=mrs_ablation_experiments, required=True
    )
    parser.add_argument("--number_of_repetitions", default=100, type=int)
    parser.add_argument("--drop", default=1, type=int)
    return parser.parse_args()


def parse_mrs_analysis_command_line_arguments():
    """Parses the command line arguments for the MRS analysis.

    :return: Parsed command line arguments for the MRS analysis.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set_name", choices=dataset_list, required=True)
    parser.add_argument("--number_of_repetitions", default=10, type=int)
    parser.add_argument("--bias_type", choices=bias_choice, default="none")
    parser.add_argument("--drop", default=1, type=int)

    return parser.parse_args()


def parse_command_line_arguments_statistical_analysis():
    """Parses the command line arguments for the statistical experiment.

    :return: Parsed command line arguments for the statistical experiment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_one", choices=method_list, required=True)
    parser.add_argument("--method_two", choices=method_list, required=True)
    return parser.parse_args()


def get_weighting_function(method_name):
    """Returns the function to the function name.

    :param method_name: Method name
    :return: corresponding weighting function
    """
    if method_name == "uniform":
        return uniform_weighting
    elif method_name == "logistic_regression":
        return propensity_score_adjustmen
    elif method_name == "neural_network_mmd_loss":
        return neural_network_mmd_loss_weighting
    elif method_name == "soft-mrs": 
        return soft_mrs_weighting
    elif method_name == "soft-mrs-exponential":
        return functools.partial(soft_mrs_weighting, exponential=True)
    elif method_name == "mrs":
        return repeated_MRS
    elif method_name == "kmm":
        return kernel_mean_matching


def get_experiment_function(dataset_name):
    """Returns the experiment function to a name.

    :param dataset_name: Data set name
    :return: Experiment function
    """
    if dataset_name == "gbs_gesis":
        return gbs_gesis_experiment
    elif dataset_name in down_stream_data_sets:
        return downstream_experiment
    else:
        return gbs_allensbach_experiment
