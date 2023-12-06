import argparse

from experiments import downstream_experiment

from feature_weighting_methods import (
    logistic_regression_sample_weights,
    uniform_feature_weighting,
    random_weighting,
    random_forest_weighting,
    mutual_information,
)

from sample_weighting_methods import (
    soft_mrs_weighting,
    kernel_mean_matching,
    propensity_score_adjustment,
    repeated_MRS,
    uniform_sample_weighting,
)


# Possible weighting methods
sample_weighting_method_list = [
    "psa",
    "uniform",
    "soft-mrs",
    "mrs",
    "kmm",
]

feature_weighting_method_list = [
    "logistic_regression",
    "uniform",
    "random",
    "random_forest",
    "mutual_information",
]

# Possible bias types
bias_choice = [
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
    parser.add_argument(
        "--feature_weighting_method",
        choices=feature_weighting_method_list,
        required=True,
    )
    parser.add_argument(
        "--sample_weighting_method", choices=sample_weighting_method_list, required=True
    )
    parser.add_argument("--bias_type", choices=bias_choice, default="none")
    parser.add_argument("--number_of_repetitions", default=50, type=int)
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


def get_sample_weighting_function(method_name):
    """Returns the function to the function name.

    :param method_name: Method name
    :return: corresponding weighting function
    """
    if method_name == "uniform":
        return uniform_sample_weighting
    elif method_name == "logistic_regression":
        return propensity_score_adjustment
    elif method_name == "soft-mrs":
        return soft_mrs_weighting
    elif method_name == "mrs":
        return repeated_MRS
    elif method_name == "kmm":
        return kernel_mean_matching


def get_feature_weighting_function(method_name):
    """Returns the function to the function name.

    :param method_name: Method name
    :return: corresponding weighting function
    """
    if method_name == "uniform":
        return uniform_feature_weighting
    elif method_name == "logistic_regression":
        return logistic_regression_sample_weights
    elif method_name == "random":
        return random_weighting
    elif method_name == "random_forest":
        return random_forest_weighting
    elif method_name == "mutual_information":
        return mutual_information


def get_experiment_function():
    """Returns the experiment function to a name.

    :param dataset_name: Data set name
    :return: Experiment function
    """
    return downstream_experiment
