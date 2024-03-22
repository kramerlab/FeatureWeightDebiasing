import argparse

from experiments import (
    downstream_experiment,
    downstream_experiment_with_test_set,
    feature_weight_budget_comparison_experiment,
)


from weighting_methods import (
    soft_mrs_weighting,
    kernel_mean_matching,
    propensity_score_adjustment,
    feature_weighted_repeated_MRS,
    uniform_sample_weighting,
    train_domain_adversarial_network,
    neural_network_mmd_loss_weighting,
    repeated_MRS,
)


# Possible weighting methods
sample_weighting_method_list = [
    "psa",
    "uniform",
    "soft-mrs",
    "mrs",
    "fw-sampling-mrs",
    "kmm",
    "dann",
    "mmd_loss",
]


# Possible bias types
bias_choice = [
    "less_negative_class",
    "less_positive_class",
    "mean_difference",
    "none",
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
        "--sample_weighting_method", choices=sample_weighting_method_list, required=True
    )
    parser.add_argument("--bias_type", choices=bias_choice, default="none")
    parser.add_argument("--number_of_repetitions", default=50, type=int)
    parser.add_argument("--budget", default=0.0, type=float)
    parser.add_argument("--load_previous_results", default=False, action="store_true")
    parser.add_argument("--experiment_name", default="downstream")
    parser.add_argument("--drop", default=1, type=int)

    return parser.parse_args()


def parse_mrs_ablation_command_line_arguments():
    """Parses the command line arguments for the ablation study.

    :return: Parsed command line arguments for the ablation study.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation_experiment", choices=mrs_ablation_experiments, required=True
    )
    parser.add_argument("--dataset_name", choices=dataset_list, required=True)
    parser.add_argument("--bias_type", choices=bias_choice, default="none")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", default=1, type=int)
    parser.add_argument("--patience", default=25, type=int)
    parser.add_argument("--n_repeats", default=1000, type=int)

    return parser.parse_args()


def get_sample_weighting_function(method_name):
    """Returns the function to the function name.

    :param method_name: Method name
    :return: corresponding weighting function
    """
    if method_name == "uniform":
        return uniform_sample_weighting
    elif method_name == "psa":
        return propensity_score_adjustment
    elif method_name == "soft-mrs":
        return soft_mrs_weighting
    elif method_name == "fw-sampling-mrs":
        return feature_weighted_repeated_MRS
    elif method_name == "mrs":
        return repeated_MRS
    elif method_name == "kmm":
        return kernel_mean_matching
    elif method_name == "dann":
        return train_domain_adversarial_network
    elif method_name == "mmd_loss":
        return neural_network_mmd_loss_weighting


def get_experiment_function(experiment_name=""):
    """Returns the experiment function to a name.

    :param dataset_name: Data set name
    :return: Experiment function
    """
    if experiment_name == "test_set":
        return downstream_experiment_with_test_set
    elif experiment_name == "feature_weight_budget_comparison":
        return feature_weight_budget_comparison_experiment
    else:
        return downstream_experiment
