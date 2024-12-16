import argparse

from experiments import (
    temperature_comparison,
    perform_statistical_analysis_mrs,
    downstream_tasks_experiment,
    fairness_tasks_experiment,
    decomposition_experiment,
)


from weighting_methods import (
    soft_mrs_weighting,
    kernel_mean_matching,
    propensity_score_adjustment,
    feature_weighted_repeated_MRS,
    uniform_sample_weighting,
    mrs,
    fw_MRS_SVM,
)


# Possible weighting methods
sample_weighting_method_list = [
    "uniform",
    "mrs-forest",
    "soft-mrs-linear",
    "soft-mrs-exponential",
    "fw-mrs-temperature",
    "fw-mrs-temperature-svm",
    "kmm",
    "psa",
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
    "fairness_adult",
    "fairness_folktables_income",
    "folktables_income",
    "folktables_employment",
    "breast_cancer",
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
    parser.add_argument("--n_cv_splits", default=2, type=int)
    parser.add_argument("--n_cv_repeats", default=5, type=int)
    parser.add_argument("--budget", default=0.01, type=none_or_float)
    parser.add_argument("--load_previous_results", default=False, action="store_true")
    parser.add_argument("--experiment_name", default="downstream")
    parser.add_argument("--drop", default=1, type=int)
    parser.add_argument("--transformation_method", default="temperature")
    parser.add_argument("--validation_method", default="random_forest")
    parser.add_argument("--bias_fraction", default=0.25, type=float)

    return parser.parse_args()


def none_or_float(value):
    if value == "None":
        return None
    return float(value)


def parse_mrs_analysis_command_line_arguments():
    """Parses the command line arguments for the MRS analysis.

    :return: Parsed command line arguments for the MRS analysis.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set_name", choices=dataset_list, required=True)
    parser.add_argument("--mrs_function",  default="mrs_step")
    parser.add_argument("--n_cv_splits", default=2, type=int)
    parser.add_argument("--n_cv_repeats", default=5, type=int)
    parser.add_argument("--bias_type", choices=bias_choice, default="none")
    parser.add_argument("--drop", default=1, type=int)
    parser.add_argument("--bias_fraction", default=0.1, type=float)
    parser.add_argument("--load_previous_results", default=False, action="store_true")

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
    elif method_name in ("soft-mrs-linear", "soft-mrs-exponential"):
        return soft_mrs_weighting
    elif method_name in ("fw-mrs-temperature", ):
        return feature_weighted_repeated_MRS
    elif method_name == "mrs-forest":
        return mrs
    elif method_name == "kmm":
        return kernel_mean_matching
    elif method_name == "fw-mrs-temperature-svm":
        return fw_MRS_SVM


def get_experiment_function(experiment_name=""):
    """Returns the experiment function to a name.

    :param dataset_name: Data set name
    :return: Experiment function
    """
    if experiment_name == "downstream_task":
        return downstream_tasks_experiment
    elif experiment_name == "temperature_comparison":
        return temperature_comparison
    elif experiment_name == "statistical_analysis":
        return perform_statistical_analysis_mrs
    elif experiment_name == "fairness_task":
        return fairness_tasks_experiment
    elif experiment_name == "decomposition":
        return decomposition_experiment
    elif experiment_name == "lipidomics":
        return 
