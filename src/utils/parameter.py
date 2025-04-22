comparison_temperatures = [
    0.5,
    0.1,
    0.05,
    0.01,
    0.005,
    0.001,
    0.0,
    -0.001,
    -0.005,
    -0.05,
    -0.01,
    -0.1,
    -0.5,
]


def set_parameter(method_name, bias_type=None):
    dropped_samples_val_dict = None
    auroc_val_dict = None
    auprc_val_dict = None
    accuracy_val_dict = None
    if method_name in (
        "fw-mrs-temperature",
        "fw-mrs-temperature-negative",
        "fw-mrs-temperature-comparison",
    ):
        draw_with_feature_weights = True
        if method_name == "fw-mrs-temperature":
            temperatures = [0.01, 0.05, 0.1]
        elif method_name == "fw-mrs-temperature-negative":
            temperatures = [-0.01, -0.05, -0.1]
        elif method_name == "fw-mrs-temperature-comparison":
            temperatures = comparison_temperatures
        hyperparameter_list = [0.0, 0.001, 0.01, 0.025]
        dropped_samples_val_dict = {temperature: [] for temperature in temperatures}
        auroc_val_dict = {temperature: [] for temperature in temperatures}
        auprc_val_dict = {temperature: [] for temperature in temperatures}
        accuracy_val_dict = {temperature: [] for temperature in temperatures}
    elif method_name in ("fw-mrs-temperature-svm",):
        draw_with_feature_weights = True
        if bias_type == "less_positive_class_comparison":
            temperatures = comparison_temperatures
        else:
            temperatures = [0.01, 0.05, 0.1]

        hyperparameter_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
        dropped_samples_val_dict = {temperature: [] for temperature in temperatures}
        auroc_val_dict = {temperature: [] for temperature in temperatures}
        auprc_val_dict = {temperature: [] for temperature in temperatures}
        accuracy_val_dict = {temperature: [] for temperature in temperatures}
    elif method_name in ("mrs-forest",):
        hyperparameter_list = [0.0, 0.001, 0.01, 0.025]
        draw_with_feature_weights = False
        temperatures = None
        dropped_samples_val_dict = {
            0.0: {hyperparameter: [] for hyperparameter in hyperparameter_list}
        }
        auroc_val_dict = {
            0.0: {hyperparameter: [] for hyperparameter in hyperparameter_list}
        }
        auprc_val_dict = {
            0.0: {hyperparameter: [] for hyperparameter in hyperparameter_list}
        }
        accuracy_val_dict = {
            0.0: {hyperparameter: [] for hyperparameter in hyperparameter_list}
        }
    elif method_name == "psa":
        draw_with_feature_weights = False
        temperatures = None
        hyperparameter_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    elif method_name == "kmm":
        draw_with_feature_weights = False
        temperatures = None
        hyperparameter_list = []
    else:
        draw_with_feature_weights = False
        temperatures = None
        hyperparameter_list = []
    return (
        draw_with_feature_weights,
        temperatures,
        dropped_samples_val_dict,
        auroc_val_dict,
        auprc_val_dict,
        accuracy_val_dict,
        hyperparameter_list,
    )
