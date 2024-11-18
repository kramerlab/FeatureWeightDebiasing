def set_parameter(method_name):
    dropped_samples_val_dict = None
    auroc_val_dict = None
    auprc_val_dict = None
    accuracy_val_dict = None
    if method_name in ("fw-mrs-temperature",):
        draw_with_feature_weights = True
        temperatures = [0.1, 0.05, 0.01, 0.005]
        hyperparameter_list = [0.05, 0.025, 0.0]
        dropped_samples_val_dict = {
            temperature: {parameter: [] for parameter in hyperparameter_list}
            for temperature in temperatures
        }
        auroc_val_dict = {
            temperature: {parameter: [] for parameter in hyperparameter_list}
            for temperature in temperatures
        }
        auprc_val_dict = {
            temperature: {parameter: [] for parameter in hyperparameter_list}
            for temperature in temperatures
        }
        accuracy_val_dict = {
            temperature: {parameter: [] for parameter in hyperparameter_list}
            for temperature in temperatures
        }
    elif method_name == "fw-mrs-temperature-svm":
        draw_with_feature_weights = True
        temperatures = [0.1, 0.05, 0.01, 0.005]
        hyperparameter_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
        dropped_samples_val_dict = {
            temperature: {parameter: [] for parameter in hyperparameter_list}
            for temperature in temperatures
        }
        auroc_val_dict = {
            temperature: {parameter: [] for parameter in hyperparameter_list}
            for temperature in temperatures
        }
        auprc_val_dict = {
            temperature: {parameter: [] for parameter in hyperparameter_list}
            for temperature in temperatures
        }
        accuracy_val_dict = {
            temperature: {parameter: [] for parameter in hyperparameter_list}
            for temperature in temperatures
        }
    elif method_name == "mrs-forest":
        hyperparameter_list = [0.05, 0.025, 0.0]
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
