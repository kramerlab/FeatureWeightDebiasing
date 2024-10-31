def set_parameter(method_name):
    dropped_samples_val_dict = None
    auroc_val_dict = None
    auprc_val_dict = None
    if method_name in ("fw-mrs-temperature", "fw-mrs-temperature-mean"):
        splitter = "feature_weighted_best"
        draw_with_feature_weights = True
        temperatures = [0.1, 0.05, 0.01, 0.005]
        dropped_samples_val_dict = {temperature: [] for temperature in temperatures}
        auroc_val_dict = {temperature: [] for temperature in temperatures}
        auprc_val_dict = {temperature: [] for temperature in temperatures}
        hyperparameter_list = [0.05, 0.025, 0.0]
    if method_name == "fw-mrs-temperature-svm":
        splitter = "feature_weighted_best"
        draw_with_feature_weights = True
        temperatures = [0.1, 0.05, 0.01, 0.005]
        dropped_samples_val_dict = {temperature: [] for temperature in temperatures}
        auroc_val_dict = {temperature: [] for temperature in temperatures}
        auprc_val_dict = {temperature: [] for temperature in temperatures}
        hyperparameter_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    elif method_name == "mrs-forest":
        hyperparameter_list = [0.05, 0.025, 0.0]
        splitter = "best"
        draw_with_feature_weights = False
        temperatures = None
        dropped_samples_val_dict = {0.0: []}
        auroc_val_dict = {0.0: []}
        auprc_val_dict = {0.0: []}
    elif method_name == "psa":
        splitter = "best"
        draw_with_feature_weights = False
        temperatures = None
        hyperparameter_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    elif method_name == "kmm":
        splitter = "best"
        draw_with_feature_weights = False
        temperatures = None
        hyperparameter_list = []
    else:
        splitter = "best"
        draw_with_feature_weights = False
        temperatures = None
        hyperparameter_list = []
    return (
        splitter,
        draw_with_feature_weights,
        temperatures,
        dropped_samples_val_dict,
        auroc_val_dict,
        auprc_val_dict,
        hyperparameter_list,
    )
