import json
import numpy as np

from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_weights, save_weights
from utils.parameter import set_parameter
from utils.statistics import create_result_path
from utils.sampling import repeated_train_val_test_split_fixed_test_set, sample_N
from utils.metrics import compute_decomposition_metrics_random_forest

seed = 5
sampling_random_generator = np.random.RandomState(seed)


def decomposition_experiment(
    df,
    columns,
    sample_weighting_method,
    target: str,
    n_cv_repeats: int,
    bias_type: str = None,
    data_set_name: str = "",
    random_generator=None,
    load_previous_results=True,
    bias_fraction=0.1,
    drop=1,
    method_name=None,
    **args,
):
    """The function uses the weighting method to compute the sample weights and
    computes the metrics, visualizes the results and saves the result in a file.

    :param df: pandas.DataFrame with the data
    :param columns: Name of training columns
    :param weighting_method: The weighting function
    :param target: Target name
    :param method: Method name, defaults to ""
    :param number_of_repetitions: Number of repetetions of the experiment,
        defaults to 100
    :param bias_type: Name of the bias that will be induced, defaults to None
    :param data_set_name: Data set name, defaults to ""
    """
    predictions_list = []
    probabilities_list = []

    result_path = create_result_path(
        method_name,
        bias_type,
        data_set_name,
        experiment_name="decomposition",
        bias_fraction=bias_fraction,
    )
    sample_weights_save_path = result_path / "sample_weights"
    feature_weights_save_path = result_path / "feature_weights"

    result_path.mkdir(exist_ok=True)
    sample_weights_save_path.mkdir(exist_ok=True)
    feature_weights_save_path.mkdir(exist_ok=True)

    sample_weight_list = load_weights(sample_weights_save_path)
    feature_weight_list = load_weights(feature_weights_save_path)

    scaler = StandardScaler()

    (
        splitter,
        draw_with_feature_weights,
        temperatures,
        _,
        _,
        _,
        hyperparameter_list,
    ) = set_parameter(method_name)
    if temperatures is not None:
        predictions_val_dict = {temperature: [] for temperature in temperatures}
        probabilities_val_dict = {temperature: [] for temperature in temperatures}
    elif method_name == "mrs-forest":
        predictions_val_dict = {0.0: []}
        probabilities_val_dict = {0.0: []}


    for i, (N, R, T) in enumerate(
        repeated_train_val_test_split_fixed_test_set(
            n_cv_repeats,
            df,
            df[target],
            sampling_random_generator,
        )
    ):
        N[columns] = scaler.fit_transform(N[columns])
        R[columns] = scaler.transform(R[columns])
        T[columns] = scaler.transform(T[columns])

        N = sample_N(
            train=N,
            bias_type=bias_type,
            bias_fraction=bias_fraction,
            columns=columns,
            bias_variable=target,
            random_generator=sampling_random_generator,
        )
        N["label"] = 1
        R["label"] = 0

        if len(sample_weight_list) > i and load_previous_results:
            sample_weights = sample_weight_list[i]
            feature_weights = feature_weight_list[i]

        else:
            sample_weights, feature_weights = sample_weighting_method(
                N=N,
                R=R,
                columns=columns,
                save_path=result_path,
                bias_variable=target,
                drop=drop,
                early_stopping=True,
                random_generator=random_generator,
                target=target,
                budgets=temperatures,
                hyperparameter_list=hyperparameter_list,
                method_name=method_name,
                compute_bias=False,
            )

            if method_name in ("mrs-forest", "psa"):
                sample_weights = {0.0: sample_weights}
                feature_weights = {0.0: feature_weights}

            feature_weight_list.append(feature_weights)
            sample_weight_list.append(sample_weights)

            save_weights(sample_weights_save_path, sample_weight_list)
            save_weights(feature_weights_save_path, feature_weight_list)

        predictions, probabilities = compute_decomposition_metrics_random_forest(
            N,
            T,
            columns,
            sample_weights,
            feature_weights,
            target,
            random_state=seed,
            draw_with_feature_weights=draw_with_feature_weights,
            splitter=splitter,
            n_estimators=500,
            n_splits=5,
        )
        predictions_list.append(predictions)
        probabilities_list.append(probabilities)

        if method_name in (
            "fw-mrs-temperature",
            "fw-mrs-temperature-svm",
            "mrs-forest",
        ):
            for temperature, temperature_sample_weights in sample_weights.items():
                temperature_feature_weights = {"tmp": feature_weights[temperature]}
                temperature_sample_weights = {"tmp": temperature_sample_weights}

                predictions, probabilities = (
                    compute_decomposition_metrics_random_forest(
                        N,
                        R,
                        columns,
                        temperature_sample_weights,
                        temperature_feature_weights,
                        target,
                        random_state=seed,
                        draw_with_feature_weights=draw_with_feature_weights,
                        splitter=splitter,
                        n_estimators=500,
                        n_splits=5,
                    )
                )

                predictions_val_dict[float(temperature)].append(predictions)
                probabilities_val_dict[float(temperature)].append(probabilities)

    predictions_list = np.array(predictions_list).squeeze()
    predictions_list = predictions_list.astype(int)
    probabilities_list = np.array(probabilities_list).squeeze()
    y_true = T[target].values

    main_predictions = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions_list
    ).squeeze()

    avg_expected_loss = np.apply_along_axis(
        lambda x: (x != y_true).mean(), axis=1, arr=predictions_list
    ).mean()
    avg_bias = np.sum(main_predictions != y_true) / y_true.size
    var = np.zeros_like(predictions_list[0])
    for pred in predictions_list:
        var += (pred != main_predictions).astype(np.int_)
    var = var / n_cv_repeats
    avg_var = np.sum(var) / y_true.shape[0]

    mse_avg_expected_loss = np.apply_along_axis(
        lambda x: ((x - y_true) ** 2).mean(), axis=1, arr=probabilities_list
    ).mean()
    mse_main_predictions = np.mean(probabilities_list, axis=0)
    mse_avg_bias = np.sum((mse_main_predictions - y_true) ** 2) / y_true.size
    mse_avg_var = (
        np.sum((mse_main_predictions - probabilities_list) ** 2)
        / probabilities_list.size
    )

    result_dict = {
        "0-1 loss": {
            "average expected loss": avg_expected_loss,
            "average_bias": avg_bias,
            "average variance": avg_var,
        },
        "mse loss": {
            "average expected loss": mse_avg_expected_loss,
            "average_bias": mse_avg_bias,
            "average variance": mse_avg_var,
        },
    }
    with open(result_path / "bias_variance_decomposition.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))
