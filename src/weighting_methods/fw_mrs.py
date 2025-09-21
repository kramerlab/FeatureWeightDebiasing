import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from tqdm import trange

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
)
from utils.metrics import (
    calculate_rbf_gamma,
    compute_feature_weights_with_temperature,
    train_pu_classifier,
    weighted_maximum_mean_discrepancy,
)
from sklearn.metrics.pairwise import rbf_kernel

# Used to draw radom states
max_int = 2**32 - 1


def mrs_step(
    N,
    R,
    columns,
    target,
    n_drop: int = 1,
    n_splits=5,
    random_state=None,
    feature_weight=None,
    splitter="feature_weighted_best",
    sample_weights=None,
    compute_feature_importance=False,
    hyperparameter=0.0,
    *args,
    **attributes,
):
    """Performs one iteration of maximum representative sampling

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Columns names used for training
    :param n_drop: Number of samples to drop every iteration, defaults to 1
    :param cv: Number of cross-validation iterations, defaults to 5
    :param random_state: Random state to make results reproducible
    :return: _description_
    """
    auroc_list = []
    feature_importance_list = []
    dropped_N = N[sample_weights != 0.0]
    all_predictions = np.zeros(len(dropped_N))

    y = dropped_N[target]
    target_sum = np.sum(y)
    if target_sum <= n_splits:
        n_splits = int(target_sum)
    elif (len(dropped_N) - target_sum) <= n_splits:
        n_splits = int(len(dropped_N) - target_sum)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for (train_indices_N, test_indices_N), (train_indices_R, test_indices_R) in zip(
        skf.split(dropped_N, dropped_N[target]), kf.split(R, R[target])
    ):
        N_train, N_test = (
            dropped_N.iloc[train_indices_N],
            dropped_N.iloc[test_indices_N],
        )
        R_train, R_test = R.iloc[train_indices_R], R.iloc[test_indices_R]
        train_data = pd.concat([N_train, R_train])
        clf = train_pu_classifier(
            train_data[columns],
            train_data.label,
            random_state=random_state,
            feature_weight=feature_weight,
            splitter=splitter,
            hyperparameter=hyperparameter,
        )
        test_data = pd.concat([N_test, R_test])
        predictions = clf.predict_proba(test_data[columns])[:, 1]
        all_predictions[test_indices_N] = predictions[: len(N_test)]
        auroc_list.append(roc_auc_score(test_data.label, predictions))

        if compute_feature_importance:
            explainer = shap.TreeExplainer(clf, data=train_data[columns])
            explainer = explainer(test_data[columns].values, check_additivity=False)
            shap_values = np.abs(explainer.values[:, :, 1])
            abs_feature_importance = np.mean(shap_values, axis=0)
            feature_importance_list.append(abs_feature_importance)

    mean_auroc = np.mean(auroc_list)
    if compute_feature_importance:
        mean_feature_importance = np.mean(feature_importance_list, axis=0)
    else:
        mean_feature_importance = None
    drop_ids = np.argpartition(all_predictions, -n_drop)[-n_drop:]

    return dropped_N.index[drop_ids], mean_feature_importance, mean_auroc


def feature_weighted_repeated_MRS(
    N,
    R,
    target,
    columns,
    delta=0.0,
    early_stopping=False,
    drop=1,
    budgets=[1.0],
    random_generator=None,
    n_pu_splits=5,
    hyperparameter_list=[],
    return_metrics=False,
    *args,
    **attributes,
):
    """Performs MRS

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Name of the columns used in training
    :param delta: Delta for the stopping criterion, defaults to 0.001
    :param early_stopping: If true, stops before dropping all samples, defaults to False
    :param mrs_function: Function that is used in evers mrs iteration, defaults to mrs
    :param return_metrics: If true, return test metrics, defaults to False
    :param use_bias_mean: If true, compute relative bias, defaults to True
    :param bias_variable: Name of the biased variable, defaults to None
    :param cv: Number of cross-validation iterations, defaults to 5
    :param drop: Defines how many samples are dropped per iteration, defaults to 1
    :param random_generator: Random generator to create random_states to make results reproducible
    :return: Sample weights or test metrics
    """
    number_of_iterations = (len(N) - (n_pu_splits + 1)) // drop
    dropped_N = N.copy().reset_index(drop=True)

    best_difference_dict = {}
    best_sample_weights_dict = {}
    dropped_samples_dict = {}
    auc_difference_dict = {}
    abs_feature_importance_dict = {}
    sample_weights_dict = {}
    feature_weights_dict = {}
    feature_weighted_aurocs_dict = {}
    finished_dict = {}
    switched_dict = {}
    auroc_dict = {}
    mmd_dict = {}
    gamma_dict = {}
    x_x_rbf_matrix_dict = {}
    x_y_rbf_matrix_dict = {}
    y_y_rbf_matrix_dict = {}

    initialize_dictionaries(
        N,
        R,
        columns,
        target,
        drop,
        budgets,
        random_generator,
        n_pu_splits,
        hyperparameter_list,
        dropped_N,
        best_difference_dict,
        best_sample_weights_dict,
        dropped_samples_dict,
        auc_difference_dict,
        abs_feature_importance_dict,
        sample_weights_dict,
        feature_weights_dict,
        feature_weighted_aurocs_dict,
        finished_dict,
        switched_dict,
        auroc_dict,
        mmd_dict,
        gamma_dict=gamma_dict,
        x_x_rbf_matrix_dict=x_x_rbf_matrix_dict,
        x_y_rbf_matrix_dict=x_y_rbf_matrix_dict,
        y_y_rbf_matrix_dict=y_y_rbf_matrix_dict,
        mrs_step=mrs_step,
    )

    if return_metrics:
        for temperature in budgets:
            for hyperparameter in hyperparameter_list:
                # Compute and save mmd inputs to save time
                gamma = calculate_rbf_gamma(
                    np.append(N[columns], R[columns], axis=0),
                    feature_weights_dict[temperature][hyperparameter],
                )
                gamma_dict[temperature][hyperparameter] = gamma
                scaled_feature_weights = (
                    feature_weights_dict[temperature][hyperparameter]
                    / np.sum(feature_weights_dict[temperature][hyperparameter])
                    * len(feature_weights_dict[temperature][hyperparameter])
                )
                scaled_N = N[columns] * np.sqrt(scaled_feature_weights)
                scaled_R = R[columns] * np.sqrt(scaled_feature_weights)
                x_x_rbf_matrix_dict[temperature][hyperparameter] = rbf_kernel(
                    scaled_N, scaled_N, gamma=gamma_dict[temperature][hyperparameter]
                )
                x_y_rbf_matrix_dict[temperature][hyperparameter] = rbf_kernel(
                    scaled_N, scaled_R, gamma=gamma_dict[temperature][hyperparameter]
                )
                y_y_rbf_matrix_dict[temperature][hyperparameter] = rbf_kernel(
                    scaled_R, scaled_R, gamma=gamma_dict[temperature][hyperparameter]
                )

    for i in trange(number_of_iterations):
        for temperature in budgets:
            for hyperparameter in hyperparameter_list:
                if finished_dict[temperature][hyperparameter] and not return_metrics:
                    continue
                splitter = "best" if temperature is None else "feature_weighted_best"
                drop_ids, _, auroc = mrs_step(
                    N=dropped_N,
                    R=R,
                    target=target,
                    columns=columns,
                    n_drop=drop,
                    random_state=int(random_generator.randint(max_int, dtype=np.int64)),
                    n_splits=n_pu_splits,
                    feature_weight=np.array(
                        feature_weights_dict[temperature][hyperparameter]
                    ),
                    splitter=splitter,
                    sample_weights=sample_weights_dict[temperature][hyperparameter],
                    hyperparameter=hyperparameter,
                )

                if drop_ids is None:
                    finished_dict[temperature][hyperparameter] = True
                    continue
                feature_weighted_aurocs_dict[temperature][hyperparameter].append(auroc)
                auc_difference = abs(auroc - 0.5)

                if return_metrics:
                    auroc_dict[temperature][hyperparameter].append(auroc)
                    mmd = weighted_maximum_mean_discrepancy(
                        dropped_N[columns],
                        R[columns],
                        sample_weights_dict[temperature][hyperparameter],
                        feature_weights=feature_weights_dict[temperature][
                            hyperparameter
                        ],
                        gamma=gamma_dict[temperature][hyperparameter],
                        x_x_rbf_matrix=x_x_rbf_matrix_dict[temperature][hyperparameter],
                        x_y_rbf_matrix=x_y_rbf_matrix_dict[temperature][hyperparameter],
                        y_y_rbf_matrix=y_y_rbf_matrix_dict[temperature][hyperparameter],
                    )
                    mmd_dict[temperature][hyperparameter].append(mmd)

                if (
                    (
                        auc_difference
                        <= best_difference_dict[temperature][hyperparameter]
                        and auc_difference <= delta
                    )
                    or (not switched_dict[temperature][hyperparameter] and auroc <= 0.5)
                ) and not finished_dict[temperature][hyperparameter]:
                    best_difference_dict[temperature][hyperparameter] = auc_difference
                    dropped_samples_dict[temperature][hyperparameter] = i * drop
                    sample_weights = sample_weights_dict[temperature][hyperparameter]
                    best_sample_weights_dict[temperature][hyperparameter][0] = (
                        (sample_weights / np.sum(sample_weights)).tolist().copy()
                    )
                    if not switched_dict[temperature][hyperparameter] and auroc <= 0.5:
                        switched_dict[temperature][hyperparameter] = True

                remaining = dropped_N[
                    sample_weights_dict[temperature][hyperparameter] != 0.0
                ]
                if (
                    len(remaining) <= drop
                    or len(remaining) <= n_pu_splits
                    or auc_difference <= delta
                    or switched_dict[temperature][hyperparameter]
                ):
                    finished_dict[temperature][hyperparameter] = True

                sample_weights_dict[temperature][hyperparameter][drop_ids] = 0

        if (
            all(all(finished.values()) for finished in finished_dict.values())
            and early_stopping
        ):
            break

    if return_metrics:
        return (
            auroc_dict,
            mmd_dict,
            best_sample_weights_dict,
            feature_weights_dict,
            abs_feature_importance_dict,
        )
    else:
        return (
            best_sample_weights_dict,
            feature_weights_dict,
        )


def initialize_dictionaries(
    N,
    R,
    columns,
    target,
    drop,
    budgets,
    random_generator,
    n_pu_splits,
    hyperparameter_list,
    dropped_N,
    best_difference_dict,
    best_sample_weights_dict,
    dropped_samples_dict,
    auc_difference_dict,
    abs_feature_importance_dict,
    sample_weights_dict,
    feature_weights_dict,
    feature_weighted_aurocs_dict,
    finished_dict,
    switched_dict={},
    auc_dict={},
    mmd_dict={},
    gamma_dict={},
    x_x_rbf_matrix_dict={},
    x_y_rbf_matrix_dict={},
    y_y_rbf_matrix_dict={},
    mrs_step=mrs_step,
):
    for temperature in budgets:
        best_difference_dict[temperature] = {}
        auc_difference_dict[temperature] = {}
        dropped_samples_dict[temperature] = {}
        feature_weighted_aurocs_dict[temperature] = {}
        sample_weights_dict[temperature] = {}
        abs_feature_importance_dict[temperature] = {}
        feature_weights_dict[temperature] = {}
        best_sample_weights_dict[temperature] = {}
        finished_dict[temperature] = {}
        switched_dict[temperature] = {}
        auc_dict[temperature] = {}
        mmd_dict[temperature] = {}
        gamma_dict[temperature] = {}
        x_x_rbf_matrix_dict[temperature] = {}
        x_y_rbf_matrix_dict[temperature] = {}
        y_y_rbf_matrix_dict[temperature] = {}
        for hyperparameter in hyperparameter_list:
            finished_dict[temperature][hyperparameter] = False
            switched_dict[temperature][hyperparameter] = False
            best_difference_dict[temperature][hyperparameter] = np.inf
            auc_difference_dict[temperature][hyperparameter] = 1
            dropped_samples_dict[temperature][hyperparameter] = 0
            auc_dict[temperature][hyperparameter] = []
            mmd_dict[temperature][hyperparameter] = []
            feature_weighted_aurocs_dict[temperature][hyperparameter] = []
            sample_weights_dict[temperature][hyperparameter] = np.ones(len(N))
            abs_feature_importance_dict[temperature][hyperparameter] = np.ones(
                len(columns)
            ).tolist()
            best_sample_weights_dict[temperature][hyperparameter] = {}

    random_state = int(int(random_generator.randint(max_int, dtype=np.int64)))
    for hyperparameter in hyperparameter_list:
        _, abs_feature_importance, _ = mrs_step(
            N=dropped_N,
            R=R,
            target=target,
            columns=columns,
            n_drop=drop,
            random_state=random_state,
            n_splits=n_pu_splits,
            feature_weight=np.ones(len(columns)),
            splitter="best",
            sample_weights=np.ones(len(N)),
            compute_feature_importance=True,
            hyperparameter=hyperparameter,
            class_weights="balanced",
        )

        for temperature in budgets:
            feature_weights_dict[temperature][hyperparameter] = (
                compute_feature_weights_with_temperature(
                    temperature, np.array(abs_feature_importance)
                ).tolist()
            )
            abs_feature_importance_dict[temperature][
                hyperparameter
            ] = abs_feature_importance.tolist()
