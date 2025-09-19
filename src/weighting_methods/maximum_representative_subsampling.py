import random
import numpy as np
import pandas as pd

from tqdm import trange

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
)
from sklearn.metrics import roc_auc_score

from utils.metrics import (
    calculate_mean_roc,
    calculate_rbf_gamma,
    compute_relative_bias,
    interpolate_roc,
    train_pu_classifier_mrs,
    wasserstein_distance,
    weighted_maximum_mean_discrepancy,
)

# Used to draw random states
max_int = 2**32 - 1


def mrs_step(
    N,
    R,
    columns,
    target,
    n_drop: int = 1,
    n_splits=5,
    random_state=None,
    calculate_roc=False,
    sample_weights=None,
    hyperparameter=0.0,
    *args,
    **attributes
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
    ifpr_list = []
    itpr_list = []

    dropped_N = N[sample_weights != 0.0]

    y = dropped_N[target]
    target_sum = np.sum(y)
    if (target_sum <= n_splits) or ((len(dropped_N) - target_sum) <= n_splits):
        if calculate_roc:
            return None, None, None, None
        else:
            return None, None

    all_predictions = np.zeros(len(dropped_N))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for (train_indices_N, test_indices_N), (train_indices_R, test_indices_R) in zip(
        skf.split(dropped_N, dropped_N[target]), kf.split(R)
    ):
        N_train, N_test = (
            dropped_N.iloc[train_indices_N],
            dropped_N.iloc[test_indices_N],
        )
        R_train, R_test = R.iloc[train_indices_R], R.iloc[test_indices_R]
        train_data = pd.concat([N_train, R_train])
        clf = train_pu_classifier_mrs(
            train_data[columns],
            train_data.label,
            random_state=random_state,
            hyperparameter=hyperparameter,
        )
        test_data = pd.concat([N_test, R_test])
        predictions = clf.predict_proba(test_data[columns])[:, 1]
        all_predictions[test_indices_N] += predictions[: len(N_test)]
        auroc_list.append(roc_auc_score(test_data.label, predictions))

        if calculate_roc:
            interpolated_fpr, interpolated_tpr = interpolate_roc(
                test_data.label, predictions
            )
            ifpr_list.append(interpolated_fpr)
            itpr_list.append(interpolated_tpr)

    drop_ids = np.argpartition(all_predictions, -n_drop)[-n_drop:]

    if calculate_roc:
        mean_ifpr_list, mean_itpr_list, _ = calculate_mean_roc(ifpr_list, itpr_list)
        return (
            dropped_N.index[drop_ids],
            np.mean(auroc_list),
            mean_ifpr_list,
            mean_itpr_list,
        )
    else:
        return dropped_N.index[drop_ids], np.mean(auroc_list)


def mrs(
    N,
    R,
    columns,
    delta=0.0,
    early_stopping=False,
    mrs_function=mrs_step,
    return_metrics=False,
    compute_bias=True,
    target=None,
    n_pu_splits=5,
    drop=1,
    random_generator=None,
    hyperparameter_list=[0.0],
    wasserstein_target=None,
    *args,
    **attributes
):
    """Performs the whole mrs

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Name of the columns used in training
    :param delta: Delta for the stopping criterion, defaults to 0.001
    :param early_stopping: If true, stops before dropping all samples, defaults to False
    :param mrs_function: Function that is used in evers mrs iteration, defaults to mrs
    :param return_metrics: If true, return test metrics, defaults to False
    :param compute_bias: If true, compute relative bias, defaults to True
    :param bias_variable: Name of the biased variable, defaults to None
    :param cv: Number of cross-validation iterations, defaults to 5
    :param drop: Defines how many samples are dropped per iteration, defaults to 1
    :param random_generator: Random generator to create random_states to make results reproducible
    :return: Sample weights or test metrics
    """
    auc_dict = {}
    relative_bias_dict = {}
    mmd_dict = {}
    roc_dict = {}
    mrs_iteration_dict = {}
    sample_weights_dict = {}
    switched_dict = {}
    best_weights_dict = {}
    finished_dict = {}
    wasserstein_dict = {}

    mrs_function = random_drops if mrs_function == "random" else mrs_step
    wasserstein_target = target if wasserstein_target is None else wasserstein_target

    for hyperparameter in hyperparameter_list:
        auc_dict[hyperparameter] = []
        relative_bias_dict[hyperparameter] = []
        mmd_dict[hyperparameter] = []
        roc_dict[hyperparameter] = []
        mrs_iteration_dict[hyperparameter] = 0
        sample_weights_dict[hyperparameter] = np.ones(len(N))
        switched_dict[hyperparameter] = False
        finished_dict[hyperparameter] = False
        wasserstein_dict[hyperparameter] = []
        best_weights_dict[hyperparameter] = {}

    number_of_iterations = ((len(N) - n_pu_splits) // drop) - 1
    dropped_N = N.copy().reset_index(drop=True)
    roc_iteration = (len(N) // drop // 3.5) + 1

    # Compute and save mmd inputs to save time
    # Start values
    if return_metrics:
        for hyperparameter in hyperparameter_list:
            # Compute and save mmd inputs to save time
            gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))
            x_x_rbf_matrix = rbf_kernel(N[columns], N[columns], gamma=gamma)
            x_y_rbf_matrix = rbf_kernel(N[columns], R[columns], gamma=gamma)
            y_y_rbf_matrix = rbf_kernel(R[columns], R[columns], gamma=gamma)

    for i in trange(number_of_iterations):
        for hyperparameter in hyperparameter_list:
            if finished_dict[hyperparameter] and not return_metrics:
                continue
            if i % roc_iteration == 0 and return_metrics:
                (
                    drop_ids,
                    auroc,
                    mean_ifpr_list,
                    mean_itpr_list,
                ) = mrs_function(
                    N=dropped_N,
                    R=R,
                    columns=columns,
                    target=target,
                    n_drop=drop,
                    n_splits=n_pu_splits,
                    random_state=int(random_generator.randint(max_int, dtype=np.int64)),
                    calculate_roc=True,
                    sample_weights=sample_weights_dict[hyperparameter],
                    hyperparameter=hyperparameter,
                )
                if mean_ifpr_list:
                    roc_dict[hyperparameter].append(
                        [mean_ifpr_list.tolist(), mean_itpr_list.tolist(), i * drop]
                    )
                if drop_ids is None:
                    finished_dict[hyperparameter] = True
                    continue
            else:
                drop_ids, auroc = mrs_function(
                    N=dropped_N,
                    R=R,
                    columns=columns,
                    target=target,
                    n_drop=drop,
                    n_splits=n_pu_splits,
                    random_state=int(random_generator.randint(max_int, dtype=np.int64)),
                    sample_weights=sample_weights_dict[hyperparameter],
                    hyperparameter=hyperparameter,
                )

                if drop_ids is None:
                    finished_dict[hyperparameter] = True
                    continue

            if compute_bias and target is not None:
                relative_bias = compute_relative_bias(
                    dropped_N[wasserstein_target],
                    R[wasserstein_target],
                    sample_weights_dict[hyperparameter],
                )
                relative_bias_dict[hyperparameter].append(relative_bias)
                wasserstein_distance_value = wasserstein_distance(
                    N[wasserstein_target],
                    R[wasserstein_target],
                    sample_weights_dict[hyperparameter],
                )
                wasserstein_dict[hyperparameter].append(wasserstein_distance_value)

            auc_difference = abs(auroc - 0.5)
            if not switched_dict[hyperparameter]:
                best_weights_dict[hyperparameter][0] = (
                    sample_weights_dict[hyperparameter]
                    .copy()
                    .astype(np.float64)
                    .tolist()
                )
                mrs_iteration_dict[hyperparameter] = i * drop
                switched_dict[hyperparameter] = (
                    True
                    if auroc <= 0.5 and not switched_dict[hyperparameter]
                    else False
                )

            if return_metrics:
                auc_dict[hyperparameter].append(auroc)
                mmd_dict[hyperparameter].append(
                    weighted_maximum_mean_discrepancy(
                        dropped_N[columns],
                        R[columns],
                        sample_weights_dict[hyperparameter],
                        gamma=gamma,
                        x_x_rbf_matrix=x_x_rbf_matrix,
                        x_y_rbf_matrix=x_y_rbf_matrix,
                        y_y_rbf_matrix=y_y_rbf_matrix,
                    )
                )

            sample_weights_dict[hyperparameter][drop_ids] = 0.0
            remaining = dropped_N[sample_weights_dict[hyperparameter] != 0.0]
            if (
                len(remaining) <= drop
                or len(remaining) <= n_pu_splits
                or auc_difference <= delta
                or switched_dict[hyperparameter]
            ):
                finished_dict[hyperparameter] = True

        if all(finished for finished in finished_dict.values()) and early_stopping:
            break

    feature_weights = {}
    for hyperparameter in hyperparameter_list:
        feature_weights[hyperparameter] = np.ones(len(columns)).tolist()

    if return_metrics:
        return (
            auc_dict,
            mmd_dict,
            relative_bias_dict,
            mrs_iteration_dict,
            roc_dict,
            wasserstein_dict,
        )
    else:
        return (best_weights_dict, feature_weights)


def random_drops(
    N,
    R,
    columns,
    target,
    n_drop: int = 1,
    n_splits=5,
    random_state=None,
    calculate_roc=False,
    sample_weights=None,
    hyperparameter=0.0,
    *args,
    **attributes
):
    """MRS variant that drops sample randomly

    :param N: Non-representative data set
    :param n_drop: Defines how many samples are dropped per iteration, defaults to 1
    :return: Index of the samples to drop
    """

    auroc_list = []
    ifpr_list = []
    itpr_list = []

    dropped_N = N[sample_weights != 0.0]
    y = dropped_N[target]
    target_sum = np.sum(y)
    if (target_sum <= n_splits) or ((len(dropped_N) - target_sum) <= n_splits):
        if calculate_roc:
            return None, None, None, None
        else:
            return None, None
    all_predictions = np.zeros(len(dropped_N))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for (train_indices_N, test_indices_N), (train_indices_R, test_indices_R) in zip(
        skf.split(dropped_N, dropped_N[target]), kf.split(R)
    ):
        N_train, N_test = (
            dropped_N.iloc[train_indices_N],
            dropped_N.iloc[test_indices_N],
        )
        R_train, R_test = R.iloc[train_indices_R], R.iloc[test_indices_R]
        train_data = pd.concat([N_train, R_train])
        clf = train_pu_classifier_mrs(
            train_data[columns],
            train_data.label,
            random_state=random_state,
            hyperparameter=hyperparameter,
        )

        test_data = pd.concat([N_test, R_test])
        predictions = clf.predict_proba(test_data[columns])[:, 1]
        all_predictions[test_indices_N] += predictions[: len(N_test)]
        auroc_list.append(roc_auc_score(test_data.label, predictions))

        if calculate_roc:
            interpolated_fpr, interpolated_tpr = interpolate_roc(
                test_data.label, predictions
            )
            ifpr_list.append(interpolated_fpr)
            itpr_list.append(interpolated_tpr)

    drop_ids = random.sample(range(0, len(dropped_N)), n_drop)

    if calculate_roc:
        mean_ifpr_list, mean_itpr_list, _ = calculate_mean_roc(ifpr_list, itpr_list)
        return (
            dropped_N.index[drop_ids],
            np.mean(auroc_list),
            mean_ifpr_list,
            mean_itpr_list,
        )
    else:
        return dropped_N.index[drop_ids], np.mean(auroc_list)
