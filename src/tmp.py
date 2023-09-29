import random
import numpy as np
from loguru import logger
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
from tqdm import trange
from utils.data_loader import load_dataset
from utils.metrics import calculate_rbf_gamma, compute_metrics
from utils.sampling import sample
from methods.soft_mrs import soft_mrs_weighting
from pathlib import Path

seed = 5
random_generator = np.random.RandomState(seed)
np.random.seed(seed)
random.seed(seed)


def compute_feature_drop():
    data_set_name = "folktables_income"
    method = "drop_features"
    bias_type = "mean_difference"
    result_path = create_result_path(method, bias_type, data_set_name)
    df, columns, target = load_dataset(data_set_name)
    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    data = df.sample(5000).copy()
    data[columns] = scaler.transform(data[columns])

    N, R = sample(
        bias_type,
        data,
        target,
        train_fraction=0.5,
        bias_fraction=0.1,
        columns=columns,
    )
    gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))
    weights = np.ones(len(N)) / len(N)
    (
        mmd,
        _,
    ) = compute_metrics(
        N.copy(),
        R.copy(),
        weights,
        scaler,
        columns,
        columns,
        gamma=gamma,
    )
    logger.info(f"Unweighted MMD: {mmd}")


    random_indices = np.random.choice(len(columns), len(columns), replace=False)
    random_drop_columns = columns[random_indices]

    random_mmd_list = drop_columns(N, R, random_drop_columns, weights, scaler, gamma)
    sorted_mmd_list = drop_columns_mi(N, R, weights, scaler, gamma, columns)

    x = list(range(len(random_drop_columns), 0, -1))
    sns.lineplot(x=x, y=sorted_mmd_list, label="Sorted")
    ax = sns.lineplot(x=x, y=random_mmd_list, label="Random")
    ax.get_figure().savefig(f"{result_path}/mmd_feature_drop.pdf")


def drop_columns_mi(N, R, weights, scaler, gamma, columns):
    sorted_mmd_list = []
    remaining_columns = columns

    N = N[columns]
    R = R[columns]
    (
        weighted_mmd,
        _,
    ) = compute_metrics(
        N.copy(),
        R.copy(),
        weights,
        scaler,
        columns,
        columns,
        gamma,
    )
    sorted_mmd_list.insert(0, weighted_mmd)
    
    for _ in trange(1, len(columns)):
        X = np.concatenate([N[remaining_columns], R[remaining_columns]])
        y = np.concatenate([np.ones(len(N)), np.zeros(len(R))])
        mis = mutual_info_classif(X, y, random_state=seed)
        sorted_indices = np.argsort(mis)
        sorted_drop_columns = remaining_columns[sorted_indices]

        sorted_unbiased_features = sorted_drop_columns[:-1]
        remaining_columns = sorted_unbiased_features
        N = N[sorted_unbiased_features]
        R = R[sorted_unbiased_features]
        (
            weighted_mmd,
            _,
        ) = compute_metrics(
            N.copy(),
            R.copy(),
            weights,
            scaler,
            sorted_unbiased_features,
            sorted_unbiased_features,
            gamma,
        )
        sorted_mmd_list.insert(0, weighted_mmd)
    return sorted_mmd_list

def drop_columns(N, R, drop_columns_list, weights, scaler, gamma):
    sorted_mmd_list = []
    N = N[drop_columns_list]
    R = R[drop_columns_list]

    (
        weighted_mmd,
        _,
    ) = compute_metrics(
        N.copy(),
        R.copy(),
        weights,
        scaler,
        drop_columns_list,
        drop_columns_list,
        gamma,
    )
    sorted_mmd_list.insert(0, weighted_mmd)
    for i in trange(1, len(drop_columns_list)):
        sorted_unbiased_features = drop_columns_list[:-i]
        N = N[sorted_unbiased_features]
        R = R[sorted_unbiased_features]

        (
            weighted_mmd,
            _,
        ) = compute_metrics(
            N.copy(),
            R.copy(),
            weights,
            scaler,
            sorted_unbiased_features,
            sorted_unbiased_features,
            gamma,
        )
        sorted_mmd_list.insert(0, weighted_mmd)
    return sorted_mmd_list


def compute_feature_weights():
    data_set_name = "folktables_income"
    method = "soft-mrs"
    bias_type = "mean_difference"
    result_path = create_result_path(method, bias_type, data_set_name)
    df, columns, target = load_dataset(data_set_name)
    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    data = df.sample(5000).copy()
    data[columns] = scaler.transform(data[columns])

    N, R = sample(
        bias_type,
        data,
        target,
        train_fraction=0.5,
        bias_fraction=0.1,
        columns=columns,
    )
    gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))
    weights = np.ones(len(N)) / len(N)
    (
        mmd,
        _,
    ) = compute_metrics(
        N.copy(),
        R.copy(),
        weights,
        scaler,
        columns,
        columns,
        gamma,
    )

    # Sample weights
    soft_mrs_weights = soft_mrs_weighting(
        N,
        R,
        columns,
        save_path=result_path,
        bias_variable=target,
        drop=1,
        early_stopping=True,
        random_generator=random_generator,
        patience=25,
    )

    (
        sample_weighted_mmd,
        _,
    ) = compute_metrics(
        N.copy(),
        R.copy(),
        soft_mrs_weights,
        scaler,
        columns,
        columns,
        gamma,
    )

    # Feature weights
    X = np.concatenate([N[columns], R[columns]])
    y = np.concatenate([np.ones(len(N)), np.zeros(len(R))])
    mis = mutual_info_classif(X, y, random_state=seed)

    weight_updates = 1 - (mis * 100)
    weight_updates[weight_updates < 0] = 0
    N[columns] = N[columns] * weight_updates
    R[columns] = R[columns] * weight_updates

    (
        feature_weighted_mmd,
        _,
    ) = compute_metrics(
        N.copy(),
        R.copy(),
        weights,
        scaler,
        columns,
        columns,
        gamma,
    )

    (
        sample_and_feature_weighted_mmd,
        _,
    ) = compute_metrics(
        N.copy(),
        R.copy(),
        soft_mrs_weights,
        scaler,
        columns,
        columns,
        gamma,
    )

    logger.info(f"Unweighted MMD: {mmd}")
    logger.info(f"Weighted Samples {sample_weighted_mmd}")
    logger.info(f"Weighted Features {feature_weighted_mmd}")
    logger.info(f"Weighted Samples & Features {sample_and_feature_weighted_mmd}")


def create_result_path(method_name, bias_type, data_set_name):
    """The function creates the save path and makes the directory.

    :param method: Method name
    :param bias_type: Bias type name
    :param data_set_name: Data set name
    :return: The result path
    """
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../results")
    result_path = result_path / method_name / "downstream" / data_set_name / bias_type
    result_path.mkdir(exist_ok=True, parents=True)
    return result_path


if __name__ == "__main__":
    # compute_feature_weights()
    compute_feature_drop()
