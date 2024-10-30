import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

seed = 5


def sample(
    bias_type,
    df,
    bias_variable,
    bias_fraction=0.1,
    train_fraction=0.25,
    columns=None,
    random_generator=None,
):
    """Samples a biased and a representative data set.

    :param bias_type: Defines how the data should be biased
    :param df: Data set as pandas.DataFrame
    :param bias_variable: The target variable
    :param bias_fraction: Defines which fraction of the biased class is samples, defaults to 0.1
    :param train_fraction: Defines the size of the train set, defaults to 0.25
    :param columns: Columns that are used to compute the mean sample, defaults to None
    :return: A biased and a representative data set
    """
    # Sample from the data set because the complete one is too big.
    if len(df) > 5000:
        df = df.sample(5000, random_state=random_generator, replace=False)
    train = df.groupby(bias_variable, group_keys=False).apply(
        lambda x: x.sample(
            frac=train_fraction, random_state=random_generator, replace=False
        )
    )
    R = df.drop(train.index).reset_index(drop=True)
    N = sample_N(
        bias_type,
        bias_fraction,
        columns,
        train,
        bias_variable,
    )

    N["label"] = 1
    R["label"] = 0

    return N, R


def sample_N(bias_type, bias_fraction, columns, train, bias_variable, random_generator):
    positive_samples = train[train[bias_variable] == 1]
    negative_samples = train[train[bias_variable] == 0]
    if bias_type in ("less_positive_class", "less_negative_class"):
        if bias_type == "less_positive_class":
            positive_fraction = bias_fraction
            negative_fraction = 1
        elif bias_type == "less_negative_class":
            positive_fraction = 1
            negative_fraction = bias_fraction

        N = sample_class_biased_N(
            positive_samples,
            negative_samples,
            positive_fraction=positive_fraction,
            negative_fraction=negative_fraction,
            random_generator=random_generator,
        )
    elif bias_type == "mean_difference":
        N = less_outlier_sampling(train, bias_fraction, columns, random_generator)
    else:
        N = train.reset_index(drop=True)
    return N


def less_outlier_sampling(train, bias_fraction, columns, random_generator):
    mean_sample = train[columns].mean().values
    differences = (
        np.linalg.norm(
            train[columns].values - mean_sample,
            axis=1,
        )
        ** 3
    )
    temperature = -(1 / 20)
    sample_weights = np.exp(temperature * differences)
    N = train.sample(
        frac=bias_fraction,
        weights=sample_weights,
        random_state=random_generator,
    )
    return N


def sample_with_test_set(
    bias_type,
    df,
    bias_variable,
    bias_fraction=0.1,
    train_fraction=0.25,
    test_fraction=0.1,
    columns=None,
    random_generator=None,
):
    """Samples a biased and a representative data set.

    :param bias_type: Defines how the data should be biased
    :param df: Data set as pandas.DataFrame
    :param bias_variable: The target variable
    :param bias_fraction: Defines which fraction of the biased class is samples, defaults to 0.1
    :param train_fraction: Defines the size of the train set, defaults to 0.25
    :param columns: Columns that are used to compute the mean sample, defaults to None
    :return: A biased and a representative data set
    """
    T = (
        df.groupby(bias_variable, group_keys=False)
        .apply(
            lambda x: x.sample(
                frac=test_fraction,
                random_state=random_generator,
                replace=False,
            )
        )
        .copy()
    )
    df_without_T = df.drop(T.index).copy()
    R = (
        df_without_T.groupby(bias_variable, group_keys=False)
        .apply(
            lambda x: x.sample(
                frac=1 - train_fraction,
                random_state=random_generator,
                replace=False,
            )
        )
        .copy()
    )
    train = df_without_T.drop(R.index).copy()

    N = sample_N(
        bias_type,
        bias_fraction,
        columns,
        train,
        bias_variable,
    )

    N["label"] = 1
    R["label"] = 0

    return N, R, T


def sample_class_biased_N(
    positive_samples,
    negative_samples,
    positive_fraction,
    negative_fraction,
    random_generator,
):
    """Samples a biased data set

    :param positive_samples: Samples of the positive class
    :param negative_samples: Samples of the negative class
    :param positive_fraction: Fraction for positive class
    :param negative_fraction: Fraction for negative class
    :return: The sampled biased data ste
    """
    N = (
        pd.concat(
            [
                positive_samples.sample(
                    frac=positive_fraction, random_state=random_generator
                ),
                negative_samples.sample(
                    frac=negative_fraction, random_state=random_generator
                ),
            ]
        )
        .copy()
        .reset_index(drop=True)
    )

    return N


def repeated_train_val_test_split(
    n_cv_splits, n_cv_repeats, df, target_values, random_generator
):
    # Is used to draw radom states
    max_int = 2**32 - 1
    for _ in range(n_cv_repeats):
        skf = StratifiedKFold(
            n_splits=n_cv_splits,
            shuffle=True,
            random_state=random_generator.randint(max_int),
        )
        for train_val_index, test_index in skf.split(df, target_values):
            train_val_samples = df.iloc[train_val_index].copy()
            train_val_target_values = target_values.iloc[train_val_index]
            test_samples = df.iloc[test_index].copy()

            train_samples, val_samples, _, _ = train_test_split(
                train_val_samples,
                train_val_target_values,
                stratify=train_val_target_values,
                random_state=random_generator.randint(max_int),
                test_size=0.5,
            )

            yield train_samples, val_samples, test_samples
