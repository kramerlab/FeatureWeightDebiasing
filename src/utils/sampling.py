import numpy as np
import pandas as pd

seed = 5
sampling_random_generator = np.random.RandomState(seed)


def sample(
    bias_type, df, bias_variable, bias_fraction=0.1, train_fraction=0.25, columns=None
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
        df = df.sample(5000, random_state=sampling_random_generator, replace=False)
    train = df.groupby(bias_variable, group_keys=False).apply(
        lambda x: x.sample(
            frac=train_fraction, random_state=sampling_random_generator, replace=False
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


def sample_N(bias_type, bias_fraction, columns, train, bias_variable):
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
        )
    elif bias_type == "mean_difference":
        N = less_outlier_sampling(train, bias_fraction, columns)
    else:
        N = train.reset_index(drop=True)
    return N


def less_outlier_sampling(train, bias_fraction, columns):
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
        frac=bias_fraction + 0.4,
        weights=sample_weights,
        random_state=sampling_random_generator,
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
    upper_sample_limit = 8000 
    if len(df) > upper_sample_limit:
        df = df.sample(upper_sample_limit, random_state=sampling_random_generator).copy()
    T = df.groupby(bias_variable, group_keys=False).apply(
        lambda x: x.sample(
            frac=test_fraction, random_state=sampling_random_generator, replace=False
        )
    ).copy()
    df_without_T = df.drop(T.index).copy()
    R = df_without_T.groupby(bias_variable, group_keys=False).apply(
        lambda x: x.sample(
            frac=1 - train_fraction,
            random_state=sampling_random_generator,
            replace=False,
        )
    ).copy()
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
    positive_samples, negative_samples, positive_fraction, negative_fraction
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
                    frac=positive_fraction, random_state=sampling_random_generator
                ),
                negative_samples.sample(
                    frac=negative_fraction, random_state=sampling_random_generator
                ),
            ]
        )
        .copy()
        .reset_index(drop=True)
    )

    return N
