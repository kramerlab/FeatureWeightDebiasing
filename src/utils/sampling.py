import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

seed = 5


def sample_N(bias_type, bias_fraction, columns, train, bias_variable, random_generator):
    """Sample from fata to create the non-representative data set N

    :param bias_type: Bias Type
    :param bias_fraction: Bias Strength
    :param columns: Feature column names
    :param train: Train data set
    :param bias_variable: The variable to bias on
    :param random_generator: Random generator for repeatability
    :return: Non-representative data set N
    """
    positive_samples = train[train[bias_variable] == 1]
    negative_samples = train[train[bias_variable] == 0]
    if bias_type in (
        "less_positive_class",
        "less_negative_class",
    ):
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
    else:
        N = train.reset_index(drop=True)
    return N


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
    :param random_generator:
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
    """Repeated cross_validation

    :param n_cv_splits: Number of cross-validation splits
    :param n_cv_repeats: Number of repetetions
    :param df: Data frame
    :param target_values: Target values
    :param random_generator: Random generator for repeatability
    :yield: Returns the training, validation and test set of the current iteration
    """
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


def repeated_train_val_test_split_fixed_test_set(
    n_cv_repeats, df, target_values, random_generator
):
    """Repeated sampling with fixed test set for decomposition

    :param n_cv_repeats: Number repetetions
    :param df: Data frame
    :param target_values: Target values
    :param random_generator: Random generator for reproducibility
    :yield: Current training, validation and fixed test set
    """
    # Is used to draw radom states
    max_int = 2**32 - 1

    train_val_samples, test_samples, train_val_targets, _ = train_test_split(
        df,
        target_values,
        stratify=target_values,
        random_state=random_generator.randint(max_int),
        test_size=(1 / 3),
    )
    for _ in range(n_cv_repeats):
        train_samples, val_samples, _, _ = train_test_split(
            train_val_samples,
            train_val_targets,
            stratify=train_val_targets,
            random_state=random_generator.randint(max_int),
            test_size=0.5,
        )

        yield train_samples.copy(), val_samples.copy(), test_samples.copy()
