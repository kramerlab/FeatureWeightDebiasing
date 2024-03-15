import json
from tqdm import trange
from sklearn.discriminant_analysis import StandardScaler
from utils.statistics import create_result_path
from utils.sampling import sample
from weighting_methods.feature_weighted_maximum_representative_subsampling import (
    feature_weighted_repeated_MRS,
)


seed = 5
budgets = [0.0, 0.5, 0.9, 1]


def feature_weight_budget_comparison_experiment(
    df,
    columns,
    target: str,
    number_of_repetitions: int = 50,
    bias_type: str = None,
    data_set_name: str = "",
    random_generator=None,
    method_name=None,
    drop=1,
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

    result_path = create_result_path(
        method_name, bias_type, data_set_name, experiment_name="budget_comparison"
    )

    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    sample_df = df.copy()

    for i in trange(number_of_repetitions):
        N, R = sample(
            bias_type,
            sample_df,
            target,
            train_fraction=0.5,
            bias_fraction=0.1,
            columns=columns,
        )

        feature_weighted_aurocs, feature_weight_list = feature_weighted_repeated_MRS(
            N=N,
            R=R,
            columns=columns,
            save_path=result_path,
            bias_variable=target,
            drop=drop,
            early_stopping=False,
            random_generator=random_generator,
            max_patience=len(N),
            target=target,
            budgets=budgets,
            return_auroc=True,
        )

        with open(result_path / f"aurocs_{i}.json", "w", encoding="utf-8") as file:
            json.dump(feature_weighted_aurocs, file, indent=4)
