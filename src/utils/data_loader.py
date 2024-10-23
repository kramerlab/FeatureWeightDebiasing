import json
import pathlib

import pandas as pd
import numpy as np
from fairlearn.datasets import fetch_adult

from folktables import (
    ACSDataSource,
    generate_categories,
    BasicProblem,
    adult_filter,
    ACSEmployment,
)

file_path = pathlib.Path(__file__).parent
seed = 5
upper_sample_limit = 6000


def load_dataset(dataset_name):
    """Load the data set to a given name.

    :param dataset_name: Data set name
    :return: Data set as pandas.DataFrame
    """
    if dataset_name == "gbs_allensbach":
        return load_gbs_allensbach()
    elif dataset_name == "folktables_income":
        return load_folktables_income_data()
    elif dataset_name == "folktables_employment":
        return load_folktables_employment_data()
    elif dataset_name == "breast_cancer":
        return load_breast_cancer_data()
    elif dataset_name == "gbs_gesis":
        return load_gbs_gesis()
    elif dataset_name == "hr_analytics":
        return load_hr_analytics()
    elif dataset_name == "loan_prediction":
        return load_loan_prediction()
    elif dataset_name == "fairness_adult":
        return load_fairness_adult()
    else:
        print("No valid data set name given!")
        exit()


def load_gbs_allensbach():
    """Load GBS and allensbach

    :return: GBS and Allensbach data
    """
    allensbach_path = f"{file_path}/../../data/allensbach_mrs.csv"
    allensbach = pd.read_csv(allensbach_path)
    allensbach.drop(["Unnamed: 0", "Gruppe", "GBS-CODE"], axis=1, inplace=True)
    allensbach_columns = [
        "Alter",
        "Berufsgruppe",
        "Erwerbstaetigkeit",
        "Geschlecht",
        "Optimismus",
        "Pessimismus",
        "Schulabschluss",
        "woechentlicheArbeitszeit",
        "Resilienz",
    ]
    return allensbach, allensbach_columns, "Wahlteilnahme"


def load_gbs_gesis():
    """Load GBS and GESIS

    :return: GBS and GESIS data
    """
    gesis_columns = [
        "Geschlecht",
        "Geburtsjahr",
        "Geburtsland",
        "Nationalitaet",
        "Familienstand",
        "Hoechster Bildungsabschluss",
        "Berufliche Ausbildung",
        "Erwerbstaetigkeit",
        "Nettoeinkommen Selbst",
        "Zufriedenheit Wahlergebnis",
        "Gesellig",
        "Andere kritisieren",
        "Gruendlich",
        "Nervoes",
        "Phantasievoll",
        "Berufsgruppe",
        "BRS6",
    ]

    gesis = pd.read_csv(f"{file_path}/../../data/gesis_processed.csv", engine="python")
    gbs = pd.read_csv(f"{file_path}/../../data/gbs_processed.csv", engine="python")

    N = gbs.copy()
    R = gesis.copy()
    N["BRS6"] = 6 - N["BRS6"]
    N[N["Erwerbstaetigkeit"] == 4] = 3
    N = N.drop(N[N["Wahlteilnahme"] == 3].index)

    N["label"] = 1
    R["label"] = 0

    gesis_gbs = pd.concat([N, R], ignore_index=True)
    return gesis_gbs, gesis_columns, "Wahlteilnahme"


def load_folktables_income_data():
    """Load folktable income

    :return: Folktable income data
    """
    # Create a new income problem loader.
    ACSIncomeNew = BasicProblem(
        features=[
            "AGEP",
            "COW",
            "SCHL",
            "MAR",
            "RELP",
            "WKHP",
            "SEX",
            "RAC1P",
        ],
        target="PINCP",
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    usa_data = data_source.get_data(states=["CA"], download=True)
    definition_df = data_source.get_definitions(download=True)
    categories = generate_categories(
        features=ACSIncomeNew.features, definition_df=definition_df
    )

    df, us_labels, _ = ACSIncomeNew.df_to_pandas(
        usa_data, categories=categories, dummies=True
    )

    columns = df.columns
    df["Binary Income"] = [
        1 if us_label >= 50000 else 0 for us_label in us_labels.values
    ]
    df = df.dropna()
    df = df.replace({False: 0, True: 1}).infer_objects(copy=False)

    if len(df) > upper_sample_limit:
        df = df.sample(upper_sample_limit, random_state=seed).copy()

    return df, columns, "Binary Income"


def load_hr_analytics():
    """Load HR Analytics

    :return: HR ANalytics data
    """
    categorical_variables = [
        "gender",
        "relevent_experience",
        "enrolled_university",
        "education_level",
        "major_discipline",
        "company_size",
        "company_type",
        "last_new_job",
    ]
    experience_replacing = {"<1": 0, ">20": 21}
    hr_analystics = pd.read_csv(
        f"{file_path}/../../data/hr_analytics.csv", engine="python"
    )
    hr_analystics = hr_analystics.drop(columns=["enrollee_id", "city"])
    hr_analystics = hr_analystics.dropna()
    hr_analystics["experience"] = hr_analystics["experience"].replace(
        experience_replacing
    )
    hr_analystics = pd.get_dummies(
        hr_analystics, columns=categorical_variables, drop_first=True
    )

    columns = hr_analystics.drop(columns=["target"]).columns
    return hr_analystics, columns, "target"


def load_loan_prediction():
    """Load Loan

    :return: Loan data
    """
    categorical_columns = [
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]
    target = "Loan_Status"
    dependent_replacing = {"0": 0, "1": 1, "2": 2, "3+": 3}
    target_replacing = {"Y": 1, "N": 0}
    loan_predictions = pd.read_csv(
        f"{file_path}/../../data/loan_prediction.csv", engine="python"
    )
    loan_predictions = loan_predictions.drop(columns=["Loan_ID"])
    loan_predictions = loan_predictions.dropna()
    loan_predictions = pd.get_dummies(
        loan_predictions, columns=categorical_columns, drop_first=True
    )
    loan_predictions["Dependents"] = loan_predictions["Dependents"].replace(
        dependent_replacing
    )

    loan_predictions[target] = loan_predictions[target].replace(target_replacing)
    loan_predictions = loan_predictions.drop(columns=[])
    columns = loan_predictions.drop(columns=[target]).columns

    loan_predictions.replace({False: 0, True: 1}, inplace=True)
    return loan_predictions, columns, target


def load_folktables_employment_data():
    """Load folktables exployment

    :return: Folktables employment data
    """
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    data = data_source.get_data(states=["CA"], download=True)
    definition_df = data_source.get_definitions(download=True)
    categories = generate_categories(
        features=ACSEmployment.features, definition_df=definition_df
    )

    df, us_labels, _ = ACSEmployment.df_to_pandas(
        data, categories=categories, dummies=True
    )

    columns = df.columns
    df["Employment"] = [1 if us_label == True else 0 for us_label in us_labels.values]
    df = df.dropna()
    df = df.replace({False: 0, True: 1}).infer_objects(copy=False)

    if len(df) > upper_sample_limit:
        df = df.sample(upper_sample_limit, random_state=seed).copy()
    return df, columns, "Employment"


breast_cancer_names = [
    "sample_code_number",
    "clump_thickness",
    "uniformity_of_cell_size",
    "uniformity_of_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class",
]


def load_breast_cancer_data():
    """Load Wisconsin Breast cancer

    :return: Wisconsin Breast cancer data
    """
    df = pd.read_csv(
        f"{file_path}/../../data/breast_cancer/breast-cancer-wisconsin.data",
        na_values=["?"],
        names=breast_cancer_names,
    )
    df = df.dropna()
    replace_dict = {2: 1, 4: 0}
    df["class"] = df["class"].replace(replace_dict)
    df = df.drop(columns=["sample_code_number"])
    columns = df.drop(columns=["class"]).columns
    df[columns] = df[columns]

    return df, columns, "class"


def load_fairness_adult():
    data = fetch_adult(as_frame=True)
    X = pd.get_dummies(data.data)
    X = X.replace({False: 0, True: 1}).infer_objects(copy=False)
    X = X.drop(columns="fnlwgt")
    columns = X.columns
    y_true = (data.target == ">50K") * 1
    sex = data.data["sex"]
    sex = sex.replace({"Male": 0, "Female": 1}).infer_objects(copy=False)

    df = pd.concat([X, sex, pd.DataFrame({"income": y_true})], axis=1)

    return df, columns, ("sex", "income")


def save_weights(path, weights_list, file_name="weights"):
    with open(path / f"{file_name}.json", "w") as file:
        json.dump(weights_list, file, indent=4)


def load_weights(path, file_name="weights"):
    weight_file = path / f"{file_name}.json"
    if weight_file.is_file():
        with open(weight_file, "r") as file:
            weights = json.load(file)
        weights = {int(k):v for k,v in weights.items()}
    else:
        weights = []
    return weights
