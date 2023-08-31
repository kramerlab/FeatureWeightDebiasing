import pathlib

import pandas as pd
import numpy as np

from folktables import (
    ACSDataSource,
    generate_categories,
    BasicProblem,
    adult_filter,
    ACSEmployment,
)

file_path = pathlib.Path(__file__).parent


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
        return load_brast_cancer_data()
    elif dataset_name == "gbs_gesis":
        return load_gbs_gesis()
    elif dataset_name == "hr_analytics":
        return load_hr_analytics()
    elif dataset_name == "loan_prediction":
        return load_loan_prediction()
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
    return allensbach, allensbach_columns, ""


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
        "Wahlteilnahme",
        "BRS6",
    ]

    gesis = pd.read_csv(f"{file_path}/../../data/gesis_processed.csv", engine="python")
    gbs = pd.read_csv(f"{file_path}/../../data/gbs_processed.csv", engine="python")

    N = gbs.copy()
    R = gesis.copy()

    N["label"] = 1
    R["label"] = 0

    gesis_gbs = pd.concat([N, R], ignore_index=True)
    return gesis_gbs, gesis_columns, ""


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

    usa_features, us_labels, _ = ACSIncomeNew.df_to_pandas(
        usa_data, categories=categories, dummies=True
    )

    columns = usa_features.columns
    usa_features["Income"] = us_labels
    usa_features["Binary Income"] = [
        1 if us_label >= 50000 else 0 for us_label in us_labels.values
    ]
    usa_features = usa_features.dropna()

    return usa_features, columns, "Binary Income"


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
    dependent_replacing = {"3+": 3}
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

    features, us_labels, _ = ACSEmployment.df_to_pandas(
        data, categories=categories, dummies=True
    )

    columns = features.columns
    features["Employment"] = [
        1 if us_label == True else 0 for us_label in us_labels.values
    ]
    features = features.dropna()

    return features, columns, "Employment"


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


def load_brast_cancer_data():
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
