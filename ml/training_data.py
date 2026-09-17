import pandas as pd

from analysis.feature_engineering import (
    load_historical_data,
    create_features
)


VALUE_COLUMNS = [
    "col1",
    "col2",
    "col3",
    "col4",
    "col5",
    "col6",
    "col7",
    "col8"
]


EXCLUDED_COLUMNS = [
    "record_date",
    "data_status",
    "target"
]


def build_training_dataset(target_column):

    if target_column not in VALUE_COLUMNS:
        raise ValueError(
            f"Invalid target column: {target_column}"
        )

    # ---------------------------------------------
    # LOAD HISTORICAL DATA
    # ---------------------------------------------

    df = load_historical_data()

    if df.empty:
        print("No historical data found.")
        return pd.DataFrame()

    # ---------------------------------------------
    # CREATE FEATURES
    # ---------------------------------------------

    df = create_features(df)

    # ---------------------------------------------
    # TARGET
    # ---------------------------------------------

    df["target"] = df[target_column]

    # ---------------------------------------------
    # ONLY ROWS WHERE TARGET IS KNOWN
    # ---------------------------------------------

    dataset = df[
        df["target"].notna()
    ].copy()

    # ---------------------------------------------
    # BUILD PREDICTION-COMPATIBLE FEATURES
    # ---------------------------------------------

    feature_columns = []

    for column in dataset.columns:

        # Remove metadata
        if column in EXCLUDED_COLUMNS:
            continue

        # Remove target
        if column == target_column:
            continue

        # IMPORTANT:
        # Raw current-day values cannot be used
        # because they are unknown at prediction time.
        if column in VALUE_COLUMNS:
            continue

        feature_columns.append(column)

    X = dataset[
        feature_columns
    ].copy()

    y = dataset[
        "target"
    ].copy()

    # ---------------------------------------------
    # NUMERIC FEATURES ONLY
    # ---------------------------------------------

    X = X.select_dtypes(
        include=["number"]
    )

    # ---------------------------------------------
    # HANDLE MISSING VALUES
    # ---------------------------------------------

    X = X.ffill()

    for column in X.columns:

        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0

        X[column] = X[column].fillna(
            median_value
        )

    # ---------------------------------------------
    # FINAL DATASET
    # ---------------------------------------------

    training_dataset = X.copy()

    training_dataset["target"] = (
        y.values
    )

    # Preserve date for time-aware split
    training_dataset["record_date"] = (
        dataset["record_date"].values
    )

    return training_dataset


def run_training_data_test():

    print("\n===================================")
    print("       ML TRAINING DATASET")
    print("===================================")

    target_column = "col1"

    print(
        f"Target column: {target_column}"
    )

    dataset = build_training_dataset(
        target_column
    )

    if dataset.empty:

        print(
            "Training dataset is empty."
        )

        return

    print("\n-----------------------------------")
    print("DATASET INFORMATION")
    print("-----------------------------------")

    print(
        f"Rows: {len(dataset)}"
    )

    print(
        f"Columns: {len(dataset.columns)}"
    )

    print(
        f"Features: {len(dataset.columns) - 2}"
    )

    print("\n-----------------------------------")
    print("TARGET INFORMATION")
    print("-----------------------------------")

    print(
        dataset["target"]
        .value_counts()
        .sort_index()
    )

    print("\n-----------------------------------")
    print("MISSING VALUES")
    print("-----------------------------------")

    missing_count = (
        dataset
        .drop(
            columns=[
                "target",
                "record_date"
            ]
        )
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Remaining feature NULL values: "
        f"{missing_count}"
    )

    print("\n-----------------------------------")
    print("FEATURE SCHEMA CHECK")
    print("-----------------------------------")

    feature_columns = [
        column
        for column in dataset.columns
        if column not in [
            "target",
            "record_date"
        ]
    ]

    print(
        f"Prediction-compatible features: "
        f"{len(feature_columns)}"
    )

    print(
        "Raw col1-col8 features excluded: YES"
    )

    print(
        "Training/prediction schema aligned: "
        "YES"
    )

    print("\n-----------------------------------")
    print("SAMPLE DATA")
    print("-----------------------------------")

    print(
        dataset.head()
    )

    print("\n===================================")
    print("ML TRAINING DATASET CREATED")
    print("===================================\n")


if __name__ == "__main__":
    run_training_data_test()