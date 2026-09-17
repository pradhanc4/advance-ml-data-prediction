import numpy as np
import pandas as pd

from analysis.feature_engineering import (
    load_historical_data,
    create_features
)


# ===================================
# CONFIGURATION
# ===================================

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
    "data_status"
]

EXPECTED_FEATURE_COUNT = 108


# ===================================
# LOAD PREDICTION INPUT
# ===================================

def prepare_prediction_input():

    print("\n===================================")
    print("      PREDICTION INPUT")
    print("===================================")

    df = load_historical_data()

    if df.empty:

        print(
            "No historical data found."
        )

        return None

    df = df.sort_values(
        "record_date"
    ).reset_index(
        drop=True
    )

    print(
        f"Historical records loaded: "
        f"{len(df)}"
    )

    latest_record = df.iloc[-1]

    print(
        f"Latest record date: "
        f"{latest_record['record_date']}"
    )

    print("\nLatest prediction input:")

    display_columns = [
        "record_date",
        "data_status"
    ] + VALUE_COLUMNS

    print(
        df[
            display_columns
        ].tail(1).to_string(
            index=False
        )
    )

    # ===================================
    # REQUIRED COLUMN CHECK
    # ===================================

    missing_columns = [
        column
        for column in display_columns
        if column not in df.columns
    ]

    if missing_columns:

        print(
            "Missing required columns:"
        )

        print(
            missing_columns
        )

        return None

    print(
        "All required columns present: YES"
    )

    print(
        "SQL historical data loaded: YES"
    )

    print(
        "Latest record identified: YES"
    )

    return df


# ===================================
# CREATE PREDICTION FEATURES
# ===================================

def prepare_prediction_features():

    print("\n===================================")
    print("      PREDICTION FEATURES")
    print("===================================")

    df = prepare_prediction_input()

    if df is None:

        return None

    # ===================================
    # CREATE FEATURES
    # ===================================

    feature_df = create_features(
        df
    )

    if feature_df.empty:

        print(
            "Feature generation returned "
            "an empty dataset."
        )

        return None

    # ===================================
    # GET LATEST FEATURE ROW
    # ===================================

    latest_features = (
        feature_df
        .tail(1)
        .copy()
        .reset_index(
            drop=True
        )
    )

    # ===================================
    # REMOVE NON-PREDICTION COLUMNS
    # ===================================

    prediction_feature_columns = []

    for column in latest_features.columns:

        if column in EXCLUDED_COLUMNS:

            continue

        if column in VALUE_COLUMNS:

            continue

        prediction_feature_columns.append(
            column
        )

    X = latest_features[
        prediction_feature_columns
    ].copy()

    # ===================================
    # KEEP NUMERIC FEATURES ONLY
    # ===================================

    X = X.select_dtypes(
        include=["number"]
    )

    print(
        f"Historical rows: "
        f"{len(feature_df)}"
    )

    print(
        f"Total feature columns: "
        f"{len(feature_df.columns)}"
    )

    print(
        f"Prediction feature columns: "
        f"{X.shape[1]}"
    )

    print(
        f"Prediction date: "
        f"{latest_features['record_date'].iloc[0]}"
    )

    # ===================================
    # HANDLE MISSING VALUES
    # ===================================

    X = X.ffill()

    for column in X.columns:

        median_value = X[column].median()

        if pd.isna(median_value):

            median_value = 0

        X[column] = (
            X[column]
            .fillna(
                median_value
            )
        )

    # ===================================
    # NULL VALIDATION
    # ===================================

    remaining_nulls = (
        X.isna()
        .sum()
        .sum()
    )

    print("\n-----------------------------------")
    print("FEATURE VALIDATION")
    print("-----------------------------------")

    print(
        f"Remaining NULL values: "
        f"{remaining_nulls}"
    )

    if remaining_nulls > 0:

        print(
            "Feature validation: FAILED"
        )

        return None

    # ===================================
    # INFINITE VALUE VALIDATION
    # ===================================

    infinite_values = np.isinf(
        X.to_numpy()
    ).sum()

    print(
        f"Infinite values: "
        f"{infinite_values}"
    )

    if infinite_values > 0:

        print(
            "Feature validation: FAILED"
        )

        return None

    # ===================================
    # NUMERIC VALIDATION
    # ===================================

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns

    numeric_only = (
        len(numeric_columns)
        == len(X.columns)
    )

    print(
        f"Numeric features only: "
        f"{'YES' if numeric_only else 'NO'}"
    )

    if not numeric_only:

        print(
            "Feature validation: FAILED"
        )

        return None

    # ===================================
    # FEATURE COUNT VALIDATION
    # ===================================

    print(
        f"Expected feature count: "
        f"{EXPECTED_FEATURE_COUNT}"
    )

    print(
        f"Actual feature count: "
        f"{X.shape[1]}"
    )

    if (
        X.shape[1]
        != EXPECTED_FEATURE_COUNT
    ):

        print(
            "Feature count validation: FAILED"
        )

        return None

    print(
        "Feature count validation: PASSED"
    )

    # ===================================
    # FINAL FEATURE ROW
    # ===================================

    print(
        "Historical feature generation: YES"
    )

    print(
        "Prediction feature row created: YES"
    )

    print(
        "Feature validation: PASSED"
    )

    print("\n===================================")
    print(
        "PREDICTION FEATURES COMPLETED"
    )
    print("===================================\n")

    return X


# ===================================
# RUN PREDICTION ENGINE TEST
# ===================================

def run_prediction_engine_test():

    print("\n===================================")
    print("       PREDICTION ENGINE")
    print("===================================")

    X = prepare_prediction_features()

    if X is None:

        print(
            "Prediction engine failed."
        )

        return

    print(
        f"Final prediction shape: "
        f"{X.shape}"
    )

    print(
        "Prediction engine status: "
        "READY"
    )

    print("\n===================================")
    print(
        "PREDICTION ENGINE TEST COMPLETED"
    )
    print("===================================\n")


# ===================================
# DIRECT EXECUTION
# ===================================

if __name__ == "__main__":

    run_prediction_engine_test()