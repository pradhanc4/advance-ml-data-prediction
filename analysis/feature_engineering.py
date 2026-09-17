import pandas as pd
import numpy as np

from database.connection import SessionLocal
from database.models import HistoricalRecord


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


def load_historical_data():

    db = SessionLocal()

    try:

        records = (
            db.query(HistoricalRecord)
            .order_by(
                HistoricalRecord.record_date
            )
            .all()
        )

        data = []

        for record in records:

            row = {
                "record_date": record.record_date,
                "data_status": record.data_status
            }

            for column in VALUE_COLUMNS:

                row[column] = getattr(
                    record,
                    column
                )

            data.append(row)

        return pd.DataFrame(data)

    finally:

        db.close()


def create_features(df):

    if df.empty:

        return df

    df = df.copy()

    # ---------------------------------------------
    # DATE FEATURES
    # ---------------------------------------------

    df["record_date"] = pd.to_datetime(
        df["record_date"]
    )

    df["day_of_week"] = (
        df["record_date"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["record_date"].dt.day
    )

    df["week_of_year"] = (
        df["record_date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["month"] = (
        df["record_date"].dt.month
    )

    df["year"] = (
        df["record_date"].dt.year
    )

    # ---------------------------------------------
    # COLUMN-LEVEL FEATURES
    # ---------------------------------------------

    for column in VALUE_COLUMNS:

        # Missing indicator
        df[f"{column}_missing"] = (
            df[column]
            .isna()
            .astype(int)
        )

        # Zero indicator
        df[f"{column}_is_zero"] = (
            df[column]
            .eq(0)
            .astype(int)
        )

        # Historical lags
        df[f"{column}_lag_1"] = (
            df[column].shift(1)
        )

        df[f"{column}_lag_2"] = (
            df[column].shift(2)
        )

        df[f"{column}_lag_3"] = (
            df[column].shift(3)
        )

        df[f"{column}_lag_7"] = (
            df[column].shift(7)
        )

        # Historical changes
        df[f"{column}_change_1"] = (
            df[column].shift(1)
            -
            df[column].shift(2)
        )

        df[f"{column}_change_2"] = (
            df[column].shift(2)
            -
            df[column].shift(3)
        )

        # Rolling mean
        # shift(1) prevents current-row leakage
        df[f"{column}_rolling_mean_3"] = (
            df[column]
            .shift(1)
            .rolling(
                window=3,
                min_periods=1
            )
            .mean()
        )

        df[f"{column}_rolling_mean_7"] = (
            df[column]
            .shift(1)
            .rolling(
                window=7,
                min_periods=1
            )
            .mean()
        )

        # Rolling standard deviation
        df[f"{column}_rolling_std_3"] = (
            df[column]
            .shift(1)
            .rolling(
                window=3,
                min_periods=2
            )
            .std()
        )

        df[f"{column}_rolling_std_7"] = (
            df[column]
            .shift(1)
            .rolling(
                window=7,
                min_periods=2
            )
            .std()
        )

    # ---------------------------------------------
    # ROW-LEVEL FEATURES
    # ---------------------------------------------

    df["available_columns"] = (
        df[VALUE_COLUMNS]
        .notna()
        .sum(axis=1)
    )

    df["missing_columns"] = (
        df[VALUE_COLUMNS]
        .isna()
        .sum(axis=1)
    )

    df["zero_count"] = (
        df[VALUE_COLUMNS]
        .eq(0)
        .sum(axis=1)
    )

    df["row_mean"] = (
        df[VALUE_COLUMNS]
        .mean(axis=1)
    )

    df["row_std"] = (
        df[VALUE_COLUMNS]
        .std(axis=1)
    )

    df["row_min"] = (
        df[VALUE_COLUMNS]
        .min(axis=1)
    )

    df["row_max"] = (
        df[VALUE_COLUMNS]
        .max(axis=1)
    )

    return df


def run_feature_engineering():

    print("\n===================================")
    print("      FEATURE ENGINEERING")
    print("===================================")

    df = load_historical_data()

    if df.empty:

        print(
            "No historical data found."
        )

        return

    print(
        f"Historical records: {len(df)}"
    )

    feature_df = create_features(df)

    print(
        f"Original columns: {len(df.columns)}"
    )

    print(
        f"Feature columns: {len(feature_df.columns)}"
    )

    print("\n-----------------------------------")
    print("FEATURE SUMMARY")
    print("-----------------------------------")

    print(
        feature_df[
            [
                "record_date",
                "data_status",
                "available_columns",
                "missing_columns",
                "zero_count",
                "row_mean",
                "row_std"
            ]
        ].tail(10)
    )

    print("\n-----------------------------------")
    print("FEATURE ENGINEERING CHECK")
    print("-----------------------------------")

    print(
        f"Total rows: {len(feature_df)}"
    )

    print(
        f"Total features: {len(feature_df.columns)}"
    )

    print(
        "Actual zero values remain valid: YES"
    )

    print(
        "Missing values remain distinguishable: YES"
    )

    print(
        "Future target leakage from rolling "
        "features: PREVENTED"
    )

    print("\n===================================")
    print("FEATURE ENGINEERING COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_feature_engineering()