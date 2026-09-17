import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    mutual_info_classif,
    VarianceThreshold
)

from ml.training_data import build_training_dataset


def run_advanced_feature_selection(
    target_column="col1",
    top_n=30
):

    print("\n===================================")
    print("     ADVANCED FEATURE SELECTION")
    print("===================================")

    print(
        f"Target column: {target_column}"
    )

    dataset = build_training_dataset(
        target_column
    )

    if dataset.empty:
        print("No training data available.")
        return

    feature_columns = [
        column
        for column in dataset.columns
        if column not in [
            "target",
            "record_date"
        ]
    ]

    X = dataset[
        feature_columns
    ].copy()

    y = dataset[
        "target"
    ].copy()

    print("\n-----------------------------------")
    print("ORIGINAL DATA")
    print("-----------------------------------")

    print(
        f"Rows: {len(X)}"
    )

    print(
        f"Original features: {len(X.columns)}"
    )

    # -----------------------------------
    # NUMERIC FEATURES
    # -----------------------------------

    X = X.select_dtypes(
        include=["number"]
    )

    # Safety cleaning
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.ffill()

    for column in X.columns:

        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0

        X[column] = X[column].fillna(
            median_value
        )

    # -----------------------------------
    # VARIANCE FILTER
    # -----------------------------------

    print("\n-----------------------------------")
    print("VARIANCE FILTER")
    print("-----------------------------------")

    variance_filter = VarianceThreshold(
        threshold=0.0
    )

    variance_filter.fit(X)

    variance_columns = X.columns[
        variance_filter.get_support()
    ]

    X_variance = X[
        variance_columns
    ].copy()

    removed_variance = (
        len(X.columns)
        -
        len(X_variance.columns)
    )

    print(
        f"Features before filter: "
        f"{len(X.columns)}"
    )

    print(
        f"Features after filter: "
        f"{len(X_variance.columns)}"
    )

    print(
        f"Constant features removed: "
        f"{removed_variance}"
    )

    if X_variance.empty:

        print(
            "\nNo features remain after variance filtering."
        )

        return

    # -----------------------------------
    # MUTUAL INFORMATION
    # -----------------------------------

    print("\n-----------------------------------")
    print("MUTUAL INFORMATION")
    print("-----------------------------------")

    if len(X_variance) < 5:

        print(
            "Dataset is too small for reliable "
            "mutual-information analysis."
        )

        mutual_information = np.zeros(
            len(X_variance.columns)
        )

    else:

        try:

            mutual_information = (
                mutual_info_classif(
                    X_variance,
                    y,
                    random_state=42
                )
            )

        except Exception as error:

            print(
                "Mutual information failed:"
            )

            print(error)

            mutual_information = np.zeros(
                len(X_variance.columns)
            )

    mi_scores = pd.Series(
        mutual_information,
        index=X_variance.columns
    )

    # -----------------------------------
    # RANDOM FOREST IMPORTANCE
    # -----------------------------------

    print("\n-----------------------------------")
    print("RANDOM FOREST IMPORTANCE")
    print("-----------------------------------")

    try:

        classes = sorted(
            y.unique()
        )

        class_to_index = {
            value: index
            for index, value in enumerate(classes)
        }

        y_encoded = y.map(
            class_to_index
        )

        if len(classes) >= 2:

            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )

            rf.fit(
                X_variance,
                y_encoded
            )

            rf_importance = pd.Series(
                rf.feature_importances_,
                index=X_variance.columns
            )

        else:

            rf_importance = pd.Series(
                0.0,
                index=X_variance.columns
            )

    except Exception as error:

        print(
            "Random Forest importance failed:"
        )

        print(error)

        rf_importance = pd.Series(
            0.0,
            index=X_variance.columns
        )

    # -----------------------------------
    # NORMALIZE SCORES
    # -----------------------------------

    def normalize_series(series):

        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:

            return pd.Series(
                0.0,
                index=series.index
            )

        return (
            (series - minimum)
            /
            (maximum - minimum)
        )

    mi_normalized = normalize_series(
        mi_scores
    )

    rf_normalized = normalize_series(
        rf_importance
    )

    # -----------------------------------
    # COMBINED SCORE
    # -----------------------------------

    combined_score = (
        0.50 * mi_normalized
        +
        0.50 * rf_normalized
    )

    ranking = pd.DataFrame({

        "feature":
            X_variance.columns,

        "mutual_information":
            mi_scores.values,

        "rf_importance":
            rf_importance.values,

        "combined_score":
            combined_score.values
    })

    ranking = ranking.sort_values(
        "combined_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    # -----------------------------------
    # SELECT FEATURES
    # -----------------------------------

    selected_count = min(
        top_n,
        len(ranking)
    )

    selected_features = (
        ranking.head(
            selected_count
        )["feature"].tolist()
    )

    # -----------------------------------
    # RESULTS
    # -----------------------------------

    print("\n===================================")
    print("FEATURE IMPORTANCE RANKING")
    print("===================================")

    display_count = min(
        20,
        len(ranking)
    )

    print(
        ranking.head(
            display_count
        ).to_string(
            index=False
        )
    )

    print("\n-----------------------------------")
    print("SELECTED FEATURES")
    print("-----------------------------------")

    print(
        f"Requested top features: "
        f"{top_n}"
    )

    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    for index, feature in enumerate(
        selected_features,
        start=1
    ):

        score = ranking.loc[
            ranking["feature"] == feature,
            "combined_score"
        ].iloc[0]

        print(
            f"{index:02d}. "
            f"{feature} "
            f"-> score={score:.4f}"
        )

    # -----------------------------------
    # FEATURE GROUP SUMMARY
    # -----------------------------------

    print("\n-----------------------------------")
    print("FEATURE GROUP SUMMARY")
    print("-----------------------------------")

    groups = {
        "Calendar": [
            "day_of_week",
            "day_of_month",
            "week_of_year",
            "month",
            "year"
        ],

        "Lag": [
            "_lag_"
        ],

        "Change": [
            "_change_"
        ],

        "Rolling": [
            "_rolling_"
        ],

        "Missing": [
            "_missing"
        ],

        "Zero": [
            "_is_zero"
        ],

        "Row Statistics": [
            "available_columns",
            "missing_columns",
            "zero_count",
            "row_mean",
            "row_std",
            "row_min",
            "row_max"
        ]
    }

    for group_name, patterns in groups.items():

        count = 0

        for feature in selected_features:

            if any(
                pattern in feature
                for pattern in patterns
            ):

                count += 1

        print(
            f"{group_name}: {count}"
        )

    # -----------------------------------
    # FINAL SUMMARY
    # -----------------------------------

    print("\n-----------------------------------")
    print("SELECTION SUMMARY")
    print("-----------------------------------")

    print(
        f"Original features: "
        f"{len(feature_columns)}"
    )

    print(
        f"After variance filter: "
        f"{len(X_variance.columns)}"
    )

    print(
        f"Final selected features: "
        f"{len(selected_features)}"
    )

    if len(dataset) < 100:

        print("\nWARNING:")
        print(
            "Dataset is currently small."
        )

        print(
            "Feature importance can change significantly "
            "when more historical data is added."
        )

        print(
            "Treat this selection as experimental."
        )

    print("\n===================================")
    print(
        "ADVANCED FEATURE SELECTION COMPLETED"
    )
    print("===================================\n")


if __name__ == "__main__":

    run_advanced_feature_selection(
        target_column="col1",
        top_n=30
    )