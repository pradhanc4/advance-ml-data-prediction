import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss
)

from ml.training_data import build_training_dataset


MODEL_NAMES = [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost"
]


def load_oof_predictions():

    file_path = "ml/oof_predictions.csv"

    if not os.path.exists(file_path):

        print("\nERROR:")
        print(
            "OOF prediction file was not found."
        )

        print(
            "Run Phase 8.7 first:"
        )

        print(
            "python -m ml.oof_predictions"
        )

        return None

    df = pd.read_csv(
        file_path
    )

    if df.empty:

        print(
            "OOF prediction file is empty."
        )

        return None

    return df


def get_probability_columns(
    oof_data,
    model_name
):

    prefix = (
        f"{model_name}_prob_"
    )

    return [
        column
        for column in oof_data.columns
        if column.startswith(prefix)
    ]


def build_meta_features(
    oof_data
):

    meta_features = pd.DataFrame(
        index=oof_data.index
    )

    # -----------------------------------
    # MODEL PREDICTIONS
    # -----------------------------------

    for model_name in MODEL_NAMES:

        prediction_column = (
            f"{model_name}_prediction"
        )

        if prediction_column in oof_data.columns:

            meta_features[
                prediction_column
            ] = oof_data[
                prediction_column
            ]

    # -----------------------------------
    # MODEL PROBABILITIES
    # -----------------------------------

    for model_name in MODEL_NAMES:

        probability_columns = (
            get_probability_columns(
                oof_data,
                model_name
            )
        )

        for column in probability_columns:

            meta_features[column] = (
                oof_data[column]
            )

    return meta_features


def run_stacking_ensemble(
    target_column="col1"
):

    print("\n===================================")
    print("        STACKING ENSEMBLE")
    print("===================================")

    print(
        f"Target column: {target_column}"
    )

    # -----------------------------------
    # LOAD OOF DATA
    # -----------------------------------

    oof_data = (
        load_oof_predictions()
    )

    if oof_data is None:
        return

    print("\n-----------------------------------")
    print("OOF DATA")
    print("-----------------------------------")

    print(
        f"OOF rows: {len(oof_data)}"
    )

    print(
        f"OOF columns: {len(oof_data.columns)}"
    )

    # -----------------------------------
    # TARGET
    # -----------------------------------

    if "actual_target" not in oof_data.columns:

        print(
            "actual_target column is missing."
        )

        return

    y = oof_data[
        "actual_target"
    ].copy()

    # -----------------------------------
    # BUILD META FEATURES
    # -----------------------------------

    meta_features = build_meta_features(
        oof_data
    )

    if meta_features.empty:

        print(
            "No meta-features available."
        )

        return

    print("\n-----------------------------------")
    print("META FEATURES")
    print("-----------------------------------")

    print(
        f"Meta-features: "
        f"{len(meta_features.columns)}"
    )

    # -----------------------------------
    # CLEAN META FEATURES
    # -----------------------------------

    meta_features = (
        meta_features
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    meta_features = (
        meta_features
        .ffill()
    )

    meta_features = (
        meta_features
        .fillna(0)
    )

    # -----------------------------------
    # TARGET CLASSES
    # -----------------------------------

    classes = sorted(
        y.unique()
    )

    class_count = len(
        classes
    )

    print(
        f"Target classes: "
        f"{class_count}"
    )

    if class_count < 2:

        print(
            "\nAt least 2 target classes are required."
        )

        return

    class_to_index = {
        value: index
        for index, value in enumerate(classes)
    }

    index_to_class = {
        index: value
        for value, index in class_to_index.items()
    }

    y_encoded = y.map(
        class_to_index
    )

    # -----------------------------------
    # CHECK DATA SIZE
    # -----------------------------------

    if len(meta_features) < 10:

        print("\nWARNING:")
        print(
            "Very few OOF records are available."
        )

        print(
            "The stacking model is experimental."
        )

    # -----------------------------------
    # META MODEL
    # -----------------------------------

    print("\n===================================")
    print("TRAINING META MODEL")
    print("===================================")

    meta_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    )

    try:

        meta_model.fit(
            meta_features,
            y_encoded
        )

    except Exception as error:

        print(
            "\nMeta-model training failed:"
        )

        print(error)

        return

    print(
        "Meta model: Logistic Regression"
    )

    print(
        "Meta model training completed."
    )

    # -----------------------------------
    # META MODEL PROBABILITIES
    # -----------------------------------

    probabilities = (
        meta_model.predict_proba(
            meta_features
        )
    )

    probabilities = (
        probabilities /
        probabilities.sum(
            axis=1,
            keepdims=True
        )
    )

    predicted_indices = (
        probabilities.argmax(
            axis=1
        )
    )

    predictions = [
        index_to_class[
            index
        ]
        for index in predicted_indices
    ]

    # -----------------------------------
    # RESULTS
    # -----------------------------------

    print("\n===================================")
    print("STACKING RESULTS")
    print("===================================")

    for index in range(
        len(y)
    ):

        actual = y.iloc[index]

        predicted = predictions[index]

        confidence = (
            probabilities[
                index
            ].max()
            * 100
        )

        result = (
            "CORRECT"
            if actual == predicted
            else "WRONG"
        )

        print(
            f"{index + 1}. "
            f"Actual={actual}, "
            f"Prediction={predicted}, "
            f"Probability={confidence:.2f}% "
            f"-> {result}"
        )

    # -----------------------------------
    # TOP PREDICTIONS
    # -----------------------------------

    print("\n-----------------------------------")
    print("TOP STACKING PREDICTIONS")
    print("-----------------------------------")

    for index in range(
        len(y)
    ):

        ranking = (
            probabilities[index]
            .argsort()[::-1]
        )

        print(
            f"\nOOF Record {index + 1}:"
        )

        for rank, class_index in enumerate(
            ranking[:5],
            start=1
        ):

            predicted_class = (
                index_to_class[
                    class_index
                ]
            )

            probability = (
                probabilities[
                    index,
                    class_index
                ]
                * 100
            )

            print(
                f"  {rank}. "
                f"{predicted_class} "
                f"-> {probability:.2f}%"
            )

    # -----------------------------------
    # METRICS
    # -----------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y,
            predictions
        )
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    try:

        log_loss_value = log_loss(
            y_encoded,
            probabilities,
            labels=list(
                range(class_count)
            )
        )

    except Exception:

        log_loss_value = np.nan

    print("\n-----------------------------------")
    print("STACKING METRICS")
    print("-----------------------------------")

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Log Loss: "
        f"{log_loss_value:.4f}"
    )

    # -----------------------------------
    # META MODEL COEFFICIENTS
    # -----------------------------------

    print("\n-----------------------------------")
    print("META MODEL FEATURE IMPORTANCE")
    print("-----------------------------------")

    coefficient_matrix = (
        meta_model.coef_
    )

    importance = np.mean(
        np.abs(
            coefficient_matrix
        ),
        axis=0
    )

    importance_df = pd.DataFrame({

        "feature":
            meta_features.columns,

        "importance":
            importance
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        importance_df.head(
            20
        ).to_string(
            index=False
        )
    )

    # -----------------------------------
    # SAVE STACKING FEATURES
    # -----------------------------------

    output = oof_data.copy()

    output[
        "stacking_prediction"
    ] = predictions

    output[
        "stacking_confidence"
    ] = probabilities.max(
        axis=1
    )

    output_file = (
        "ml/stacking_predictions.csv"
    )

    output.to_csv(
        output_file,
        index=False
    )

    print("\n-----------------------------------")
    print("OUTPUT")
    print("-----------------------------------")

    print(
        f"Saved stacking results to:"
    )

    print(
        output_file
    )

    # -----------------------------------
    # WARNING
    # -----------------------------------

    print("\n-----------------------------------")
    print("IMPORTANT")
    print("-----------------------------------")

    if len(oof_data) < 100:

        print(
            "Current OOF dataset is small."
        )

        print(
            "Stacking performance is experimental."
        )

        print(
            "The meta-model must be revalidated "
            "with substantially more historical data."
        )

    print("\n===================================")
    print("     STACKING ENSEMBLE COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_stacking_ensemble(
        target_column="col1"
    )