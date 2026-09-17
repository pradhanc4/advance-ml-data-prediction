import os
import numpy as np
import pandas as pd

from ml.training_data import build_training_dataset

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


OUTPUT_FILE = "ml/final_ensemble_predictions.csv"

TARGET_COLUMN = "col1"


def train_models(X, y):

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        ),

        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            random_seed=42
        )
    }

    trained_models = {}

    for name, model in models.items():

        try:
            model.fit(X, y)

            trained_models[name] = model

            print(f"{name}: TRAINED")

        except Exception as e:

            print(f"{name}: FAILED")
            print(f"Reason: {e}")

    return trained_models


def align_probability_columns(
    model,
    probabilities,
    global_classes
):

    result = np.zeros(
        (
            len(probabilities),
            len(global_classes)
        )
    )

    model_classes = model.classes_

    for index, model_class in enumerate(
        model_classes
    ):

        matching = np.where(
            global_classes == model_class
        )[0]

        if len(matching) > 0:

            result[
                :,
                matching[0]
            ] = probabilities[
                :,
                index
            ]

    return result


def create_weighted_ensemble(
    models,
    X,
    global_classes
):

    weights = {
        "Random Forest": 0.25,
        "XGBoost": 0.25,
        "LightGBM": 0.25,
        "CatBoost": 0.25
    }

    final_probability = np.zeros(
        (
            len(X),
            len(global_classes)
        )
    )

    model_predictions = {}

    model_probabilities = {}

    total_weight = 0

    for name, model in models.items():

        probabilities = model.predict_proba(X)

        probabilities = align_probability_columns(
            model,
            probabilities,
            global_classes
        )

        weight = weights.get(
            name,
            0
        )

        final_probability += (
            probabilities * weight
        )

        total_weight += weight

        predictions = model.predict(X)

        model_predictions[name] = predictions

        model_probabilities[name] = (
            probabilities
        )

    if total_weight > 0:

        final_probability /= total_weight

    # -----------------------------------------
    # NORMALIZE
    # -----------------------------------------

    row_sum = final_probability.sum(
        axis=1,
        keepdims=True
    )

    row_sum[row_sum == 0] = 1

    final_probability = (
        final_probability / row_sum
    )

    return (
        final_probability,
        model_predictions,
        model_probabilities
    )


def calculate_model_agreement(
    model_predictions
):

    if not model_predictions:
        return np.zeros(0)

    normalized_predictions = {}

    for name, predictions in model_predictions.items():

        predictions = np.asarray(predictions)

        # Convert (n, 1) -> (n,)
        if predictions.ndim > 1:
            predictions = predictions.reshape(-1)

        normalized_predictions[name] = predictions

    prediction_df = pd.DataFrame(
        normalized_predictions
    )

    agreement = []

    for _, row in prediction_df.iterrows():

        counts = row.value_counts()

        if len(counts) == 0:
            agreement.append(0.0)
            continue

        highest_count = counts.iloc[0]

        total_models = len(row)

        score = (
            highest_count /
            total_models
        )

        agreement.append(score)

    return np.array(agreement)

    if not model_predictions:

        return np.zeros(0)

    prediction_df = pd.DataFrame(
        model_predictions
    )

    agreement = []

    for _, row in prediction_df.iterrows():

        counts = row.value_counts()

        highest_count = counts.iloc[0]

        total_models = len(row)

        score = (
            highest_count /
            total_models
        )

        agreement.append(score)

    return np.array(agreement)


def create_output(
    dataset,
    y,
    final_probability,
    final_predictions,
    confidence,
    agreement,
    model_predictions,
    global_classes
):

    output = pd.DataFrame()

    # -----------------------------------------
    # DATE
    # -----------------------------------------

    if "record_date" in dataset.columns:

        output["record_date"] = (
            dataset["record_date"].values
        )

    # -----------------------------------------
    # ACTUAL TARGET
    # -----------------------------------------

    output["actual_target"] = (
        y.values
    )

    # -----------------------------------------
    # FINAL PREDICTION
    # -----------------------------------------

    output["final_prediction"] = (
        final_predictions
    )

    # -----------------------------------------
    # CONFIDENCE
    # -----------------------------------------

    output["confidence"] = (
        confidence
    )

    # -----------------------------------------
    # MODEL AGREEMENT
    # -----------------------------------------

    output["model_agreement"] = (
        agreement
    )

    # -----------------------------------------
    # FINAL PROBABILITIES
    # -----------------------------------------

    for index, class_label in enumerate(
        global_classes
    ):

        output[
            f"final_probability_{class_label}"
        ] = final_probability[
            :,
            index
        ]

    # -----------------------------------------
    # INDIVIDUAL MODEL PREDICTIONS
    # -----------------------------------------

    for name, predictions in (
        model_predictions.items()
    ):

        safe_name = (
            name.lower()
            .replace(" ", "_")
        )

        output[
            f"{safe_name}_prediction"
        ] = predictions

    return output


def main():

    print("\n===================================")
    print("      FINAL ENSEMBLE ENGINE")
    print("===================================")

    print(
        f"\nTarget column: {TARGET_COLUMN}"
    )

    # -----------------------------------------
    # LOAD TRAINING DATA
    # -----------------------------------------

    try:

        dataset = build_training_dataset(
            TARGET_COLUMN
        )

    except Exception as e:

        print(
            "\nERROR: Could not build "
            "training dataset."
        )

        print(
            f"Reason: {e}"
        )

        return

    if dataset.empty:

        print(
            "\nERROR: Training dataset is empty."
        )

        return

    # -----------------------------------------
    # DATA INFORMATION
    # -----------------------------------------

    print("\n-----------------------------------")
    print("TRAINING DATA")
    print("-----------------------------------")

    print(
        f"Rows: {len(dataset)}"
    )

    print(
        f"Total columns: {len(dataset.columns)}"
    )

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

    # -----------------------------------------
    # NUMERIC FEATURES
    # -----------------------------------------

    X = X.select_dtypes(
        include=["number"]
    )

    # -----------------------------------------
    # TARGET
    # -----------------------------------------

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    valid_rows = y.notna()

    X = X.loc[
        valid_rows
    ].reset_index(drop=True)

    y = y.loc[
        valid_rows
    ].reset_index(drop=True)

    dataset = dataset.loc[
        valid_rows
    ].reset_index(drop=True)

    # -----------------------------------------
    # SMALL DATA WARNING
    # -----------------------------------------

    if len(X) < 50:

        print("\nWARNING")
        print("-----------------------------------")
        print(
            "Historical dataset is currently small."
        )
        print(
            f"Training rows: {len(X)}"
        )
        print(
            "Predictions are experimental."
        )
        print(
            "More historical data is required"
        )
        print(
            "for reliable model evaluation."
        )
        print("-----------------------------------")

    # -----------------------------------------
    # CLASS CHECK
    # -----------------------------------------

    global_classes = np.array(
        sorted(
            y.unique()
        )
    )

    print(
        f"\nTarget classes: "
        f"{len(global_classes)}"
    )

    print(
        f"Feature count: "
        f"{X.shape[1]}"
    )

    if len(global_classes) < 2:

        print(
            "\nERROR: Target contains fewer "
            "than 2 classes."
        )

        print(
            "More historical variation is required."
        )

        return

    # -----------------------------------------
    # TRAIN MODELS
    # -----------------------------------------

    print("\n-----------------------------------")
    print("TRAINING BASE MODELS")
    print("-----------------------------------")

    models = train_models(
        X,
        y
    )

    if not models:

        print(
            "\nERROR: No models were trained."
        )

        return

    # -----------------------------------------
    # ENSEMBLE
    # -----------------------------------------

    print("\n-----------------------------------")
    print("BUILDING WEIGHTED ENSEMBLE")
    print("-----------------------------------")

    (
        final_probability,
        model_predictions,
        model_probabilities
    ) = create_weighted_ensemble(
        models,
        X,
        global_classes
    )

    # -----------------------------------------
    # FINAL PREDICTION
    # -----------------------------------------

    final_indices = np.argmax(
        final_probability,
        axis=1
    )

    final_predictions = (
        global_classes[
            final_indices
        ]
    )

    # -----------------------------------------
    # CONFIDENCE
    # -----------------------------------------

    confidence = np.max(
        final_probability,
        axis=1
    )

    # -----------------------------------------
    # MODEL AGREEMENT
    # -----------------------------------------

    agreement = calculate_model_agreement(
        model_predictions
    )

    # -----------------------------------------
    # OUTPUT
    # -----------------------------------------

    output = create_output(
        dataset,
        y,
        final_probability,
        final_predictions,
        confidence,
        agreement,
        model_predictions,
        global_classes
    )

    # -----------------------------------------
    # SAVE
    # -----------------------------------------

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    print("\n-----------------------------------")
    print("FINAL ENSEMBLE SUMMARY")
    print("-----------------------------------")

    print(
        f"Records processed: "
        f"{len(output)}"
    )

    print(
        f"Models trained: "
        f"{len(models)}"
    )

    print(
        "Aggregation: Weighted Probability"
    )

    print(
        "Random Forest weight: 25%"
    )

    print(
        "XGBoost weight: 25%"
    )

    print(
        "LightGBM weight: 25%"
    )

    print(
        "CatBoost weight: 25%"
    )

    print(
        "Confidence: ENABLED"
    )

    print(
        "Model agreement: ENABLED"
    )

    print(
        "Probability normalization: ENABLED"
    )

    print("\n-----------------------------------")
    print("SAMPLE FINAL PREDICTIONS")
    print("-----------------------------------")

    print(
        output[
            [
                "final_prediction",
                "confidence",
                "model_agreement"
            ]
        ].head(10)
    )

    print("\n-----------------------------------")
    print("OUTPUT")
    print("-----------------------------------")

    print(
        f"Saved successfully:"
    )

    print(
        os.path.abspath(
            OUTPUT_FILE
        )
    )

    print("\n===================================")
    print(" FINAL ENSEMBLE ENGINE COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    main()