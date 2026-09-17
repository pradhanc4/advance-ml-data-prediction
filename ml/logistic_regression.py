import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss
)

from ml.training_data import build_training_dataset
from ml.time_split import time_aware_split


def run_logistic_regression(target_column="col1"):

    print("\n===================================")
    print("       LOGISTIC REGRESSION")
    print("===================================")

    print(f"Target column: {target_column}")

    # ---------------------------------------------
    # BUILD DATASET
    # ---------------------------------------------

    dataset = build_training_dataset(
        target_column
    )

    if dataset.empty:
        print("No training data available.")
        return

    # ---------------------------------------------
    # TIME-AWARE SPLIT
    # ---------------------------------------------

    train_data, test_data = time_aware_split(
        dataset,
        test_size=0.20
    )

    feature_columns = [
        column
        for column in dataset.columns
        if column not in [
            "target",
            "record_date"
        ]
    ]

    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]

    y_train = train_data["target"]
    y_test = test_data["target"]

    print("\n-----------------------------------")
    print("DATA")
    print("-----------------------------------")

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Features: {len(feature_columns)}")

    # ---------------------------------------------
    # CHECK TARGET CLASSES
    # ---------------------------------------------

    unique_classes = y_train.nunique()

    print(
        f"Training classes: {unique_classes}"
    )

    if unique_classes < 2:

        print("\nWARNING:")
        print(
            "Logistic Regression requires at least "
            "2 different target classes in training data."
        )

        print(
            "Add more historical data and try again."
        )

        return

    # ---------------------------------------------
    # CREATE MODEL
    # ---------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    # ---------------------------------------------
    # TRAIN
    # ---------------------------------------------

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # ---------------------------------------------
    # PREDICTION
    # ---------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    # ---------------------------------------------
    # METRICS
    # ---------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    # Log loss can fail when test contains classes
    # that weren't present during training.
    try:

        loss = log_loss(
            y_test,
            probabilities,
            labels=model.classes_
        )

    except ValueError:

        loss = None

    # ---------------------------------------------
    # RESULTS
    # ---------------------------------------------

    print("\n-----------------------------------")
    print("MODEL RESULTS")
    print("-----------------------------------")

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{macro_f1 * 100:.2f}%"
    )

    if loss is not None:

        print(
            f"Log Loss: "
            f"{loss:.4f}"
        )

    else:

        print(
            "Log Loss: Not available"
        )

    # ---------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------

    print("\n-----------------------------------")
    print("TEST PREDICTIONS")
    print("-----------------------------------")

    for index in range(len(y_test)):

        actual = y_test.iloc[index]

        predicted = predictions[index]

        probability = (
            probabilities[index].max()
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
            f"Predicted={predicted}, "
            f"Confidence={probability:.2f}% "
            f"-> {result}"
        )

    # ---------------------------------------------
    # TOP PROBABILITIES
    # ---------------------------------------------

    print("\n-----------------------------------")
    print("PREDICTED PROBABILITY DISTRIBUTION")
    print("-----------------------------------")

    for index in range(len(y_test)):

        probability_row = probabilities[index]

        ranked_indices = (
            probability_row
            .argsort()[::-1]
        )

        print(
            f"\nTest record {index + 1}:"
        )

        for rank, class_index in enumerate(
            ranked_indices[:5],
            start=1
        ):

            predicted_class = (
                model.classes_[class_index]
            )

            probability = (
                probability_row[class_index]
                * 100
            )

            print(
                f"  {rank}. "
                f"{predicted_class} "
                f"-> {probability:.2f}%"
            )

    print("\n===================================")
    print("LOGISTIC REGRESSION COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_logistic_regression("col1")