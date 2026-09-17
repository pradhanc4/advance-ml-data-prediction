from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss
)

from ml.training_data import build_training_dataset
from ml.time_split import time_aware_split


def run_extra_trees(target_column="col1"):

    print("\n===================================")
    print("           EXTRA TREES")
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

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    # ---------------------------------------------
    # CHECK CLASSES
    # ---------------------------------------------

    unique_classes = y_train.nunique()

    print(
        f"Training classes: {unique_classes}"
    )

    if unique_classes < 2:

        print("\nWARNING:")

        print(
            "Extra Trees requires at least "
            "2 different target classes."
        )

        return

    # ---------------------------------------------
    # CREATE MODEL
    # ---------------------------------------------

    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
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

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

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
    # FEATURE IMPORTANCE
    # ---------------------------------------------

    print("\n-----------------------------------")
    print("TOP FEATURE IMPORTANCE")
    print("-----------------------------------")

    importance_pairs = list(
        zip(
            feature_columns,
            model.feature_importances_
        )
    )

    importance_pairs.sort(
        key=lambda item: item[1],
        reverse=True
    )

    for feature, importance in (
        importance_pairs[:15]
    ):

        print(
            f"{feature}: "
            f"{importance:.6f}"
        )

    # ---------------------------------------------
    # TEST PREDICTIONS
    # ---------------------------------------------

    print("\n-----------------------------------")
    print("TEST PREDICTIONS")
    print("-----------------------------------")

    for index in range(len(y_test)):

        actual = y_test.iloc[index]

        predicted = predictions[index]

        confidence = (
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
            f"Confidence={confidence:.2f}% "
            f"-> {result}"
        )

    # ---------------------------------------------
    # TOP PROBABILITIES
    # ---------------------------------------------

    print("\n-----------------------------------")
    print("TOP PREDICTED PROBABILITIES")
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
    print("EXTRA TREES COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_extra_trees("col1")