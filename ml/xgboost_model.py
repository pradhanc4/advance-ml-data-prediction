import os
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss
)

from ml.training_data import build_training_dataset
from ml.time_split import time_aware_split


MODEL_PATH = "ml/xgboost_model.pkl"


def run_xgboost(target_column="col1"):

    print("\n===================================")
    print("             XGBOOST")
    print("===================================")

    print(f"Target column: {target_column}")

    # ---------------------------------------------
    # BUILD DATASET
    # ---------------------------------------------

    dataset = build_training_dataset(target_column)

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
    # TARGET CLASSES
    # ---------------------------------------------

    classes = sorted(y_train.unique())
    class_count = len(classes)

    print(f"Training classes: {class_count}")

    if class_count < 2:

        print("\nWARNING:")
        print(
            "XGBoost requires at least "
            "2 target classes."
        )

        return

    # ---------------------------------------------
    # CLASS ENCODING
    # ---------------------------------------------

    class_to_index = {
        value: index
        for index, value in enumerate(classes)
    }

    index_to_class = {
        index: value
        for value, index in class_to_index.items()
    }

    y_train_encoded = y_train.map(
        class_to_index
    )

    y_test_encoded = y_test.map(
        class_to_index
    )

    # ---------------------------------------------
    # CREATE MODEL
    # ---------------------------------------------

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=class_count,
        eval_metric="mlogloss",
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    # ---------------------------------------------
    # TRAIN
    # ---------------------------------------------

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train_encoded
    )

    print("Training completed.")

    # ---------------------------------------------
    # PREDICTION
    # ---------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )

    predicted_indices = (
        probabilities.argmax(axis=1)
    )

    predictions = [
        index_to_class[index]
        for index in predicted_indices
    ]

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

    try:

        loss = log_loss(
            y_test_encoded,
            probabilities,
            labels=list(range(class_count))
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
                index_to_class[class_index]
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

    # ---------------------------------------------
    # SAVE TRAINED MODEL
    # ---------------------------------------------

    print("\n-----------------------------------")
    print("SAVING TRAINED MODEL")
    print("-----------------------------------")

    model_bundle = {
        "model": model,
        "feature_names": feature_columns,
        "target_column": target_column,
        "feature_version": "training_v1",
        "model_name": "XGBoost",
        "class_to_index": class_to_index,
        "index_to_class": index_to_class
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    print(
        f"Model saved: {MODEL_PATH}"
    )

    print(
        f"Saved features: "
        f"{len(feature_columns)}"
    )

    # ---------------------------------------------
    # VERIFY SAVED MODEL
    # ---------------------------------------------

    print("\n-----------------------------------")
    print("MODEL SAVE VERIFICATION")
    print("-----------------------------------")

    if os.path.exists(MODEL_PATH):

        file_size = os.path.getsize(
            MODEL_PATH
        )

        print(
            "File exists: YES"
        )

        print(
            f"File size: "
            f"{file_size} bytes"
        )

        loaded_bundle = joblib.load(
            MODEL_PATH
        )

        loaded_model = (
            loaded_bundle["model"]
        )

        loaded_features = (
            loaded_bundle["feature_names"]
        )

        print(
            "Model reload: SUCCESS"
        )

        print(
            f"Reloaded features: "
            f"{len(loaded_features)}"
        )

        print(
            f"Reloaded model classes: "
            f"{len(loaded_model.classes_)}"
        )

    else:

        print(
            "File exists: NO"
        )

        print(
            "Model save verification FAILED"
        )

        return

    print("\n===================================")
    print("XGBOOST COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_xgboost("col1")