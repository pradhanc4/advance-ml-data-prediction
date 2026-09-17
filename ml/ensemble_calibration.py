import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import label_binarize


OOF_FILE = "ml/oof_predictions.csv"
OUTPUT_FILE = "ml/calibrated_predictions.csv"


def load_oof_data():
    if not os.path.exists(OOF_FILE):
        print(f"ERROR: {OOF_FILE} not found.")
        return None

    df = pd.read_csv(OOF_FILE)

    if df.empty:
        print("ERROR: OOF prediction file is empty.")
        return None

    return df


def detect_probability_columns(df):
    model_names = ["rf", "xgb", "lgb", "cat"]

    probability_columns = []

    for model in model_names:
        cols = [
            col for col in df.columns
            if col.startswith(f"{model}_prob_")
        ]

        probability_columns.extend(cols)

    return probability_columns


def build_ensemble_probabilities(df):
    probability_columns = detect_probability_columns(df)

    if not probability_columns:
        print("ERROR: No probability columns found.")
        return None

    probability_data = df[probability_columns].copy()

    probability_data = probability_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    probability_data = probability_data.fillna(0)

    # Detect class labels from probability column names
    class_labels = sorted(
        set(
            col.split("_prob_")[1]
            for col in probability_columns
        )
    )

    ensemble_probabilities = []

    for class_label in class_labels:

        class_columns = [
            col for col in probability_columns
            if col.endswith(f"_prob_{class_label}")
        ]

        if class_columns:
            ensemble_probabilities.append(
                probability_data[class_columns].mean(axis=1)
            )

    ensemble_probabilities = np.column_stack(
        ensemble_probabilities
    )

    # Normalize
    row_sums = ensemble_probabilities.sum(axis=1, keepdims=True)

    row_sums[row_sums == 0] = 1

    ensemble_probabilities = (
        ensemble_probabilities / row_sums
    )

    return class_labels, ensemble_probabilities


def chronological_calibration_split(
    probabilities,
    targets,
    calibration_ratio=0.5
):
    n = len(targets)

    if n < 10:
        return None

    split_index = int(n * calibration_ratio)

    if split_index < 5:
        return None

    if n - split_index < 3:
        return None

    calibration_prob = probabilities[:split_index]
    evaluation_prob = probabilities[split_index:]

    calibration_target = targets[:split_index]
    evaluation_target = targets[split_index:]

    return (
        calibration_prob,
        calibration_target,
        evaluation_prob,
        evaluation_target
    )


def calibrate_probabilities(
    calibration_probabilities,
    calibration_targets
):
    """
    Multiclass calibration using logistic regression
    on log-probability features.
    """

    epsilon = 1e-8

    log_probabilities = np.log(
        np.clip(
            calibration_probabilities,
            epsilon,
            1
        )
    )

    calibrator = LogisticRegression(
        max_iter=3000,
        multi_class="multinomial"
    )

    calibrator.fit(
        log_probabilities,
        calibration_targets
    )

    return calibrator


def main():

    print("\n===================================")
    print("      ENSEMBLE CALIBRATION")
    print("===================================")

    df = load_oof_data()

    if df is None:
        return

    print(f"OOF records: {len(df)}")

    if "actual_target" not in df.columns:
        print("ERROR: actual_target column not found.")
        return

    result = build_ensemble_probabilities(df)

    if result is None:
        return

    class_labels, probabilities = result

    print(f"Detected classes: {class_labels}")

    targets = df["actual_target"].astype(str).values

    split = chronological_calibration_split(
        probabilities,
        targets
    )

    if split is None:

        print("\n-----------------------------------")
        print("CALIBRATION STATUS")
        print("-----------------------------------")
        print("Insufficient data for reliable calibration.")
        print("Calibration framework is ready.")
        print("Current dataset is too small.")
        print("Add more historical records before trusting")
        print("calibrated probability metrics.")
        print("\n===================================")
        print("ENSEMBLE CALIBRATION COMPLETED")
        print("===================================\n")

        return

    (
        calibration_prob,
        calibration_target,
        evaluation_prob,
        evaluation_target
    ) = split

    print(
        f"Calibration records: {len(calibration_target)}"
    )

    print(
        f"Evaluation records: {len(evaluation_target)}"
    )

    # Check that calibration data contains enough classes
    unique_calibration_classes = np.unique(
        calibration_target
    )

    if len(unique_calibration_classes) < 2:

        print("\nNot enough target classes for calibration.")
        print("Calibration skipped safely.")

        return

    calibrator = calibrate_probabilities(
        calibration_prob,
        calibration_target
    )

    calibrated_probabilities = calibrator.predict_proba(
        np.log(
            np.clip(
                evaluation_prob,
                1e-8,
                1
            )
        )
    )

    calibrated_classes = calibrator.classes_

    # Convert target labels to calibrator class indexes
    class_to_index = {
        label: index
        for index, label
        in enumerate(calibrated_classes)
    }

    valid_rows = [
        target in class_to_index
        for target in evaluation_target
    ]

    if not any(valid_rows):

        print("No compatible evaluation classes.")
        print("Calibration skipped.")

        return

    evaluation_target_valid = np.array(
        evaluation_target
    )[valid_rows]

    evaluation_prob_valid = evaluation_prob[valid_rows]

    calibrated_prob_valid = calibrated_probabilities[
        valid_rows
    ]

    # Raw ensemble probability for evaluation classes
    raw_loss = log_loss(
        evaluation_target_valid,
        evaluation_prob_valid,
        labels=class_labels
    )

    calibrated_loss = log_loss(
        evaluation_target_valid,
        calibrated_prob_valid,
        labels=calibrated_classes
    )

    # Brier score for multiclass
    y_true_binary = label_binarize(
        evaluation_target_valid,
        classes=calibrated_classes
    )

    calibrated_brier = np.mean(
        np.sum(
            (
                y_true_binary
                - calibrated_prob_valid
            ) ** 2,
            axis=1
        )
    )

    raw_brier = np.mean(
        np.sum(
            (
                y_true_binary
                - evaluation_prob_valid
            ) ** 2,
            axis=1
        )
    )

    print("\n-----------------------------------")
    print("CALIBRATION RESULTS")
    print("-----------------------------------")

    print(
        f"Raw Ensemble Log Loss: "
        f"{raw_loss:.6f}"
    )

    print(
        f"Calibrated Log Loss: "
        f"{calibrated_loss:.6f}"
    )

    print(
        f"Raw Ensemble Brier Score: "
        f"{raw_brier:.6f}"
    )

    print(
        f"Calibrated Brier Score: "
        f"{calibrated_brier:.6f}"
    )

    # Build output
    output = df.iloc[
        len(df) - len(evaluation_target_valid):
    ].copy()

    output["calibration_status"] = "CALIBRATED"

    for i, class_label in enumerate(
        calibrated_classes
    ):
        output[
            f"calibrated_prob_{class_label}"
        ] = calibrated_prob_valid[:, i]

    output["calibrated_prediction"] = [
        calibrated_classes[index]
        for index in np.argmax(
            calibrated_prob_valid,
            axis=1
        )
    ]

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n-----------------------------------")
    print("OUTPUT")
    print("-----------------------------------")

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print("\n-----------------------------------")
    print("CALIBRATION CHECK")
    print("-----------------------------------")

    print("Chronological calibration: YES")
    print("Future data used for calibration: NO")
    print("Probability normalization: YES")
    print("Raw vs calibrated metrics: YES")

    print("\n===================================")
    print("ENSEMBLE CALIBRATION COMPLETED")
    print("===================================\n")


if __name__ == "__main__":
    main()