from collections import Counter

from ml.training_data import build_training_dataset
from ml.time_split import time_aware_split


def run_baseline(target_column="col1"):

    print("\n===================================")
    print("          BASELINE MODEL")
    print("===================================")

    print(
        f"Target column: {target_column}"
    )

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

    y_train = train_data["target"]
    y_test = test_data["target"]

    # ---------------------------------------------
    # FIND MOST FREQUENT TRAINING VALUE
    # ---------------------------------------------

    frequency = Counter(y_train)

    baseline_value = frequency.most_common(1)[0][0]

    # ---------------------------------------------
    # GENERATE BASELINE PREDICTIONS
    # ---------------------------------------------

    predictions = [
        baseline_value
        for _ in range(len(y_test))
    ]

    # ---------------------------------------------
    # ACCURACY
    # ---------------------------------------------

    correct = sum(
        actual == predicted
        for actual, predicted
        in zip(y_test, predictions)
    )

    accuracy = (
        correct / len(y_test)
        if len(y_test) > 0
        else 0
    )

    # ---------------------------------------------
    # RESULTS
    # ---------------------------------------------

    print("\n-----------------------------------")
    print("BASELINE INFORMATION")
    print("-----------------------------------")

    print(
        f"Most frequent training value: "
        f"{baseline_value}"
    )

    print(
        f"Training records: {len(y_train)}"
    )

    print(
        f"Testing records: {len(y_test)}"
    )

    print("\n-----------------------------------")
    print("BASELINE RESULT")
    print("-----------------------------------")

    print(
        f"Correct predictions: {correct}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print("\n-----------------------------------")
    print("TEST PREDICTIONS")
    print("-----------------------------------")

    for index, (
        actual,
        predicted
    ) in enumerate(
        zip(y_test, predictions),
        start=1
    ):

        result = (
            "CORRECT"
            if actual == predicted
            else "WRONG"
        )

        print(
            f"Test {index}: "
            f"Actual={actual}, "
            f"Predicted={predicted} "
            f"-> {result}"
        )

    print("\n===================================")
    print("BASELINE MODEL COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_baseline("col1")