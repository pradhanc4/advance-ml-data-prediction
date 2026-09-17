import pandas as pd

from ml.training_data import (
    build_training_dataset,
    VALUE_COLUMNS
)


def analyze_target(target_column):

    if target_column not in VALUE_COLUMNS:
        raise ValueError(
            f"Invalid target column: {target_column}"
        )

    print("\n===================================")
    print("        TARGET ANALYSIS")
    print("===================================")

    print(f"Target column: {target_column}")

    dataset = build_training_dataset(
        target_column
    )

    if dataset.empty:
        print("No training data available.")
        return

    target = dataset["target"]

    print("\n-----------------------------------")
    print("TARGET SUMMARY")
    print("-----------------------------------")

    print(
        f"Total observations: {len(target)}"
    )

    print(
        f"Unique target values: "
        f"{target.nunique()}"
    )

    print(
        f"Minimum value: {target.min()}"
    )

    print(
        f"Maximum value: {target.max()}"
    )

    print("\n-----------------------------------")
    print("TARGET FREQUENCY")
    print("-----------------------------------")

    frequency = (
        target
        .value_counts()
        .sort_index()
    )

    for value, count in frequency.items():

        percentage = (
            count / len(target)
        ) * 100

        print(
            f"{value}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    print("\n-----------------------------------")
    print("CLASS DISTRIBUTION")
    print("-----------------------------------")

    largest_class = frequency.max()

    smallest_class = frequency.min()

    imbalance_ratio = (
        largest_class / smallest_class
        if smallest_class > 0
        else 0
    )

    print(
        f"Largest class: {largest_class}"
    )

    print(
        f"Smallest class: {smallest_class}"
    )

    print(
        f"Imbalance ratio: "
        f"{imbalance_ratio:.2f}"
    )

    print("\n-----------------------------------")
    print("ML TARGET TYPE")
    print("-----------------------------------")

    if target.nunique() <= 50:

        print(
            "Classification candidate: YES"
        )

        print(
            "Reason: target has a limited "
            "number of distinct values."
        )

    else:

        print(
            "Classification candidate: "
            "REQUIRES REVIEW"
        )

        print(
            "Target contains many distinct "
            "values."
        )

    print("\n-----------------------------------")
    print("IMPORTANT")
    print("-----------------------------------")

    print(
        "Actual zero values are treated "
        "as valid target values."
    )

    print(
        "NULL target values are excluded "
        "from supervised training."
    )

    print("\n===================================")
    print("TARGET ANALYSIS COMPLETED")
    print("===================================\n")


def run_target_analysis():

    # Start with col1.
    analyze_target("col1")


if __name__ == "__main__":
    run_target_analysis()