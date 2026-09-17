import pandas as pd

from ml.training_data import build_training_dataset


def time_aware_split(
    dataset,
    test_size=0.20
):
    """
    Split historical ML data chronologically.

    Earlier records -> training
    Later records   -> testing
    """

    if dataset.empty:
        raise ValueError(
            "Cannot split an empty dataset."
        )

    dataset = dataset.copy()

    # Make sure dates are datetime
    dataset["record_date"] = pd.to_datetime(
        dataset["record_date"]
    )

    # Sort oldest -> newest
    dataset = dataset.sort_values(
        "record_date"
    ).reset_index(drop=True)

    total_rows = len(dataset)

    if total_rows < 5:
        raise ValueError(
            "Not enough records for a reliable "
            "train/test split. Add more historical data."
        )

    # Calculate test rows
    test_rows = max(
        1,
        int(total_rows * test_size)
    )

    train_rows = total_rows - test_rows

    if train_rows < 2:
        raise ValueError(
            "Training dataset is too small."
        )

    train_data = dataset.iloc[
        :train_rows
    ].copy()

    test_data = dataset.iloc[
        train_rows:
    ].copy()

    return train_data, test_data


def run_time_split_test():

    print("\n===================================")
    print("       TIME-AWARE TRAIN / TEST")
    print("===================================")

    target_column = "col1"

    print(
        f"Target column: {target_column}"
    )

    dataset = build_training_dataset(
        target_column
    )

    if dataset.empty:
        print("No training data available.")
        return

    train_data, test_data = time_aware_split(
        dataset,
        test_size=0.20
    )

    print("\n-----------------------------------")
    print("DATASET")
    print("-----------------------------------")

    print(
        f"Total records: {len(dataset)}"
    )

    print(
        f"Training records: {len(train_data)}"
    )

    print(
        f"Testing records: {len(test_data)}"
    )

    print("\n-----------------------------------")
    print("TRAINING PERIOD")
    print("-----------------------------------")

    print(
        f"Start: {train_data['record_date'].min()}"
    )

    print(
        f"End:   {train_data['record_date'].max()}"
    )

    print("\n-----------------------------------")
    print("TESTING PERIOD")
    print("-----------------------------------")

    print(
        f"Start: {test_data['record_date'].min()}"
    )

    print(
        f"End:   {test_data['record_date'].max()}"
    )

    print("\n-----------------------------------")
    print("CHRONOLOGICAL CHECK")
    print("-----------------------------------")

    train_end = train_data["record_date"].max()
    test_start = test_data["record_date"].min()

    if train_end < test_start:

        print(
            "Chronological order: PASSED"
        )

        print(
            "Future data leakage: PREVENTED"
        )

    else:

        print(
            "Chronological order: FAILED"
        )

    print("\n-----------------------------------")
    print("TRAIN TARGET DISTRIBUTION")
    print("-----------------------------------")

    print(
        train_data["target"]
        .value_counts()
        .sort_index()
    )

    print("\n-----------------------------------")
    print("TEST TARGET DISTRIBUTION")
    print("-----------------------------------")

    print(
        test_data["target"]
        .value_counts()
        .sort_index()
    )

    print("\n===================================")
    print("TIME-AWARE SPLIT COMPLETED")
    print("===================================\n")


if __name__ == "__main__":
    run_time_split_test()