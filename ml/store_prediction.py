from datetime import timedelta

from ml.prediction_result import (
    run_prediction_result_formatter
)

from ml.prediction_storage import (
    save_prediction_to_sql
)

from ml.prediction_engine import (
    prepare_prediction_input
)


def run_sql_storage_test():

    print("\n===================================")
    print("       9.9 SQL PREDICTION STORAGE")
    print("===================================")

    # Generate formatted prediction
    result = run_prediction_result_formatter()

    if result is None:
        print("Prediction result creation failed.")
        return None

    # Load historical data
    historical_data = prepare_prediction_input()

    if historical_data is None:
        print("Historical data unavailable.")
        return None

    # Get latest historical date
    latest_date = historical_data["record_date"].max()

    # Next date becomes prediction target
    prediction_date = latest_date + timedelta(days=1)

    print("\n-----------------------------------")
    print("PREDICTION DATE")
    print("-----------------------------------")

    print(
        f"Latest historical date: {latest_date}"
    )

    print(
        f"Prediction date: {prediction_date}"
    )

    # Save to SQL
    record = save_prediction_to_sql(
        result,
        prediction_date
    )

    if record is None:
        print("SQL storage failed.")
        return None

    # Validation
    print("\n-----------------------------------")
    print("SQL STORAGE VALIDATION")
    print("-----------------------------------")

    print("Prediction ID created: YES")

    print(
        f"Prediction ID: {record.id}"
    )

    print(
        f"Prediction status: "
        f"{record.prediction_status}"
    )

    if record.actual_value is None:

        print(
            "Actual value: NULL "
            "(waiting for actual result)"
        )

    else:

        print(
            f"Actual value: "
            f"{record.actual_value}"
        )

    print("\n===================================")
    print(
        "9.9 STORE PREDICTIONS IN SQL "
        "COMPLETED"
    )
    print("===================================\n")

    return record


if __name__ == "__main__":

    run_sql_storage_test()