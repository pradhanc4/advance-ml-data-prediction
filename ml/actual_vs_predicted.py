from database.connection import SessionLocal
from database.models import (
    PredictionRecord,
    HistoricalRecord
)


def get_actual_value(db, prediction):
    """
    Find the actual historical value for the prediction date
    and target column.
    """

    historical_record = (
        db.query(HistoricalRecord)
        .filter(
            HistoricalRecord.record_date
            == prediction.prediction_date
        )
        .first()
    )

    # No historical record exists for this date
    if historical_record is None:
        return None, None, "PENDING"

    # Get target column dynamically
    target_column = prediction.target_column

    if not hasattr(
        historical_record,
        target_column
    ):
        raise ValueError(
            f"Invalid target column: {target_column}"
        )

    actual_value = getattr(
        historical_record,
        target_column
    )

    # Record exists, but actual value is NULL
    if actual_value is None:

        return (
            None,
            historical_record.record_date,
            "NO_DATA"
        )

    return (
        int(actual_value),
        historical_record.record_date,
        "RESOLVED"
    )


def evaluate_prediction(prediction_id=None):
    """
    Compare stored prediction with actual historical value.
    """

    db = SessionLocal()

    try:

        # --------------------------------------------------
        # GET PREDICTION
        # --------------------------------------------------

        if prediction_id is not None:

            prediction = (
                db.query(PredictionRecord)
                .filter(
                    PredictionRecord.id
                    == prediction_id
                )
                .first()
            )

        else:

            prediction = (
                db.query(PredictionRecord)
                .order_by(
                    PredictionRecord.prediction_date.desc()
                )
                .first()
            )

        if prediction is None:

            print("\nNo prediction records found.")

            return None

        # --------------------------------------------------
        # DISPLAY PREDICTION
        # --------------------------------------------------

        print("\n===================================")
        print("     9.11 ACTUAL VS PREDICTED")
        print("===================================")

        print("\n-----------------------------------")
        print("PREDICTION INFORMATION")
        print("-----------------------------------")

        print(
            f"Prediction ID: {prediction.id}"
        )

        print(
            f"Prediction date: "
            f"{prediction.prediction_date}"
        )

        print(
            f"Target column: "
            f"{prediction.target_column}"
        )

        print(
            f"Predicted value: "
            f"{prediction.predicted_value}"
        )

        print(
            f"Prediction probability: "
            f"{prediction.ensemble_probability:.2f}%"
        )

        print(
            f"Confidence score: "
            f"{prediction.confidence_score:.2f}%"
        )

        print(
            f"Current status: "
            f"{prediction.prediction_status}"
        )

        # --------------------------------------------------
        # GET ACTUAL VALUE
        # --------------------------------------------------

        (
            actual_value,
            actual_record_date,
            evaluation_status
        ) = get_actual_value(
            db,
            prediction
        )

        # --------------------------------------------------
        # NO ACTUAL RECORD YET
        # --------------------------------------------------

        if evaluation_status == "PENDING":

            print("\n-----------------------------------")
            print("ACTUAL RESULT")
            print("-----------------------------------")

            print(
                "Actual value: NOT AVAILABLE"
            )

            print(
                "Prediction status: PENDING"
            )

            print(
                "Reason: Historical record for "
                "prediction date does not exist yet."
            )

            print("\n===================================")
            print(
                "9.11 ACTUAL VS PREDICTED "
                "CHECK COMPLETED"
            )
            print("===================================\n")

            return prediction

        # --------------------------------------------------
        # ACTUAL RECORD EXISTS BUT VALUE IS NULL
        # --------------------------------------------------

        if evaluation_status == "NO_DATA":

            prediction.actual_value = None

            prediction.actual_record_date = (
                actual_record_date
            )

            prediction.prediction_status = (
                "NO_DATA"
            )

            db.commit()

            print("\n-----------------------------------")
            print("ACTUAL RESULT")
            print("-----------------------------------")

            print(
                f"Actual record date: "
                f"{actual_record_date}"
            )

            print(
                "Actual value: NULL"
            )

            print(
                "Prediction status: NO_DATA"
            )

            print(
                "Reason: Record exists, but the "
                "target value is missing."
            )

            print("\n===================================")
            print(
                "9.11 ACTUAL VS PREDICTED "
                "CHECK COMPLETED"
            )
            print("===================================\n")

            return prediction

        # --------------------------------------------------
        # ACTUAL VALUE FOUND
        # --------------------------------------------------

        predicted_value = (
            prediction.predicted_value
        )

        prediction.actual_value = (
            actual_value
        )

        prediction.actual_record_date = (
            actual_record_date
        )

        prediction.prediction_status = (
            "RESOLVED"
        )

        # --------------------------------------------------
        # COMPARE
        # --------------------------------------------------

        is_correct = (
            predicted_value
            == actual_value
        )

        print("\n-----------------------------------")
        print("ACTUAL VS PREDICTED")
        print("-----------------------------------")

        print(
            f"Prediction date: "
            f"{prediction.prediction_date}"
        )

        print(
            f"Predicted value: "
            f"{predicted_value}"
        )

        print(
            f"Actual value: "
            f"{actual_value}"
        )

        print(
            f"Actual record date: "
            f"{actual_record_date}"
        )

        if is_correct:

            print(
                "Result: CORRECT"
            )

        else:

            print(
                "Result: INCORRECT"
            )

        print(
            f"Prediction status: "
            f"{prediction.prediction_status}"
        )

        # --------------------------------------------------
        # SAVE RESULT
        # --------------------------------------------------

        db.commit()

        print("\n-----------------------------------")
        print("DATABASE UPDATE")
        print("-----------------------------------")

        print(
            "Actual value stored: YES"
        )

        print(
            "Actual record date stored: YES"
        )

        print(
            "Prediction status updated: YES"
        )

        print(
            f"Final status: "
            f"{prediction.prediction_status}"
        )

        print("\n===================================")
        print(
            "9.11 ACTUAL VS PREDICTED "
            "COMPLETED"
        )
        print("===================================\n")

        return prediction

    except Exception as e:

        db.rollback()

        print("\n-----------------------------------")
        print("ERROR")
        print("-----------------------------------")

        print(
            f"Actual vs predicted evaluation "
            f"failed: {e}"
        )

        raise

    finally:

        db.close()


def run_actual_vs_predicted_test():

    """
    Test the latest prediction record.
    """

    return evaluate_prediction()


if __name__ == "__main__":

    run_actual_vs_predicted_test()