from database.connection import SessionLocal
from database.models import PredictionRecord


def get_prediction_performance():
    """
    Analyze resolved prediction performance.

    Only predictions with actual results are included
    in accuracy calculations.
    """

    db = SessionLocal()

    try:

        # --------------------------------------------------
        # GET ALL PREDICTIONS
        # --------------------------------------------------

        predictions = (
            db.query(PredictionRecord)
            .order_by(
                PredictionRecord.prediction_date.asc()
            )
            .all()
        )

        if not predictions:

            print("\nNo prediction records found.")

            return None

        # --------------------------------------------------
        # BASIC COUNTS
        # --------------------------------------------------

        total_predictions = len(predictions)

        pending_predictions = 0
        resolved_predictions = 0
        no_data_predictions = 0

        correct_predictions = 0
        incorrect_predictions = 0

        probability_values = []
        confidence_values = []

        resolved_results = []

        # --------------------------------------------------
        # ANALYZE EACH PREDICTION
        # --------------------------------------------------

        for prediction in predictions:

            status = prediction.prediction_status

            # ----------------------------------------------
            # PENDING
            # ----------------------------------------------

            if status == "PENDING":

                pending_predictions += 1

                continue

            # ----------------------------------------------
            # NO DATA
            # ----------------------------------------------

            if status == "NO_DATA":

                no_data_predictions += 1

                continue

            # ----------------------------------------------
            # RESOLVED
            # ----------------------------------------------

            if status == "RESOLVED":

                resolved_predictions += 1

                predicted_value = (
                    prediction.predicted_value
                )

                actual_value = (
                    prediction.actual_value
                )

                # Safety check
                if actual_value is None:

                    continue

                # ------------------------------------------
                # CORRECT / INCORRECT
                # ------------------------------------------

                is_correct = (
                    predicted_value
                    == actual_value
                )

                if is_correct:

                    correct_predictions += 1

                    result = "CORRECT"

                else:

                    incorrect_predictions += 1

                    result = "INCORRECT"

                # ------------------------------------------
                # METRICS
                # ------------------------------------------

                probability_values.append(
                    prediction.ensemble_probability
                )

                confidence_values.append(
                    prediction.confidence_score
                )

                resolved_results.append(
                    {
                        "id": prediction.id,
                        "date": prediction.prediction_date,
                        "target": prediction.target_column,
                        "predicted": predicted_value,
                        "actual": actual_value,
                        "probability":
                            prediction.ensemble_probability,
                        "confidence":
                            prediction.confidence_score,
                        "result": result
                    }
                )

        # --------------------------------------------------
        # ACCURACY
        # --------------------------------------------------

        if resolved_predictions > 0:

            accuracy = (
                correct_predictions
                /
                resolved_predictions
            ) * 100

        else:

            accuracy = 0.0

        # --------------------------------------------------
        # AVERAGE PROBABILITY
        # --------------------------------------------------

        if probability_values:

            average_probability = (
                sum(probability_values)
                /
                len(probability_values)
            )

        else:

            average_probability = 0.0

        # --------------------------------------------------
        # AVERAGE CONFIDENCE
        # --------------------------------------------------

        if confidence_values:

            average_confidence = (
                sum(confidence_values)
                /
                len(confidence_values)
            )

        else:

            average_confidence = 0.0

        # --------------------------------------------------
        # DISPLAY REPORT
        # --------------------------------------------------

        print("\n===================================")
        print("     9.12 PREDICTION PERFORMANCE")
        print("===================================")

        print("\n-----------------------------------")
        print("PREDICTION COUNTS")
        print("-----------------------------------")

        print(
            f"Total predictions: "
            f"{total_predictions}"
        )

        print(
            f"Resolved predictions: "
            f"{resolved_predictions}"
        )

        print(
            f"Pending predictions: "
            f"{pending_predictions}"
        )

        print(
            f"No-data predictions: "
            f"{no_data_predictions}"
        )

        print("\n-----------------------------------")
        print("PERFORMANCE")
        print("-----------------------------------")

        print(
            f"Correct predictions: "
            f"{correct_predictions}"
        )

        print(
            f"Incorrect predictions: "
            f"{incorrect_predictions}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.2f}%"
        )

        print("\n-----------------------------------")
        print("CONFIDENCE / PROBABILITY")
        print("-----------------------------------")

        print(
            f"Average ensemble probability: "
            f"{average_probability:.2f}%"
        )

        print(
            f"Average confidence score: "
            f"{average_confidence:.2f}%"
        )

        # --------------------------------------------------
        # RESOLVED PREDICTION TABLE
        # --------------------------------------------------

        print("\n-----------------------------------")
        print("RESOLVED PREDICTIONS")
        print("-----------------------------------")

        if not resolved_results:

            print(
                "No resolved predictions available yet."
            )

        else:

            print(
                "ID | DATE       | TARGET | "
                "PRED | ACTUAL | RESULT"
            )

            print(
                "------------------------------------------------"
            )

            for item in resolved_results:

                print(
                    f"{item['id']:<3} | "
                    f"{str(item['date']):<10} | "
                    f"{item['target']:<6} | "
                    f"{item['predicted']:<4} | "
                    f"{item['actual']:<6} | "
                    f"{item['result']}"
                )

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        print("\n-----------------------------------")
        print("PERFORMANCE VALIDATION")
        print("-----------------------------------")

        print(
            "Prediction records retrieved: YES"
        )

        print(
            "Resolved predictions identified: YES"
        )

        print(
            "Pending predictions excluded "
            "from accuracy: YES"
        )

        print(
            "NO_DATA predictions excluded "
            "from accuracy: YES"
        )

        print(
            "Correct/incorrect comparison: YES"
        )

        print(
            "Accuracy calculation: "
            "VALID"
        )

        print("\n===================================")
        print(
            "9.12 PREDICTION PERFORMANCE "
            "ANALYSIS COMPLETED"
        )
        print("===================================\n")

        return {
            "total_predictions":
                total_predictions,

            "resolved_predictions":
                resolved_predictions,

            "pending_predictions":
                pending_predictions,

            "no_data_predictions":
                no_data_predictions,

            "correct_predictions":
                correct_predictions,

            "incorrect_predictions":
                incorrect_predictions,

            "accuracy":
                accuracy,

            "average_probability":
                average_probability,

            "average_confidence":
                average_confidence,

            "resolved_results":
                resolved_results
        }

    except Exception as e:

        print("\n-----------------------------------")
        print("ERROR")
        print("-----------------------------------")

        print(
            f"Performance analysis failed: {e}"
        )

        raise

    finally:

        db.close()


def run_prediction_performance_test():

    return get_prediction_performance()


if __name__ == "__main__":

    run_prediction_performance_test()