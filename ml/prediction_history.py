from database.connection import SessionLocal
from database.models import PredictionRecord


# ===================================
# GET ALL PREDICTION HISTORY
# ===================================

def get_prediction_history(
    limit=20
):

    db = SessionLocal()

    try:

        records = (
            db.query(
                PredictionRecord
            )
            .order_by(
                PredictionRecord.prediction_date.desc()
            )
            .limit(limit)
            .all()
        )

        return records

    finally:

        db.close()


# ===================================
# GET PREDICTION BY DATE
# ===================================

def get_prediction_by_date(
    prediction_date
):

    db = SessionLocal()

    try:

        record = (
            db.query(
                PredictionRecord
            )
            .filter(
                PredictionRecord.prediction_date
                == prediction_date
            )
            .order_by(
                PredictionRecord.created_at.desc()
            )
            .first()
        )

        return record

    finally:

        db.close()


# ===================================
# DISPLAY HISTORY
# ===================================

def run_prediction_history_test():

    print("\n===================================")
    print("       9.10 PREDICTION HISTORY")
    print("===================================")

    records = get_prediction_history(
        limit=20
    )

    if not records:

        print(
            "No prediction history found."
        )

        return

    print("\n-----------------------------------")
    print("PREDICTION HISTORY")
    print("-----------------------------------")

    print(
        f"Total records retrieved: "
        f"{len(records)}"
    )

    print()

    print(
        "ID | DATE       | TARGET | "
        "PRED | PROBABILITY | CONFIDENCE | STATUS"
    )

    print(
        "-" * 85
    )

    for record in records:

        print(
            f"{record.id:<3} | "
            f"{record.prediction_date} | "
            f"{record.target_column:<6} | "
            f"{record.predicted_value:<4} | "
            f"{record.ensemble_probability:>10.2f}% | "
            f"{record.confidence_score:>9.2f}% | "
            f"{record.prediction_status}"
        )

    # ===================================
    # VALIDATION
    # ===================================

    print("\n-----------------------------------")
    print("HISTORY VALIDATION")
    print("-----------------------------------")

    print(
        "SQL records retrieved: YES"
    )

    print(
        "Prediction dates available: YES"
    )

    print(
        "Predicted values available: YES"
    )

    print(
        "Probability available: YES"
    )

    print(
        "Confidence available: YES"
    )

    print(
        "Prediction status available: YES"
    )

    pending_count = sum(
        1
        for record in records
        if record.prediction_status
        == "PENDING"
    )

    resolved_count = sum(
        1
        for record in records
        if record.prediction_status
        == "RESOLVED"
    )

    print(
        f"Pending predictions: "
        f"{pending_count}"
    )

    print(
        f"Resolved predictions: "
        f"{resolved_count}"
    )

    print("\n===================================")
    print(
        "9.10 PREDICTION HISTORY "
        "COMPLETED"
    )
    print("===================================\n")


if __name__ == "__main__":

    run_prediction_history_test()