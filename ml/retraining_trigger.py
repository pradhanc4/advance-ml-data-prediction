from datetime import datetime
from pathlib import Path

from database.connection import SessionLocal
from database.models import HistoricalRecord, ModelRun


# ============================================================
# 9.13 AUTOMATIC RETRAINING TRIGGER
# ============================================================

MINIMUM_DATASET_SIZE = 100
MIN_NEW_RECORDS = 20
PERFORMANCE_THRESHOLD = 0.50


def get_dataset_size(db):
    """
    Return the total number of historical records.
    """
    return db.query(HistoricalRecord).count()


def get_latest_training_run(db):
    """
    Return the most recent model training run.
    """
    return (
        db.query(ModelRun)
        .order_by(ModelRun.training_date.desc())
        .first()
    )


def get_records_since_training(db, latest_training):
    """
    Count records created after the latest model training.
    """

    if latest_training is None:
        return get_dataset_size(db)

    return (
        db.query(HistoricalRecord)
        .filter(
            HistoricalRecord.created_at > latest_training.training_date
        )
        .count()
    )


def get_latest_model_accuracy(db):
    """
    Return the latest recorded model accuracy.

    If no training record exists, return None.
    """

    latest_training = get_latest_training_run(db)

    if latest_training is None:
        return None

    return latest_training.accuracy


def check_retraining_trigger(
    minimum_dataset_size=MINIMUM_DATASET_SIZE,
    minimum_new_records=MIN_NEW_RECORDS,
    performance_threshold=PERFORMANCE_THRESHOLD,
):
    """
    Determine whether model retraining should be triggered.
    """

    db = SessionLocal()

    try:
        dataset_size = get_dataset_size(db)

        latest_training = get_latest_training_run(db)

        new_records = get_records_since_training(
            db,
            latest_training
        )

        latest_accuracy = get_latest_model_accuracy(db)

        reasons = []

        # ----------------------------------------------------
        # CONDITION 1: Minimum dataset size
        # ----------------------------------------------------

        if dataset_size < minimum_dataset_size:
            reasons.append(
                f"Dataset size {dataset_size} is below "
                f"minimum required size {minimum_dataset_size}"
            )

        # ----------------------------------------------------
        # CONDITION 2: New records
        # ----------------------------------------------------

        enough_new_records = (
            new_records >= minimum_new_records
        )

        if not enough_new_records:
            reasons.append(
                f"Only {new_records} new records available; "
                f"{minimum_new_records} required"
            )

        # ----------------------------------------------------
        # CONDITION 3: Model performance
        # ----------------------------------------------------

        performance_requires_retraining = False

        if latest_accuracy is not None:

            if latest_accuracy < performance_threshold:

                performance_requires_retraining = True

                reasons.append(
                    f"Latest model accuracy "
                    f"{latest_accuracy:.2%} is below "
                    f"threshold {performance_threshold:.2%}"
                )

        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------

        if (
            dataset_size >= minimum_dataset_size
            and enough_new_records
        ):
            retraining_required = True
            status = "RETRAIN_REQUIRED"

        elif performance_requires_retraining:
            retraining_required = True
            status = "RETRAIN_REQUIRED"

        else:
            retraining_required = False
            status = "RETRAIN_NOT_REQUIRED"

        result = {
            "status": status,
            "retraining_required": retraining_required,
            "dataset_size": dataset_size,
            "minimum_dataset_size": minimum_dataset_size,
            "new_records": new_records,
            "minimum_new_records": minimum_new_records,
            "latest_accuracy": latest_accuracy,
            "performance_threshold": performance_threshold,
            "latest_training_date": (
                latest_training.training_date
                if latest_training
                else None
            ),
            "checked_at": datetime.now(),
            "reasons": reasons,
        }

        return result

    finally:
        db.close()


def print_retraining_report(result):
    """
    Display a readable retraining trigger report.
    """

    print()
    print("=" * 55)
    print("       9.13 AUTOMATIC RETRAINING TRIGGER")
    print("=" * 55)

    print()
    print("DATASET")
    print("-" * 55)

    print(
        f"Dataset size: "
        f"{result['dataset_size']}"
    )

    print(
        f"Minimum required: "
        f"{result['minimum_dataset_size']}"
    )

    print()
    print("NEW RECORDS")
    print("-" * 55)

    print(
        f"New records: "
        f"{result['new_records']}"
    )

    print(
        f"Minimum required: "
        f"{result['minimum_new_records']}"
    )

    print()
    print("MODEL PERFORMANCE")
    print("-" * 55)

    accuracy = result["latest_accuracy"]

    if accuracy is None:
        print("Latest accuracy: NOT AVAILABLE")
    else:
        print(
            f"Latest accuracy: "
            f"{accuracy:.2%}"
        )

    print(
        f"Performance threshold: "
        f"{result['performance_threshold']:.2%}"
    )

    print()
    print("DECISION")
    print("-" * 55)

    print(
        f"Status: "
        f"{result['status']}"
    )

    print(
        f"Retraining required: "
        f"{'YES' if result['retraining_required'] else 'NO'}"
    )

    print()
    print("REASONS")
    print("-" * 55)

    if result["reasons"]:

        for reason in result["reasons"]:
            print(f"- {reason}")

    else:
        print("- No blocking conditions")

    print()
    print("=" * 55)
    print("9.13 RETRAINING TRIGGER CHECK COMPLETED")
    print("=" * 55)
    print()


def run_retraining_trigger_test():

    result = check_retraining_trigger()

    print_retraining_report(result)

    return result


if __name__ == "__main__":
    run_retraining_trigger_test()