from datetime import date
from sqlalchemy import func

from .connection import SessionLocal
from .models import HistoricalRecord


VALUE_COLUMNS = [
    "col1",
    "col2",
    "col3",
    "col4",
    "col5",
    "col6",
    "col7",
    "col8"
]


def validate_data():

    db = SessionLocal()

    try:

        records = (
            db.query(HistoricalRecord)
            .order_by(HistoricalRecord.record_date)
            .all()
        )

        print("\n==============================")
        print("DATA VALIDATION REPORT")
        print("==============================")

        print(f"Total records: {len(records)}")

        if not records:
            print("No historical records found.")
            return

        # --------------------------------
        # 1. Check NULL values
        # --------------------------------

        null_count = 0

        for record in records:

            for column in VALUE_COLUMNS:

                value = getattr(record, column)

                if value is None:

                    null_count += 1

                    print(
                        f"NULL value: "
                        f"{record.record_date} - {column}"
                    )

        print(f"\nTotal NULL values: {null_count}")

        # --------------------------------
        # 2. Check zero values
        # --------------------------------

        zero_count = 0

        for record in records:

            for column in VALUE_COLUMNS:

                value = getattr(record, column)

                if value == 0:

                    zero_count += 1

        print(f"Total actual zero values: {zero_count}")

        # --------------------------------
        # 3. Check invalid negative values
        # --------------------------------

        negative_count = 0

        for record in records:

            for column in VALUE_COLUMNS:

                value = getattr(record, column)

                if value is not None and value < 0:

                    negative_count += 1

                    print(
                        f"Negative value: "
                        f"{record.record_date} - "
                        f"{column} = {value}"
                    )

        print(
            f"Total negative values: "
            f"{negative_count}"
        )

        # --------------------------------
        # 4. Check duplicate dates
        # --------------------------------

        duplicate_dates = (
            db.query(
                HistoricalRecord.record_date,
                func.count(HistoricalRecord.id)
            )
            .group_by(HistoricalRecord.record_date)
            .having(
                func.count(HistoricalRecord.id) > 1
            )
            .all()
        )

        print(
            f"Duplicate dates found: "
            f"{len(duplicate_dates)}"
        )

        for record_date, count in duplicate_dates:

            print(
                f"Duplicate date: "
                f"{record_date} "
                f"({count} records)"
            )

        # --------------------------------
        # 5. Check date continuity
        # --------------------------------

        dates = [
            record.record_date
            for record in records
        ]

        dates = sorted(set(dates))

        missing_days = []

        for i in range(len(dates) - 1):

            current_date = dates[i]
            next_date = dates[i + 1]

            difference = (
                next_date - current_date
            ).days

            if difference > 1:

                missing_days_count = difference - 1

                for day_offset in range(
                    1,
                    difference
                ):

                    missing_date = (
                        current_date
                        + __import__("datetime")
                        .timedelta(days=day_offset)
                    )

                    missing_days.append(
                        missing_date
                    )

        print(
            f"Missing calendar dates: "
            f"{len(missing_days)}"
        )

        if missing_days:

            print("\nMissing dates:")

            for missing_date in missing_days:

                print(
                    f"  {missing_date}"
                )

        # --------------------------------
        # 6. Final status
        # --------------------------------

        print("\n==============================")

        if negative_count > 0:

            print(
                "STATUS: WARNING - "
                "Invalid values detected."
            )

        else:

            print(
                "STATUS: VALIDATION COMPLETED"
            )

        print("==============================\n")

    finally:

        db.close()


if __name__ == "__main__":
    validate_data()