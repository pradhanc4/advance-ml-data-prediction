from collections import Counter
from database.connection import SessionLocal
from database.models import HistoricalRecord


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


def run_basic_analysis():

    db = SessionLocal()

    try:

        records = (
            db.query(HistoricalRecord)
            .order_by(HistoricalRecord.record_date)
            .all()
        )

        print("\n===================================")
        print("      BASIC DATA ANALYSIS")
        print("===================================")

        if not records:
            print("No historical records found.")
            return

        print(f"Total records: {len(records)}")

        print(
            f"First date: "
            f"{records[0].record_date}"
        )

        print(
            f"Last date: "
            f"{records[-1].record_date}"
        )

        print("\n-----------------------------------")
        print("COLUMN ANALYSIS")
        print("-----------------------------------")

        for column in VALUE_COLUMNS:

            values = []

            for record in records:

                value = getattr(record, column)

                if value is not None:
                    values.append(value)

            if not values:
                print(f"\n{column}: No data")
                continue

            counter = Counter(values)

            most_common = counter.most_common(5)

            average = sum(values) / len(values)

            minimum = min(values)

            maximum = max(values)

            zero_count = values.count(0)

            print(f"\n{column}")

            print(f"  Valid values: {len(values)}")

            print(f"  Average: {average:.2f}")

            print(f"  Minimum: {minimum}")

            print(f"  Maximum: {maximum}")

            print(f"  Zero count: {zero_count}")

            print(
                f"  Top 5 frequencies: "
                f"{most_common}"
            )

        print("\n-----------------------------------")
        print("DATA STATUS")
        print("-----------------------------------")

        status_counter = Counter(
            record.data_status
            for record in records
        )

        for status, count in status_counter.items():

            print(
                f"{status}: {count}"
            )

        print("\n===================================")
        print("ANALYSIS COMPLETED")
        print("===================================\n")

    finally:

        db.close()


if __name__ == "__main__":
    run_basic_analysis()