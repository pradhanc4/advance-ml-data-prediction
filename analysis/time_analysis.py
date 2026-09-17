from collections import Counter
from statistics import mean

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


def calculate_period_analysis(records, period_name):

    groups = {}

    for record in records:

        if period_name == "DAILY":

            key = record.record_date.strftime("%Y-%m-%d")

        elif period_name == "WEEKLY":

            year, week, _ = record.record_date.isocalendar()

            key = f"{year}-W{week:02d}"

        elif period_name == "MONTHLY":

            key = record.record_date.strftime("%Y-%m")

        elif period_name == "YEARLY":

            key = record.record_date.strftime("%Y")

        else:
            continue

        if key not in groups:
            groups[key] = []

        groups[key].append(record)

    return groups


def analyze_group(records):

    result = {}

    result["record_count"] = len(records)

    available_count = sum(
        1
        for record in records
        if record.data_status == "AVAILABLE"
    )

    missing_count = sum(
        1
        for record in records
        if record.data_status == "MISSING"
    )

    result["available_count"] = available_count
    result["missing_count"] = missing_count

    column_results = {}

    for column in VALUE_COLUMNS:

        values = []

        for record in records:

            value = getattr(record, column)

            if value is not None:
                values.append(value)

        if values:

            counter = Counter(values)

            column_results[column] = {
                "count": len(values),
                "average": round(mean(values), 2),
                "zero_count": values.count(0),
                "most_common": counter.most_common(3)
            }

        else:

            column_results[column] = {
                "count": 0,
                "average": None,
                "zero_count": 0,
                "most_common": []
            }

    result["columns"] = column_results

    return result


def print_period_analysis(records, period_name):

    print("\n===================================")
    print(f"{period_name} ANALYSIS")
    print("===================================")

    groups = calculate_period_analysis(
        records,
        period_name
    )

    if not groups:
        print("No data available.")
        return

    for period, period_records in groups.items():

        result = analyze_group(period_records)

        print(f"\n{period}")
        print("-----------------------------------")

        print(
            f"Records: "
            f"{result['record_count']}"
        )

        print(
            f"Available: "
            f"{result['available_count']}"
        )

        print(
            f"Missing: "
            f"{result['missing_count']}"
        )

        for column in VALUE_COLUMNS:

            column_data = result["columns"][column]

            print(
                f"{column}: "
                f"count={column_data['count']}, "
                f"avg={column_data['average']}, "
                f"zeros={column_data['zero_count']}, "
                f"top={column_data['most_common']}"
            )


def run_time_analysis():

    db = SessionLocal()

    try:

        records = (
            db.query(HistoricalRecord)
            .order_by(
                HistoricalRecord.record_date
            )
            .all()
        )

        print("\n===================================")
        print("       TIME ANALYSIS ENGINE")
        print("===================================")

        print(
            f"Total database records: "
            f"{len(records)}"
        )

        if not records:
            print("No historical records found.")
            return

        print_period_analysis(
            records,
            "DAILY"
        )

        print_period_analysis(
            records,
            "WEEKLY"
        )

        print_period_analysis(
            records,
            "MONTHLY"
        )

        print_period_analysis(
            records,
            "YEARLY"
        )

        print("\n===================================")
        print("TIME ANALYSIS COMPLETED")
        print("===================================\n")

    finally:

        db.close()


if __name__ == "__main__":
    run_time_analysis()