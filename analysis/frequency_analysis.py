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


RECENT_RECORDS = 7


def analyze_column(values):

    if not values:
        return {
            "total_values": 0,
            "frequencies": {},
            "top_5": [],
            "bottom_5": [],
            "zero_count": 0,
            "recent_frequencies": {},
            "trend": "NO_DATA"
        }

    counter = Counter(values)

    top_5 = counter.most_common(5)

    bottom_5 = sorted(
        counter.items(),
        key=lambda item: (item[1], item[0])
    )[:5]

    zero_count = values.count(0)

    recent_values = values[-RECENT_RECORDS:]

    recent_counter = Counter(recent_values)

    historical_average = len(values) / len(counter)

    recent_average = (
        len(recent_values) / len(recent_counter)
        if recent_counter
        else 0
    )

    if recent_average > historical_average:
        trend = "INCREASING"
    elif recent_average < historical_average:
        trend = "DECREASING"
    else:
        trend = "STABLE"

    return {
        "total_values": len(values),
        "frequencies": dict(counter),
        "top_5": top_5,
        "bottom_5": bottom_5,
        "zero_count": zero_count,
        "recent_frequencies": dict(recent_counter),
        "trend": trend
    }


def run_frequency_analysis():

    db = SessionLocal()

    try:

        records = (
            db.query(HistoricalRecord)
            .filter(
                HistoricalRecord.data_status == "AVAILABLE"
            )
            .order_by(
                HistoricalRecord.record_date
            )
            .all()
        )

        print("\n===================================")
        print("     FREQUENCY & TREND ANALYSIS")
        print("===================================")

        print(
            f"Available records: {len(records)}"
        )

        if not records:
            print("No AVAILABLE records found.")
            return

        for column in VALUE_COLUMNS:

            values = []

            for record in records:

                value = getattr(record, column)

                if value is not None:
                    values.append(value)

            result = analyze_column(values)

            print("\n-----------------------------------")
            print(f"{column.upper()}")
            print("-----------------------------------")

            print(
                f"Total valid values: "
                f"{result['total_values']}"
            )

            print(
                f"Zero count: "
                f"{result['zero_count']}"
            )

            print(
                f"Trend: "
                f"{result['trend']}"
            )

            print("\nTop 5:")
            for value, count in result["top_5"]:
                print(
                    f"  Value {value}: "
                    f"{count} times"
                )

            print("\nBottom 5:")
            for value, count in result["bottom_5"]:
                print(
                    f"  Value {value}: "
                    f"{count} times"
                )

            print("\nRecent frequency:")
            print(
                f"  {result['recent_frequencies']}"
            )

        print("\n===================================")
        print("FREQUENCY ANALYSIS COMPLETED")
        print("===================================\n")

    finally:

        db.close()


if __name__ == "__main__":
    run_frequency_analysis()