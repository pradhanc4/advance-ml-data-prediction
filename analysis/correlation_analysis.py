import pandas as pd

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


def load_data():

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

        data = []

        for record in records:

            row = {}

            for column in VALUE_COLUMNS:

                row[column] = getattr(
                    record,
                    column
                )

            data.append(row)

        return pd.DataFrame(data)

    finally:

        db.close()


def run_correlation_analysis():

    print("\n===================================")
    print("       CORRELATION ANALYSIS")
    print("===================================")

    df = load_data()

    print(
        f"Available records: {len(df)}"
    )

    if df.empty:

        print("No AVAILABLE records found.")
        return

    print("\n-----------------------------------")
    print("CORRELATION MATRIX")
    print("-----------------------------------")

    correlation_matrix = df[
        VALUE_COLUMNS
    ].corr()

    print(
        correlation_matrix.round(3)
    )

    print("\n-----------------------------------")
    print("STRONGEST RELATIONSHIPS")
    print("-----------------------------------")

    pairs = []

    for i in range(len(VALUE_COLUMNS)):

        for j in range(i + 1, len(VALUE_COLUMNS)):

            column_a = VALUE_COLUMNS[i]
            column_b = VALUE_COLUMNS[j]

            correlation = correlation_matrix.loc[
                column_a,
                column_b
            ]

            if pd.notna(correlation):

                pairs.append(
                    (
                        column_a,
                        column_b,
                        correlation
                    )
                )

    pairs.sort(
        key=lambda item: abs(item[2]),
        reverse=True
    )

    for column_a, column_b, correlation in pairs:

        print(
            f"{column_a} <-> {column_b}: "
            f"{correlation:.3f}"
        )

    print("\n-----------------------------------")
    print("INTERPRETATION")
    print("-----------------------------------")

    print(
        "Correlation closer to +1 means "
        "strong positive linear relationship."
    )

    print(
        "Correlation closer to -1 means "
        "strong negative linear relationship."
    )

    print(
        "Correlation closer to 0 means "
        "weak linear relationship."
    )

    print("\n===================================")
    print("CORRELATION ANALYSIS COMPLETED")
    print("===================================\n")


if __name__ == "__main__":
    run_correlation_analysis()