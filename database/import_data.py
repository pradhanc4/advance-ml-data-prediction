import pandas as pd
from datetime import datetime

from .connection import SessionLocal
from .models import Market, HistoricalRecord

def clean_value(value):
    if value is None:
        return None

    if pd.isna(value):
        return None

    if str(value).strip() == "":
        return None

    return int(value)

CSV_FILE = "data/sample_market_1.csv"


def import_market_data():

    df = pd.read_csv(CSV_FILE)

    print("Data loaded successfully.")
    print(f"Rows found: {len(df)}")

    required_columns = [
        "date",
        "col1",
        "col2",
        "col3",
        "col4",
        "col5",
        "col6",
        "col7",
        "col8"
    ]

    # Check required columns
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Convert date
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Check invalid dates
    invalid_dates = df["date"].isna().sum()

    if invalid_dates > 0:
        raise ValueError(
            f"Found {invalid_dates} invalid dates."
        )

    db = SessionLocal()

    try:

        # Find or create market
        market = (
            db.query(Market)
            .filter(Market.name == "market_1")
            .first()
        )

        if not market:

            market = Market(
                name="market_1",
                description="Historical Market 1 data"
            )

            db.add(market)
            db.commit()
            db.refresh(market)

        imported = 0
        skipped = 0

        for _, row in df.iterrows():

            record_date = row["date"].date()

            # Prevent duplicate records
            existing = (
                db.query(HistoricalRecord)
                .filter(
                    HistoricalRecord.market_id == market.id,
                    HistoricalRecord.record_date == record_date
                )
                .first()
            )

            if existing:
                skipped += 1
                continue

            record = HistoricalRecord(
              values = [
    clean_value(row["col1"]),
    clean_value(row["col2"]),
    clean_value(row["col3"]),
    clean_value(row["col4"]),
    clean_value(row["col5"]),
    clean_value(row["col6"]),
    clean_value(row["col7"]),
    clean_value(row["col8"])
]

if all(value is None for value in values):
    data_status = "MISSING"
else:
    data_status = "AVAILABLE"

record = HistoricalRecord(
    market_id=market_id,
    record_date=record_date,
    day_name=record_date.strftime("%A"),

    data_status=data_status,

    col1=values[0],
    col2=values[1],
    col3=values[2],
    col4=values[3],
    col5=values[4],
    col6=values[5],
    col7=values[6],
    col8=values[7]
)
)
            db.add(record)
            imported += 1

        db.commit()

        print("-----------------------------")
        print("Import completed successfully.")
        print(f"Imported: {imported}")
        print(f"Skipped: {skipped}")
        print("-----------------------------")

    except Exception:

        db.rollback()
        raise

    finally:

        db.close()


if __name__ == "__main__":
    import_market_data()