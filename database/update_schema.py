import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "ml_prediction.db"


def update_database():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    try:

        # Check whether data_status already exists
        cursor.execute("PRAGMA table_info(historical_records)")

        columns = [
            row[1]
            for row in cursor.fetchall()
        ]

        if "data_status" in columns:

            print("data_status column already exists.")

        else:

            cursor.execute("""
                ALTER TABLE historical_records
                ADD COLUMN data_status
                VARCHAR(20)
                NOT NULL
                DEFAULT 'AVAILABLE'
            """)

            connection.commit()

            print("data_status column added successfully.")

        # Check record count
        cursor.execute(
            "SELECT COUNT(*) FROM historical_records"
        )

        count = cursor.fetchone()[0]

        print(f"Historical records preserved: {count}")

    finally:

        connection.close()


if __name__ == "__main__":
    update_database()