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


def get_values(records, column):

    values = []

    for record in records:

        value = getattr(record, column)

        if value is not None:
            values.append(value)

    return values


def analyze_transitions(values):

    transitions = []

    for i in range(len(values) - 1):

        current_value = values[i]
        next_value = values[i + 1]

        transitions.append(
            (current_value, next_value)
        )

    counter = Counter(transitions)

    return counter


def analyze_repetitions(values):

    repetitions = []

    for i in range(len(values) - 1):

        if values[i] == values[i + 1]:

            repetitions.append(values[i])

    return Counter(repetitions)


def analyze_sequences(values):

    sequences = []

    for i in range(len(values) - 2):

        sequence = (
            values[i],
            values[i + 1],
            values[i + 2]
        )

        sequences.append(sequence)

    return Counter(sequences)


def run_pattern_analysis():

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
        print("     SEQUENTIAL & PATTERN ANALYSIS")
        print("===================================")

        print(
            f"Available records: {len(records)}"
        )

        if not records:
            print("No AVAILABLE records found.")
            return

        for column in VALUE_COLUMNS:

            values = get_values(
                records,
                column
            )

            print("\n-----------------------------------")
            print(f"{column.upper()}")
            print("-----------------------------------")

            if len(values) < 2:

                print(
                    "Not enough data for "
                    "transition analysis."
                )

                continue

            # -----------------------------
            # TRANSITIONS
            # -----------------------------

            transitions = analyze_transitions(
                values
            )

            print("\nTop transitions:")

            for (
                (current_value, next_value),
                count
            ) in transitions.most_common(10):

                print(
                    f"  {current_value} -> "
                    f"{next_value}: "
                    f"{count} times"
                )

            # -----------------------------
            # REPETITIONS
            # -----------------------------

            repetitions = analyze_repetitions(
                values
            )

            print("\nRepeated consecutive values:")

            if repetitions:

                for value, count in repetitions.most_common(10):

                    print(
                        f"  {value} -> {value}: "
                        f"{count} times"
                    )

            else:

                print("  None found")

            # -----------------------------
            # 3-VALUE SEQUENCES
            # -----------------------------

            if len(values) >= 3:

                sequences = analyze_sequences(
                    values
                )

                print(
                    "\nTop 3-value sequences:"
                )

                for sequence, count in sequences.most_common(10):

                    print(
                        f"  {sequence}: "
                        f"{count} times"
                    )

        print("\n===================================")
        print("PATTERN ANALYSIS COMPLETED")
        print("===================================\n")

    finally:

        db.close()


if __name__ == "__main__":
    run_pattern_analysis()