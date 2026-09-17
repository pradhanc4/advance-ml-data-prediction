import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ml.training_data import build_training_dataset
from ml.time_split import time_aware_split


def calculate_disagreement(predictions_a, predictions_b):
    """
    Percentage of test records where two models
    make different predictions.
    """

    if len(predictions_a) == 0:
        return 0.0

    return np.mean(
        predictions_a != predictions_b
    )


def calculate_agreement(predictions_a, predictions_b):
    """
    Percentage of test records where two models
    make the same prediction.
    """

    if len(predictions_a) == 0:
        return 0.0

    return np.mean(
        predictions_a == predictions_b
    )


def calculate_average_pairwise_disagreement(
    prediction_matrix
):
    """
    Calculate average disagreement across
    all model pairs.
    """

    model_names = list(
        prediction_matrix.keys()
    )

    disagreements = []

    for i in range(
        len(model_names)
    ):

        for j in range(
            i + 1,
            len(model_names)
        ):

            model_a = model_names[i]
            model_b = model_names[j]

            disagreement = (
                calculate_disagreement(
                    prediction_matrix[model_a],
                    prediction_matrix[model_b]
                )
            )

            disagreements.append(
                disagreement
            )

    if not disagreements:
        return 0.0

    return np.mean(
        disagreements
    )


def run_ensemble_diversity(
    target_column="col1"
):

    print("\n===================================")
    print("      ENSEMBLE DIVERSITY ANALYSIS")
    print("===================================")

    print(
        f"Target column: {target_column}"
    )

    dataset = build_training_dataset(
        target_column
    )

    if dataset.empty:

        print(
            "No training data available."
        )

        return

    train_data, test_data = (
        time_aware_split(
            dataset,
            test_size=0.20
        )
    )

    feature_columns = [
        column
        for column in dataset.columns
        if column not in [
            "target",
            "record_date"
        ]
    ]

    X_train = train_data[
        feature_columns
    ]

    X_test = test_data[
        feature_columns
    ]

    y_train = train_data[
        "target"
    ]

    y_test = test_data[
        "target"
    ]

    classes = sorted(
        y_train.unique()
    )

    class_count = len(classes)

    print("\n-----------------------------------")
    print("DATA")
    print("-----------------------------------")

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print(
        f"Classes: {class_count}"
    )

    if class_count < 2:

        print(
            "\nAt least 2 target classes are required."
        )

        return

    class_to_index = {
        value: index
        for index, value in enumerate(classes)
    }

    y_train_encoded = y_train.map(
        class_to_index
    )

    models = {

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=2,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                num_class=class_count,
                eval_metric="mlogloss",
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            ),

        "LightGBM":
            LGBMClassifier(
                objective="multiclass",
                num_class=class_count,
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                min_child_samples=5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),

        "CatBoost":
            CatBoostClassifier(
                loss_function="MultiClass",
                iterations=300,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False,
                allow_writing_files=False
            )
    }

    predictions = {}

    print("\n===================================")
    print("TRAINING MODELS")
    print("===================================")

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        try:

            model.fit(
                X_train,
                y_train_encoded
            )

            model_prediction = (
                model.predict(
                    X_test
                )
            )

            model_prediction = (
                np.asarray(
                    model_prediction
                ).reshape(-1)
            )

            predictions[name] = (
                model_prediction
            )

            print(
                f"{name} completed."
            )

        except Exception as error:

            print(
                f"{name} failed:"
            )

            print(error)

    if len(predictions) < 2:

        print(
            "\nAt least 2 models are required."
        )

        return

    # -----------------------------------
    # INDIVIDUAL MODEL PREDICTIONS
    # -----------------------------------

    print("\n===================================")
    print("MODEL PREDICTIONS")
    print("===================================")

    prediction_table = pd.DataFrame(
        predictions
    )

    prediction_table.index = (
        np.arange(
            1,
            len(prediction_table) + 1
        )
    )

    prediction_table.index.name = (
        "Test Record"
    )

    print(
        prediction_table
    )

    # -----------------------------------
    # PAIRWISE AGREEMENT
    # -----------------------------------

    model_names = list(
        predictions.keys()
    )

    print("\n-----------------------------------")
    print("PAIRWISE MODEL AGREEMENT")
    print("-----------------------------------")

    for i in range(
        len(model_names)
    ):

        for j in range(
            i + 1,
            len(model_names)
        ):

            model_a = model_names[i]
            model_b = model_names[j]

            agreement = (
                calculate_agreement(
                    predictions[model_a],
                    predictions[model_b]
                )
            )

            disagreement = (
                1 - agreement
            )

            print(
                f"{model_a} vs {model_b}: "
                f"Agreement={agreement * 100:.2f}% | "
                f"Disagreement={disagreement * 100:.2f}%"
            )

    # -----------------------------------
    # AVERAGE DIVERSITY
    # -----------------------------------

    average_disagreement = (
        calculate_average_pairwise_disagreement(
            predictions
        )
    )

    print("\n-----------------------------------")
    print("DIVERSITY SUMMARY")
    print("-----------------------------------")

    print(
        f"Average pairwise disagreement: "
        f"{average_disagreement * 100:.2f}%"
    )

    print(
        f"Average pairwise agreement: "
        f"{(1 - average_disagreement) * 100:.2f}%"
    )

    # -----------------------------------
    # CONSENSUS
    # -----------------------------------

    print("\n-----------------------------------")
    print("MODEL CONSENSUS")
    print("-----------------------------------")

    prediction_values = np.array(
        list(predictions.values())
    ).T

    for index, row in enumerate(
        prediction_values
    ):

        values, counts = np.unique(
            row,
            return_counts=True
        )

        max_count_index = (
            np.argmax(counts)
        )

        consensus_class = (
            values[max_count_index]
        )

        consensus_count = (
            counts[max_count_index]
        )

        consensus_percentage = (
            consensus_count /
            len(row)
        ) * 100

        actual = y_test.iloc[index]

        print(
            f"Test Record {index + 1}: "
            f"Actual={actual} | "
            f"Consensus={consensus_class} | "
            f"Agreement={consensus_percentage:.2f}%"
        )

    # -----------------------------------
    # DIVERSITY INTERPRETATION
    # -----------------------------------

    print("\n-----------------------------------")
    print("DIVERSITY CHECK")
    print("-----------------------------------")

    if average_disagreement < 0.10:

        print(
            "Models are making very similar predictions."
        )

        print(
            "Ensemble diversity is currently low."
        )

    elif average_disagreement < 0.30:

        print(
            "Models show some prediction differences."
        )

        print(
            "There is moderate ensemble diversity."
        )

    else:

        print(
            "Models are making substantially different predictions."
        )

        print(
            "Ensemble diversity is relatively high."
        )

    print("\n===================================")
    print("ENSEMBLE DIVERSITY ANALYSIS COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_ensemble_diversity(
        target_column="col1"
    )