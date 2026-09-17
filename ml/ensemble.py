import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ml.training_data import build_training_dataset
from ml.time_split import time_aware_split


def run_ensemble(target_column="col1"):

    print("\n===================================")
    print("        SOFT VOTING ENSEMBLE")
    print("===================================")

    print(
        f"Target column: {target_column}"
    )

    # -----------------------------------
    # LOAD DATA
    # -----------------------------------

    dataset = build_training_dataset(
        target_column
    )

    if dataset.empty:

        print(
            "No training data available."
        )

        return

    # -----------------------------------
    # TIME-AWARE SPLIT
    # -----------------------------------

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

    # -----------------------------------
    # CLASSES
    # -----------------------------------

    classes = sorted(
        y_train.unique()
    )

    class_count = len(classes)

    print("\n-----------------------------------")
    print("DATA")
    print("-----------------------------------")

    print(
        f"Training rows: "
        f"{len(X_train)}"
    )

    print(
        f"Testing rows: "
        f"{len(X_test)}"
    )

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Classes: "
        f"{class_count}"
    )

    if class_count < 2:

        print(
            "\nAt least 2 target classes "
            "are required."
        )

        return

    # -----------------------------------
    # CLASS ENCODING
    # -----------------------------------

    class_to_index = {
        value: index
        for index, value in enumerate(
            classes
        )
    }

    index_to_class = {
        index: value
        for value, index
        in class_to_index.items()
    }

    y_train_encoded = y_train.map(
        class_to_index
    )

    y_test_encoded = y_test.map(
        class_to_index
    )

    if y_test_encoded.isna().any():

        print(
            "\nWARNING:"
        )

        print(
            "A test class was not present "
            "in training."
        )

        print(
            "Ensemble evaluation cannot "
            "continue safely."
        )

        return

    # ===================================
    # MODELS
    # ===================================

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

    # -----------------------------------
    # STORE PROBABILITIES
    # -----------------------------------

    probability_predictions = {}

    # ===================================
    # TRAIN MODELS
    # ===================================

    print("\n===================================")
    print("TRAINING ENSEMBLE MODELS")
    print("===================================")

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        try:

            if name in [
                "XGBoost",
                "LightGBM",
                "CatBoost"
            ]:

                model.fit(
                    X_train,
                    y_train_encoded
                )

            else:

                model.fit(
                    X_train,
                    y_train
                )

            probabilities = (
                model.predict_proba(
                    X_test
                )
            )

            probability_predictions[
                name
            ] = probabilities

            print(
                f"{name} completed."
            )

        except Exception as error:

            print(
                f"{name} failed:"
            )

            print(error)

    # -----------------------------------
    # CHECK MODELS
    # -----------------------------------

    if len(
        probability_predictions
    ) == 0:

        print(
            "\nNo models completed."
        )

        return

    # ===================================
    # SOFT VOTING
    # ===================================

    print("\n===================================")
    print("CALCULATING ENSEMBLE")
    print("===================================")

    all_probabilities = list(
        probability_predictions.values()
    )

    ensemble_probabilities = np.mean(
        all_probabilities,
        axis=0
    )

    # -----------------------------------
    # NORMALIZE
    # -----------------------------------

    ensemble_probabilities = (
        ensemble_probabilities
        /
        ensemble_probabilities.sum(
            axis=1,
            keepdims=True
        )
    )

    # -----------------------------------
    # FINAL PREDICTIONS
    # -----------------------------------

    predicted_indices = (
        ensemble_probabilities.argmax(
            axis=1
        )
    )

    predictions = [
        index_to_class[index]
        for index in predicted_indices
    ]

    # ===================================
    # INDIVIDUAL MODEL PREDICTIONS
    # ===================================

    print("\n-----------------------------------")
    print("INDIVIDUAL MODEL PREDICTIONS")
    print("-----------------------------------")

    for name, probabilities in (
        probability_predictions.items()
    ):

        model_predictions = (
            probabilities.argmax(
                axis=1
            )
        )

        readable_predictions = [
            index_to_class[index]
            for index in model_predictions
        ]

        print(
            f"\n{name}:"
        )

        for index, prediction in enumerate(
            readable_predictions
        ):

            print(
                f"  Test {index + 1}: "
                f"{prediction}"
            )

    # ===================================
    # ENSEMBLE RESULTS
    # ===================================

    print("\n===================================")
    print("ENSEMBLE PREDICTIONS")
    print("===================================")

    for index in range(
        len(y_test)
    ):

        actual = y_test.iloc[index]

        predicted = predictions[index]

        confidence = (
            ensemble_probabilities[
                index
            ].max()
            * 100
        )

        result = (
            "CORRECT"
            if actual == predicted
            else "WRONG"
        )

        print(
            f"{index + 1}. "
            f"Actual={actual}, "
            f"Ensemble={predicted}, "
            f"Probability="
            f"{confidence:.2f}% "
            f"-> {result}"
        )

    # ===================================
    # TOP ENSEMBLE PROBABILITIES
    # ===================================

    print("\n-----------------------------------")
    print("TOP ENSEMBLE PROBABILITIES")
    print("-----------------------------------")

    for index in range(
        len(y_test)
    ):

        probability_row = (
            ensemble_probabilities[
                index
            ]
        )

        ranked_indices = (
            probability_row.argsort()[
                ::-1
            ]
        )

        print(
            f"\nTest record {index + 1}:"
        )

        for rank, class_index in enumerate(
            ranked_indices[:5],
            start=1
        ):

            predicted_class = (
                index_to_class[
                    class_index
                ]
            )

            probability = (
                probability_row[
                    class_index
                ]
                * 100
            )

            print(
                f"  {rank}. "
                f"{predicted_class} "
                f"-> "
                f"{probability:.2f}%"
            )

    # ===================================
    # MODEL AGREEMENT
    # ===================================

    print("\n-----------------------------------")
    print("MODEL AGREEMENT")
    print("-----------------------------------")

    for index in range(
        len(y_test)
    ):

        predictions_for_record = []

        for probabilities in (
            probability_predictions.values()
        ):

            predicted_index = (
                probabilities[index].argmax()
            )

            predictions_for_record.append(
                index_to_class[
                    predicted_index
                ]
            )

        agreement = (
            sum(
                prediction
                == predictions_for_record[0]
                for prediction
                in predictions_for_record
            )
        )

        total_models = len(
            predictions_for_record
        )

        print(
            f"Test record {index + 1}: "
            f"{agreement}/"
            f"{total_models} models agree"
        )

    print("\n===================================")
    print("SOFT VOTING ENSEMBLE COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_ensemble("col1")