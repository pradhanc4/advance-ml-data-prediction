import numpy as np

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ml.training_data import build_training_dataset
from ml.time_split import time_aware_split


def normalize_weights(weights):

    total = sum(weights.values())

    if total <= 0:
        raise ValueError(
            "Total ensemble weight must be greater than zero."
        )

    return {
        name: weight / total
        for name, weight in weights.items()
    }


def run_weighted_ensemble(target_column="col1"):

    print("\n===================================")
    print("        WEIGHTED ENSEMBLE")
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
            "The test set contains a class "
            "that was not present in training."
        )

        print(
            "Weighted ensemble cannot "
            "continue safely."
        )

        return

    # ===================================
    # MODEL WEIGHTS
    # ===================================

    raw_weights = {

        "Random Forest": 0.25,

        "XGBoost": 0.25,

        "LightGBM": 0.25,

        "CatBoost": 0.25
    }

    weights = normalize_weights(
        raw_weights
    )

    print("\n-----------------------------------")
    print("MODEL WEIGHTS")
    print("-----------------------------------")

    for name, weight in weights.items():

        print(
            f"{name}: "
            f"{weight * 100:.2f}%"
        )

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
    # STORE MODEL PROBABILITIES
    # -----------------------------------

    model_probabilities = {}

    # ===================================
    # TRAIN MODELS
    # ===================================

    print("\n===================================")
    print("TRAINING MODELS")
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

            model_probabilities[
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

    if not model_probabilities:

        print(
            "\nNo models completed."
        )

        return

    # ===================================
    # WEIGHTED PROBABILITY AGGREGATION
    # ===================================

    print("\n===================================")
    print("CALCULATING WEIGHTED PROBABILITIES")
    print("===================================")

    ensemble_probabilities = None

    total_weight = 0.0

    for name, probabilities in (
        model_probabilities.items()
    ):

        weight = weights[name]

        weighted_probability = (
            probabilities * weight
        )

        if ensemble_probabilities is None:

            ensemble_probabilities = (
                weighted_probability.copy()
            )

        else:

            ensemble_probabilities += (
                weighted_probability
            )

        total_weight += weight

    # -----------------------------------
    # NORMALIZE
    # -----------------------------------

    if total_weight > 0:

        ensemble_probabilities /= (
            total_weight
        )

    # Make absolutely sure each row
    # represents a probability distribution.

    row_sums = (
        ensemble_probabilities.sum(
            axis=1,
            keepdims=True
        )
    )

    ensemble_probabilities /= row_sums

    # ===================================
    # FINAL PREDICTIONS
    # ===================================

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
    # RESULTS
    # ===================================

    print("\n===================================")
    print("WEIGHTED ENSEMBLE RESULTS")
    print("===================================")

    for index in range(
        len(y_test)
    ):

        actual = y_test.iloc[index]

        predicted = predictions[index]

        probability = (
            ensemble_probabilities[
                index
            ]
        )

        confidence = (
            probability.max()
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
            f"Prediction={predicted}, "
            f"Probability="
            f"{confidence:.2f}% "
            f"-> {result}"
        )

    # ===================================
    # TOP PREDICTIONS
    # ===================================

    print("\n-----------------------------------")
    print("TOP WEIGHTED PREDICTIONS")
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
    # MODEL CONTRIBUTION
    # ===================================

    print("\n-----------------------------------")
    print("MODEL CONTRIBUTION")
    print("-----------------------------------")

    for name in model_probabilities:

        print(
            f"{name}: "
            f"{weights[name] * 100:.2f}%"
        )

    print("\n===================================")
    print("WEIGHTED ENSEMBLE COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_weighted_ensemble("col1")
    