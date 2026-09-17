import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ml.training_data import build_training_dataset
from ml.time_split import time_aware_split


def normalize_rows(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)

    row_sums[row_sums == 0] = 1

    return matrix / row_sums


def probability_aggregation(model_probabilities):
    """
    Average probabilities from all models.
    """

    probability_stack = np.stack(
        list(model_probabilities.values()),
        axis=0
    )

    return np.mean(
        probability_stack,
        axis=0
    )


def rank_aggregation(model_probabilities):
    """
    Convert each model's probabilities into normalized
    rank scores.

    Highest probability gets the highest rank score.
    """

    rank_scores = []

    for probabilities in model_probabilities.values():

        rows = []

        class_count = probabilities.shape[1]

        for row in probabilities:

            ranking = np.argsort(
                np.argsort(-row)
            )

            if class_count == 1:
                score = np.ones(class_count)
            else:
                score = (
                    class_count - ranking
                ) / (class_count - 1)

            score = normalize_rows(
                score.reshape(1, -1)
            )[0]

            rows.append(score)

        rank_scores.append(
            np.array(rows)
        )

    rank_stack = np.stack(
        rank_scores,
        axis=0
    )

    return np.mean(
        rank_stack,
        axis=0
    )


def run_rank_probability_aggregation(
    target_column="col1",
    probability_weight=0.70,
    rank_weight=0.30
):

    print("\n===================================")
    print("   PROBABILITY + RANK AGGREGATION")
    print("===================================")

    print(
        f"Target column: {target_column}"
    )

    dataset = build_training_dataset(
        target_column
    )

    if dataset.empty:
        print("No training data available.")
        return

    train_data, test_data = time_aware_split(
        dataset,
        test_size=0.20
    )

    feature_columns = [
        column
        for column in dataset.columns
        if column not in [
            "target",
            "record_date"
        ]
    ]

    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]

    y_train = train_data["target"]
    y_test = test_data["target"]

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

    index_to_class = {
        index: value
        for value, index in class_to_index.items()
    }

    y_train_encoded = y_train.map(
        class_to_index
    )

    y_test_encoded = y_test.map(
        class_to_index
    )

    if y_test_encoded.isna().any():

        print("\nWARNING:")
        print(
            "Test data contains a class not seen during training."
        )

        print(
            "Aggregation cannot continue safely."
        )

        return

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

    model_probabilities = {}

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

            probabilities = model.predict_proba(
                X_test
            )

            model_probabilities[name] = (
                probabilities
            )

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

    # -----------------------------------
    # PROBABILITY AGGREGATION
    # -----------------------------------

    probability_scores = (
        probability_aggregation(
            model_probabilities
        )
    )

    probability_scores = normalize_rows(
        probability_scores
    )

    # -----------------------------------
    # RANK AGGREGATION
    # -----------------------------------

    rank_scores = (
        rank_aggregation(
            model_probabilities
        )
    )

    rank_scores = normalize_rows(
        rank_scores
    )

    # -----------------------------------
    # HYBRID AGGREGATION
    # -----------------------------------

    total_weight = (
        probability_weight +
        rank_weight
    )

    if total_weight <= 0:

        raise ValueError(
            "Aggregation weights must be greater than zero."
        )

    probability_weight /= total_weight
    rank_weight /= total_weight

    hybrid_scores = (
        probability_scores *
        probability_weight
    ) + (
        rank_scores *
        rank_weight
    )

    hybrid_scores = normalize_rows(
        hybrid_scores
    )

    predicted_indices = (
        hybrid_scores.argmax(axis=1)
    )

    predictions = [
        index_to_class[index]
        for index in predicted_indices
    ]

    # -----------------------------------
    # RESULTS
    # -----------------------------------

    print("\n===================================")
    print("AGGREGATION WEIGHTS")
    print("===================================")

    print(
        f"Probability: "
        f"{probability_weight * 100:.2f}%"
    )

    print(
        f"Rank: "
        f"{rank_weight * 100:.2f}%"
    )

    print("\n===================================")
    print("HYBRID AGGREGATION RESULTS")
    print("===================================")

    for index in range(
        len(y_test)
    ):

        actual = y_test.iloc[index]

        predicted = predictions[index]

        probability = (
            hybrid_scores[index]
        )

        confidence = (
            probability.max() * 100
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
            f"Probability={confidence:.2f}% "
            f"-> {result}"
        )

    # -----------------------------------
    # TOP PREDICTIONS
    # -----------------------------------

    print("\n-----------------------------------")
    print("TOP HYBRID PREDICTIONS")
    print("-----------------------------------")

    for index in range(
        len(y_test)
    ):

        print(
            f"\nTest record {index + 1}:"
        )

        ranking = (
            hybrid_scores[index]
            .argsort()[::-1]
        )

        for rank, class_index in enumerate(
            ranking[:5],
            start=1
        ):

            predicted_class = (
                index_to_class[class_index]
            )

            probability = (
                hybrid_scores[
                    index,
                    class_index
                ] * 100
            )

            print(
                f"  {rank}. "
                f"{predicted_class} "
                f"-> {probability:.2f}%"
            )

    # -----------------------------------
    # METRICS
    # -----------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n-----------------------------------")
    print("HYBRID ENSEMBLE METRICS")
    print("-----------------------------------")

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    print("\n===================================")
    print(
        "PROBABILITY + RANK AGGREGATION COMPLETED"
    )
    print("===================================\n")


if __name__ == "__main__":

    run_rank_probability_aggregation(
        target_column="col1",
        probability_weight=0.70,
        rank_weight=0.30
    )
    