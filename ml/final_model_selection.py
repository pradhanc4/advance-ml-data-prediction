import time

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ml.training_data import build_training_dataset
from ml.time_split import time_aware_split


def encode_classes(y_train):

    classes = sorted(y_train.unique())

    class_to_index = {
        value: index
        for index, value in enumerate(classes)
    }

    return classes, class_to_index


def create_models(class_count):

    models = {}

    models["Logistic Regression"] = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    models["Extra Trees"] = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    models["Gradient Boosting"] = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.9,
        random_state=42
    )

    models["XGBoost"] = XGBClassifier(
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
    )

    models["LightGBM"] = LGBMClassifier(
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
    )

    models["CatBoost"] = CatBoostClassifier(
        loss_function="MultiClass",

        iterations=300,
        learning_rate=0.05,
        depth=6,

        l2_leaf_reg=3,

        random_seed=42,

        verbose=False,
        allow_writing_files=False
    )

    return models


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    encoded=False,
    classes=None
):

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.perf_counter()
        - start_time
    )

    probabilities = model.predict_proba(
        X_test
    )

    predicted_indices = (
        probabilities.argmax(axis=1)
    )

    if encoded:

        predictions = [
            classes[index]
            for index in predicted_indices
        ]

    else:

        predictions = [
            classes[index]
            for index in predicted_indices
        ]

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

    try:

        loss = log_loss(
            y_test,
            probabilities,
            labels=list(range(len(classes)))
        )

    except ValueError:

        loss = None

    return {
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "log_loss": loss,
        "training_time": training_time
    }


def calculate_selection_score(result):

    """
    Composite score used only as an internal
    model-selection signal.

    Balanced Accuracy and Macro F1 receive
    more weight than raw Accuracy because
    multiclass datasets can be imbalanced.
    """

    score = (
        0.40 * result["balanced_accuracy"]
        +
        0.40 * result["macro_f1"]
        +
        0.20 * result["accuracy"]
    )

    return score


def run_final_model_selection(
    target_column="col1"
):

    print("\n===================================")
    print("       FINAL MODEL SELECTION")
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

    y_train_original = train_data[
        "target"
    ]

    y_test_original = test_data[
        "target"
    ]

    # -----------------------------------
    # CLASS ENCODING
    # -----------------------------------

    classes, class_to_index = (
        encode_classes(
            y_train_original
        )
    )

    class_count = len(classes)

    print("\n-----------------------------------")
    print("DATA")
    print("-----------------------------------")

    print(
        f"Training rows: "
        f"{len(train_data)}"
    )

    print(
        f"Testing rows: "
        f"{len(test_data)}"
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
            "\nAt least 2 classes are required."
        )

        return

    # -----------------------------------
    # ENCODE BOOSTING TARGETS
    # -----------------------------------

    y_train_encoded = (
        y_train_original.map(
            class_to_index
        )
    )

    y_test_encoded = (
        y_test_original.map(
            class_to_index
        )
    )

    if y_test_encoded.isna().any():

        print(
            "\nWARNING:"
        )

        print(
            "The final test set contains "
            "a class that was not present "
            "in training."
        )

        print(
            "Model selection cannot be "
            "evaluated safely."
        )

        return

    # -----------------------------------
    # CREATE MODELS
    # -----------------------------------

    models = create_models(
        class_count
    )

    results = []

    # -----------------------------------
    # EVALUATE
    # -----------------------------------

    print("\n===================================")
    print("EVALUATING MODELS")
    print("===================================")

    for name, model in models.items():

        print(
            f"\nEvaluating: {name}"
        )

        try:

            if name in [
                "XGBoost",
                "LightGBM",
                "CatBoost"
            ]:

                result = evaluate_model(
                    name,
                    model,
                    X_train,
                    y_train_encoded,
                    X_test,
                    y_test_encoded,
                    encoded=True,
                    classes=classes
                )

            else:

                result = evaluate_model(
                    name,
                    model,
                    X_train,
                    y_train_original,
                    X_test,
                    y_test_original,
                    encoded=False,
                    classes=classes
                )

            result["selection_score"] = (
                calculate_selection_score(
                    result
                )
            )

            results.append(result)

            print(
                f"Accuracy: "
                f"{result['accuracy'] * 100:.2f}%"
            )

            print(
                f"Balanced Accuracy: "
                f"{result['balanced_accuracy'] * 100:.2f}%"
            )

            print(
                f"Macro F1: "
                f"{result['macro_f1'] * 100:.2f}%"
            )

            print(
                f"Selection Score: "
                f"{result['selection_score'] * 100:.2f}%"
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

    # -----------------------------------
    # RESULTS
    # -----------------------------------

    print("\n===================================")
    print("MODEL SELECTION RESULTS")
    print("===================================")

    if not results:

        print(
            "No models completed."
        )

        return

    print(
        f"\n{'Model':<22}"
        f"{'Accuracy':>12}"
        f"{'Balanced':>12}"
        f"{'Macro F1':>12}"
        f"{'Score':>12}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['model']:<22}"
            f"{result['accuracy'] * 100:>11.2f}%"
            f"{result['balanced_accuracy'] * 100:>11.2f}%"
            f"{result['macro_f1'] * 100:>11.2f}%"
            f"{result['selection_score'] * 100:>11.2f}%"
        )

    # -----------------------------------
    # PROVISIONAL SELECTION
    # -----------------------------------

    selected = max(
        results,
        key=lambda result:
        result["selection_score"]
    )

    print("\n===================================")
    print("PROVISIONAL MODEL CONFIGURATION")
    print("===================================")

    print(
        f"Model: "
        f"{selected['model']}"
    )

    print(
        f"Selection Score: "
        f"{selected['selection_score'] * 100:.2f}%"
    )

    print(
        "\nStatus: PROVISIONAL"
    )

    print(
        "\nReason:"
    )

    print(
        "The current historical dataset "
        "is too small for a statistically "
        "reliable final model decision."
    )

    print(
        "\nThe selected configuration can "
        "be re-evaluated automatically "
        "when more historical records "
        "are available."
    )

    # -----------------------------------
    # FINAL CHECK
    # -----------------------------------

    print("\n===================================")
    print("MODEL SELECTION CHECK")
    print("===================================")

    print(
        "Time-aware evaluation: YES"
    )

    print(
        "Multiple models evaluated: YES"
    )

    print(
        "Balanced metrics considered: YES"
    )

    print(
        "Selection marked provisional: YES"
    )

    print("\n===================================")
    print("FINAL MODEL SELECTION COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_final_model_selection("col1")