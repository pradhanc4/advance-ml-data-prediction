import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ml.training_data import build_training_dataset
from ml.time_split import time_aware_split


def encode_target(y_train, y_validation):

    classes = sorted(y_train.unique())

    class_to_index = {
        value: index
        for index, value in enumerate(classes)
    }

    y_train_encoded = y_train.map(class_to_index)
    y_validation_encoded = y_validation.map(
        class_to_index
    )

    return (
        classes,
        y_train_encoded,
        y_validation_encoded
    )


def evaluate_model(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
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

    predictions = model.predict(
        X_validation
    )

    if encoded and classes is not None:

        predictions = [
            classes[int(prediction)]
            for prediction in predictions
        ]

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_validation,
            predictions
        )
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "training_time": training_time
    }


def create_validation_split(dataset):

    # First split:
    # historical training data vs final test data

    train_data, test_data = time_aware_split(
        dataset,
        test_size=0.20
    )

    # Second split:
    # training data vs chronological validation data

    if len(train_data) < 5:

        return None, None, None

    validation_rows = max(
        1,
        int(len(train_data) * 0.20)
    )

    if validation_rows >= len(train_data):

        return None, None, None

    inner_train = train_data.iloc[
        :-validation_rows
    ].copy()

    validation = train_data.iloc[
        -validation_rows:
    ].copy()

    return (
        inner_train,
        validation,
        test_data
    )


def run_hyperparameter_tuning(
    target_column="col1"
):

    print("\n===================================")
    print("       HYPERPARAMETER TUNING")
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

    inner_train, validation, test_data = (
        create_validation_split(dataset)
    )

    if inner_train is None:

        print(
            "\nNot enough data for "
            "time-aware tuning."
        )

        return

    feature_columns = [
        column
        for column in dataset.columns
        if column not in [
            "target",
            "record_date"
        ]
    ]

    X_train = inner_train[
        feature_columns
    ]

    X_validation = validation[
        feature_columns
    ]

    y_train = inner_train["target"]
    y_validation = validation["target"]

    print("\n-----------------------------------")
    print("TIME-AWARE DATA SPLIT")
    print("-----------------------------------")

    print(
        f"Model training rows: "
        f"{len(inner_train)}"
    )

    print(
        f"Validation rows: "
        f"{len(validation)}"
    )

    print(
        f"Final test rows: "
        f"{len(test_data)}"
    )

    print(
        "\nFinal test data is kept "
        "separate from tuning."
    )

    classes, y_train_encoded, (
        y_validation_encoded
    ) = encode_target(
        y_train,
        y_validation
    )

    class_count = len(classes)

    print(
        f"Training classes: "
        f"{class_count}"
    )

    if class_count < 2:

        print(
            "\nNot enough target classes "
            "for tuning."
        )

        return

    # ===================================
    # RANDOM FOREST
    # ===================================

    print("\n===================================")
    print("TUNING RANDOM FOREST")
    print("===================================")

    rf_parameter_sets = [

        {
            "n_estimators": 200,
            "max_depth": 5,
            "min_samples_split": 4,
            "min_samples_leaf": 2
        },

        {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2
        },

        {
            "n_estimators": 400,
            "max_depth": 10,
            "min_samples_split": 3,
            "min_samples_leaf": 1
        },

        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }
    ]

    rf_results = []

    for params in rf_parameter_sets:

        print(
            f"\nTesting parameters: "
            f"{params}"
        )

        model = RandomForestClassifier(
            **params,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        try:

            result = evaluate_model(
                model,
                X_train,
                y_train,
                X_validation,
                y_validation
            )

            result["parameters"] = params

            rf_results.append(result)

            print(
                f"Balanced Accuracy: "
                f"{result['balanced_accuracy'] * 100:.2f}%"
            )

            print(
                f"Macro F1: "
                f"{result['macro_f1'] * 100:.2f}%"
            )

        except Exception as error:

            print(
                f"Skipped: {error}"
            )

    # ===================================
    # XGBOOST
    # ===================================

    print("\n===================================")
    print("TUNING XGBOOST")
    print("===================================")

    xgb_parameter_sets = [

        {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05
        },

        {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05
        },

        {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.03
        }
    ]

    xgb_results = []

    for params in xgb_parameter_sets:

        print(
            f"\nTesting parameters: "
            f"{params}"
        )

        model = XGBClassifier(
            **params,

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

        try:

            result = evaluate_model(
                model,
                X_train,
                y_train_encoded,
                X_validation,
                y_validation_encoded,
                encoded=True,
                classes=classes
            )

            result["parameters"] = params

            xgb_results.append(result)

            print(
                f"Balanced Accuracy: "
                f"{result['balanced_accuracy'] * 100:.2f}%"
            )

            print(
                f"Macro F1: "
                f"{result['macro_f1'] * 100:.2f}%"
            )

        except Exception as error:

            print(
                f"Skipped: {error}"
            )

    # ===================================
    # LIGHTGBM
    # ===================================

    print("\n===================================")
    print("TUNING LIGHTGBM")
    print("===================================")

    lgb_parameter_sets = [

        {
            "n_estimators": 100,
            "num_leaves": 15,
            "max_depth": 4,
            "learning_rate": 0.05
        },

        {
            "n_estimators": 200,
            "num_leaves": 31,
            "max_depth": 6,
            "learning_rate": 0.05
        },

        {
            "n_estimators": 300,
            "num_leaves": 15,
            "max_depth": 5,
            "learning_rate": 0.03
        }
    ]

    lgb_results = []

    for params in lgb_parameter_sets:

        print(
            f"\nTesting parameters: "
            f"{params}"
        )

        model = LGBMClassifier(
            **params,

            min_child_samples=5,

            subsample=0.9,
            colsample_bytree=0.9,

            objective="multiclass",
            num_class=class_count,

            reg_alpha=0.1,
            reg_lambda=1.0,

            random_state=42,
            n_jobs=-1,

            verbosity=-1
        )

        try:

            result = evaluate_model(
                model,
                X_train,
                y_train_encoded,
                X_validation,
                y_validation_encoded,
                encoded=True,
                classes=classes
            )

            result["parameters"] = params

            lgb_results.append(result)

            print(
                f"Balanced Accuracy: "
                f"{result['balanced_accuracy'] * 100:.2f}%"
            )

            print(
                f"Macro F1: "
                f"{result['macro_f1'] * 100:.2f}%"
            )

        except Exception as error:

            print(
                f"Skipped: {error}"
            )

    # ===================================
    # CATBOOST
    # ===================================

    print("\n===================================")
    print("TUNING CATBOOST")
    print("===================================")

    cat_parameter_sets = [

        {
            "iterations": 150,
            "depth": 4,
            "learning_rate": 0.05
        },

        {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05
        },

        {
            "iterations": 400,
            "depth": 7,
            "learning_rate": 0.03
        }
    ]

    cat_results = []

    for params in cat_parameter_sets:

        print(
            f"\nTesting parameters: "
            f"{params}"
        )

        model = CatBoostClassifier(
            **params,

            loss_function="MultiClass",

            l2_leaf_reg=3,

            random_seed=42,

            verbose=False,

            allow_writing_files=False
        )

        try:

            result = evaluate_model(
                model,
                X_train,
                y_train_encoded,
                X_validation,
                y_validation_encoded,
                encoded=True,
                classes=classes
            )

            result["parameters"] = params

            cat_results.append(result)

            print(
                f"Balanced Accuracy: "
                f"{result['balanced_accuracy'] * 100:.2f}%"
            )

            print(
                f"Macro F1: "
                f"{result['macro_f1'] * 100:.2f}%"
            )

        except Exception as error:

            print(
                f"Skipped: {error}"
            )

    # ===================================
    # SUMMARY
    # ===================================

    all_results = []

    for name, results in [
        ("Random Forest", rf_results),
        ("XGBoost", xgb_results),
        ("LightGBM", lgb_results),
        ("CatBoost", cat_results)
    ]:

        for result in results:

            all_results.append({
                "model": name,
                **result
            })

    print("\n===================================")
    print("TUNING SUMMARY")
    print("===================================")

    if not all_results:

        print("No tuning results available.")
        return

    for result in all_results:

        print(
            f"\n{result['model']}"
        )

        print(
            f"  Accuracy: "
            f"{result['accuracy'] * 100:.2f}%"
        )

        print(
            f"  Balanced Accuracy: "
            f"{result['balanced_accuracy'] * 100:.2f}%"
        )

        print(
            f"  Macro F1: "
            f"{result['macro_f1'] * 100:.2f}%"
        )

        print(
            f"  Training Time: "
            f"{result['training_time']:.4f}s"
        )

        print(
            f"  Parameters: "
            f"{result['parameters']}"
        )

    print("\n===================================")
    print("HYPERPARAMETER TUNING COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_hyperparameter_tuning("col1")