import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ml.training_data import build_training_dataset
from ml.time_split import time_aware_split


def chronological_three_way_split(dataset):

    # -----------------------------------
    # FINAL TEST SET
    # -----------------------------------

    train_validation, test_data = time_aware_split(
        dataset,
        test_size=0.20
    )

    if len(train_validation) < 6:
        return None, None, None

    # -----------------------------------
    # CALIBRATION SET
    # -----------------------------------

    calibration_rows = max(
        1,
        int(len(train_validation) * 0.20)
    )

    if calibration_rows >= len(train_validation):
        return None, None, None

    train_data = train_validation.iloc[
        :-calibration_rows
    ].copy()

    calibration_data = train_validation.iloc[
        -calibration_rows:
    ].copy()

    return (
        train_data,
        calibration_data,
        test_data
    )


def encode_classes(y_train):

    classes = sorted(
        y_train.unique()
    )

    class_to_index = {
        value: index
        for index, value in enumerate(classes)
    }

    return classes, class_to_index


def temperature_scale(
    probabilities,
    temperature
):

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    # Prevent log(0)
    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    logits = np.log(
        probabilities
    )

    scaled_logits = (
        logits / temperature
    )

    scaled_logits = (
        scaled_logits
        - scaled_logits.max(
            axis=1,
            keepdims=True
        )
    )

    exp_logits = np.exp(
        scaled_logits
    )

    calibrated = (
        exp_logits
        / exp_logits.sum(
            axis=1,
            keepdims=True
        )
    )

    return calibrated


def find_best_temperature(
    calibration_probabilities,
    y_calibration
):

    temperatures = np.linspace(
        0.50,
        5.00,
        91
    )

    best_temperature = 1.0
    best_loss = float("inf")

    for temperature in temperatures:

        calibrated_probabilities = (
            temperature_scale(
                calibration_probabilities,
                temperature
            )
        )

        try:

            loss = log_loss(
                y_calibration,
                calibrated_probabilities
            )

        except ValueError:

            continue

        if loss < best_loss:

            best_loss = loss
            best_temperature = temperature

    return (
        best_temperature,
        best_loss
    )


def evaluate_calibration(
    model_name,
    model,
    X_train,
    y_train,
    X_calibration,
    y_calibration,
    X_test,
    y_test,
    encoded=False
):

    print("\n-----------------------------------")
    print(f"MODEL: {model_name}")
    print("-----------------------------------")

    # -----------------------------------
    # TRAIN
    # -----------------------------------

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------
    # CALIBRATION PROBABILITIES
    # -----------------------------------

    calibration_probabilities = (
        model.predict_proba(
            X_calibration
        )
    )

    # -----------------------------------
    # CHECK CALIBRATION LABELS
    # -----------------------------------

    if encoded:

        calibration_classes = np.unique(
            y_calibration
        )

        model_classes = np.arange(
            calibration_probabilities.shape[1]
        )

        if not set(
            calibration_classes
        ).issubset(
            set(model_classes)
        ):

            print(
                "Calibration skipped: "
                "unseen target class."
            )

            return None

    # -----------------------------------
    # RAW CALIBRATION LOSS
    # -----------------------------------

    try:

        raw_calibration_loss = log_loss(
            y_calibration,
            calibration_probabilities,
            labels=np.arange(
                calibration_probabilities.shape[1]
            )
        )

    except ValueError:

        print(
            "Calibration skipped: "
            "insufficient class information."
        )

        return None

    # -----------------------------------
    # FIND TEMPERATURE
    # -----------------------------------

    best_temperature, calibrated_loss = (
        find_best_temperature(
            calibration_probabilities,
            y_calibration
        )
    )

    # -----------------------------------
    # TEST PROBABILITIES
    # -----------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test
        )
    )

    # Raw probabilities
    try:

        raw_test_loss = log_loss(
            y_test,
            test_probabilities,
            labels=np.arange(
                test_probabilities.shape[1]
            )
        )

    except ValueError:

        raw_test_loss = None

    # Calibrated probabilities
    calibrated_test_probabilities = (
        temperature_scale(
            test_probabilities,
            best_temperature
        )
    )

    try:

        calibrated_test_loss = log_loss(
            y_test,
            calibrated_test_probabilities,
            labels=np.arange(
                calibrated_test_probabilities.shape[1]
            )
        )

    except ValueError:

        calibrated_test_loss = None

    # -----------------------------------
    # DISPLAY
    # -----------------------------------

    print(
        f"Raw calibration Log Loss: "
        f"{raw_calibration_loss:.4f}"
    )

    print(
        f"Best temperature: "
        f"{best_temperature:.2f}"
    )

    print(
        f"Calibrated validation Log Loss: "
        f"{calibrated_loss:.4f}"
    )

    if raw_test_loss is not None:

        print(
            f"Raw test Log Loss: "
            f"{raw_test_loss:.4f}"
        )

    else:

        print(
            "Raw test Log Loss: N/A"
        )

    if calibrated_test_loss is not None:

        print(
            f"Calibrated test Log Loss: "
            f"{calibrated_test_loss:.4f}"
        )

    else:

        print(
            "Calibrated test Log Loss: N/A"
        )

    # -----------------------------------
    # TEST PROBABILITIES
    # -----------------------------------

    print("\nTEST PROBABILITIES")

    for index in range(
        len(y_test)
    ):

        raw_confidence = (
            test_probabilities[index].max()
            * 100
        )

        calibrated_confidence = (
            calibrated_test_probabilities[
                index
            ].max()
            * 100
        )

        print(
            f"Record {index + 1}: "
            f"Raw={raw_confidence:.2f}% | "
            f"Calibrated="
            f"{calibrated_confidence:.2f}%"
        )

    return {
        "model": model_name,
        "temperature": best_temperature,
        "raw_calibration_loss":
            raw_calibration_loss,
        "calibrated_validation_loss":
            calibrated_loss,
        "raw_test_loss":
            raw_test_loss,
        "calibrated_test_loss":
            calibrated_test_loss
    }


def run_probability_calibration(
    target_column="col1"
):

    print("\n===================================")
    print("       PROBABILITY CALIBRATION")
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

    (
        train_data,
        calibration_data,
        test_data
    ) = chronological_three_way_split(
        dataset
    )

    if train_data is None:

        print(
            "\nNot enough data for "
            "three-way calibration."
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

    X_train = train_data[
        feature_columns
    ]

    X_calibration = calibration_data[
        feature_columns
    ]

    X_test = test_data[
        feature_columns
    ]

    y_train = train_data[
        "target"
    ]

    y_calibration_original = (
        calibration_data["target"]
    )

    y_test_original = (
        test_data["target"]
    )

    # -----------------------------------
    # CLASS ENCODING
    # -----------------------------------

    classes, class_to_index = (
        encode_classes(y_train)
    )

    class_count = len(classes)

    print("\n-----------------------------------")
    print("THREE-WAY TIME SPLIT")
    print("-----------------------------------")

    print(
        f"Training rows: "
        f"{len(train_data)}"
    )

    print(
        f"Calibration rows: "
        f"{len(calibration_data)}"
    )

    print(
        f"Final test rows: "
        f"{len(test_data)}"
    )

    print(
        f"Training classes: "
        f"{class_count}"
    )

    if class_count < 2:

        print(
            "At least 2 classes are required."
        )

        return

    # -----------------------------------
    # ENCODE TARGETS
    # -----------------------------------

    y_train_encoded = (
        y_train.map(
            class_to_index
        )
    )

    y_calibration_encoded = (
        y_calibration_original.map(
            class_to_index
        )
    )

    y_test_encoded = (
        y_test_original.map(
            class_to_index
        )
    )

    # -----------------------------------
    # CHECK UNSEEN CLASSES
    # -----------------------------------

    if (
        y_calibration_encoded.isna().any()
        or y_test_encoded.isna().any()
    ):

        print(
            "\nSome calibration/test values "
            "were not present in training."
        )

        print(
            "This is expected with very small "
            "datasets."
        )

        print(
            "Calibration cannot be evaluated "
            "safely for these records."
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

    results = []

    # ===================================
    # CALIBRATE EACH MODEL
    # ===================================

    for model_name, model in models.items():

        result = evaluate_calibration(
            model_name,
            model,
            X_train,
            y_train_encoded,
            X_calibration,
            y_calibration_encoded,
            X_test,
            y_test_encoded,
            encoded=True
        )

        if result is not None:

            results.append(result)

    # ===================================
    # SUMMARY
    # ===================================

    print("\n===================================")
    print("CALIBRATION SUMMARY")
    print("===================================")

    if not results:

        print(
            "No models could be calibrated "
            "with the current dataset."
        )

        print(
            "\nThis is normal with very small "
            "historical data."
        )

        return

    for result in results:

        print(
            f"\n{result['model']}"
        )

        print(
            f"  Temperature: "
            f"{result['temperature']:.2f}"
        )

        if (
            result["raw_test_loss"]
            is not None
        ):

            print(
                f"  Raw Test Log Loss: "
                f"{result['raw_test_loss']:.4f}"
            )

        if (
            result["calibrated_test_loss"]
            is not None
        ):

            print(
                f"  Calibrated Test Log Loss: "
                f"{result['calibrated_test_loss']:.4f}"
            )

    print("\n===================================")
    print("PROBABILITY CALIBRATION COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_probability_calibration("col1")