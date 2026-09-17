import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ml.training_data import build_training_dataset


def create_models(class_count):

    return {

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


def create_time_folds(
    dataset,
    n_folds=3,
    test_size=0.20
):

    dataset = dataset.sort_values(
        "record_date"
    ).reset_index(drop=True)

    total_rows = len(dataset)

    test_rows = max(
        1,
        int(total_rows * test_size)
    )

    folds = []

    for fold in range(
        n_folds
    ):

        test_end = (
            total_rows -
            fold * test_rows
        )

        test_start = (
            test_end -
            test_rows
        )

        train_end = test_start

        if train_end < 2:
            continue

        train_data = dataset.iloc[
            :train_end
        ].copy()

        test_data = dataset.iloc[
            test_start:test_end
        ].copy()

        if len(test_data) == 0:
            continue

        folds.append(
            (
                fold + 1,
                train_data,
                test_data
            )
        )

    folds.reverse()

    return folds


def run_ensemble_validation(
    target_column="col1"
):

    print("\n===================================")
    print("       ENSEMBLE VALIDATION")
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

    folds = create_time_folds(
        dataset,
        n_folds=3,
        test_size=0.20
    )

    print("\n-----------------------------------")
    print("VALIDATION SETUP")
    print("-----------------------------------")

    print(
        f"Total rows: {len(dataset)}"
    )

    print(
        f"Validation folds: {len(folds)}"
    )

    if len(folds) < 2:

        print(
            "\nNot enough data for multi-fold validation."
        )

        print(
            "Load more historical records first."
        )

        return

    all_results = []

    for fold_number, train_data, test_data in folds:

        print("\n===================================")
        print(
            f"FOLD {fold_number}"
        )
        print("===================================")

        X_train = train_data.drop(
            columns=[
                "target",
                "record_date"
            ]
        )

        X_test = test_data.drop(
            columns=[
                "target",
                "record_date"
            ]
        )

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

        print(
            f"Training rows: {len(X_train)}"
        )

        print(
            f"Testing rows: {len(X_test)}"
        )

        print(
            f"Training classes: {class_count}"
        )

        if class_count < 2:

            print(
                "Skipping fold: fewer than 2 classes."
            )

            continue

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

            print(
                "Skipping fold: unseen test class."
            )

            continue

        models = create_models(
            class_count
        )

        probabilities = []

        model_predictions = {}

        for name, model in models.items():

            print(
                f"Training {name}..."
            )

            try:

                model.fit(
                    X_train,
                    y_train_encoded
                )

                model_probability = (
                    model.predict_proba(
                        X_test
                    )
                )

                probabilities.append(
                    model_probability
                )

                encoded_prediction = (
                    model_probability.argmax(
                        axis=1
                    )
                )

                model_prediction = [
                    index_to_class[
                        index
                    ]
                    for index in encoded_prediction
                ]

                model_predictions[name] = (
                    model_prediction
                )

            except Exception as error:

                print(
                    f"{name} failed:"
                )

                print(error)

        if not probabilities:

            print(
                "No models completed."
            )

            continue

        # -----------------------------------
        # SOFT VOTING
        # -----------------------------------

        ensemble_probability = np.mean(
            np.stack(
                probabilities,
                axis=0
            ),
            axis=0
        )

        ensemble_probability = (
            ensemble_probability /
            ensemble_probability.sum(
                axis=1,
                keepdims=True
            )
        )

        encoded_prediction = (
            ensemble_probability.argmax(
                axis=1
            )
        )

        ensemble_prediction = [
            index_to_class[
                index
            ]
            for index in encoded_prediction
        ]

        accuracy = accuracy_score(
            y_test,
            ensemble_prediction
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                ensemble_prediction
            )
        )

        macro_f1 = f1_score(
            y_test,
            ensemble_prediction,
            average="macro",
            zero_division=0
        )

        try:

            test_indices = [
                class_to_index[
                    value
                ]
                for value in y_test
            ]

            fold_log_loss = log_loss(
                test_indices,
                ensemble_probability,
                labels=list(
                    range(class_count)
                )
            )

        except Exception:

            fold_log_loss = np.nan

        print("\n-----------------------------------")
        print(
            f"FOLD {fold_number} RESULTS"
        )
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

        print(
            f"Log Loss: "
            f"{fold_log_loss:.4f}"
        )

        all_results.append({

            "fold": fold_number,

            "train_rows":
                len(train_data),

            "test_rows":
                len(test_data),

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "macro_f1":
                macro_f1,

            "log_loss":
                fold_log_loss
        })

    if not all_results:

        print(
            "\nNo valid validation folds."
        )

        return

    results_df = pd.DataFrame(
        all_results
    )

    # -----------------------------------
    # VALIDATION SUMMARY
    # -----------------------------------

    print("\n===================================")
    print("      VALIDATION SUMMARY")
    print("===================================")

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\n-----------------------------------")
    print("AVERAGE VALIDATION METRICS")
    print("-----------------------------------")

    print(
        f"Mean Accuracy: "
        f"{results_df['accuracy'].mean():.4f}"
    )

    print(
        f"Mean Balanced Accuracy: "
        f"{results_df['balanced_accuracy'].mean():.4f}"
    )

    print(
        f"Mean Macro F1: "
        f"{results_df['macro_f1'].mean():.4f}"
    )

    print(
        f"Mean Log Loss: "
        f"{results_df['log_loss'].mean():.4f}"
    )

    print("\n-----------------------------------")
    print("VALIDATION STABILITY")
    print("-----------------------------------")

    print(
        f"Accuracy Std Dev: "
        f"{results_df['accuracy'].std(ddof=0):.4f}"
    )

    print(
        f"Macro F1 Std Dev: "
        f"{results_df['macro_f1'].std(ddof=0):.4f}"
    )

    print(
        f"Log Loss Std Dev: "
        f"{results_df['log_loss'].std(ddof=0):.4f}"
    )

    print("\n-----------------------------------")
    print("IMPORTANT")
    print("-----------------------------------")

    if len(dataset) < 100:

        print(
            "Current dataset is still small."
        )

        print(
            "Validation results are experimental."
        )

        print(
            "Do not treat these metrics as reliable"
        )

        print(
            "real-world prediction performance yet."
        )

    else:

        print(
            "Dataset size is more suitable for validation."
        )

    print("\n===================================")
    print("      ENSEMBLE VALIDATION COMPLETED")
    print("===================================\n")


if __name__ == "__main__":

    run_ensemble_validation(
        target_column="col1"
    )