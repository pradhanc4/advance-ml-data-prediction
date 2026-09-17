import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

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
    total_rows,
    n_folds=4
):

    folds = []

    minimum_training_rows = 4

    for fold in range(
        n_folds
    ):

        test_end = (
            total_rows
            -
            (n_folds - 1 - fold)
        )

        test_start = test_end - 1

        if test_start < minimum_training_rows:
            continue

        train_indices = np.arange(
            0,
            test_start
        )

        test_indices = np.arange(
            test_start,
            test_end
        )

        folds.append(
            (
                fold + 1,
                train_indices,
                test_indices
            )
        )

    return folds


def run_oof_predictions(
    target_column="col1"
):

    print("\n===================================")
    print("       OUT-OF-FOLD PREDICTIONS")
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

    dataset = dataset.sort_values(
        "record_date"
    ).reset_index(
        drop=True
    )

    feature_columns = [
        column
        for column in dataset.columns
        if column not in [
            "target",
            "record_date"
        ]
    ]

    X = dataset[
        feature_columns
    ].copy()

    y = dataset[
        "target"
    ].copy()

    total_rows = len(dataset)

    print("\n-----------------------------------")
    print("DATA")
    print("-----------------------------------")

    print(
        f"Total rows: {total_rows}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    if total_rows < 8:

        print("\nWARNING:")
        print(
            "Very small dataset for OOF generation."
        )

        print(
            "More historical records are recommended."
        )

    # -----------------------------------
    # GLOBAL CLASS INFORMATION
    # -----------------------------------

    classes = sorted(
        y.unique()
    )

    class_count = len(classes)

    print(
        f"Target classes: {class_count}"
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

    y_encoded = y.map(
        class_to_index
    )

    # -----------------------------------
    # CREATE FOLDS
    # -----------------------------------

    folds = create_time_folds(
        total_rows,
        n_folds=4
    )

    print("\n-----------------------------------")
    print("OOF FOLD SETUP")
    print("-----------------------------------")

    print(
        f"Usable folds: {len(folds)}"
    )

    if len(folds) == 0:

        print(
            "Not enough historical records "
            "to create OOF predictions."
        )

        return

    # -----------------------------------
    # STORAGE
    # -----------------------------------

    model_names = [
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost"
    ]

    oof_probabilities = {

        name: np.full(
            (
                total_rows,
                class_count
            ),
            np.nan
        )

        for name in model_names
    }

    oof_predictions = {

        name: np.full(
            total_rows,
            np.nan
        )

        for name in model_names
    }

    used_rows = set()

    # -----------------------------------
    # OOF TRAINING
    # -----------------------------------

    print("\n===================================")
    print("GENERATING OOF PREDICTIONS")
    print("===================================")

    for fold_number, train_indices, test_indices in folds:

        print("\n-----------------------------------")
        print(
            f"OOF FOLD {fold_number}"
        )
        print("-----------------------------------")

        X_train = X.iloc[
            train_indices
        ]

        X_test = X.iloc[
            test_indices
        ]

        y_train = y_encoded.iloc[
            train_indices
        ]

        y_test = y_encoded.iloc[
            test_indices
        ]

        training_classes = sorted(
            y_train.unique()
        )

        print(
            f"Training rows: "
            f"{len(train_indices)}"
        )

        print(
            f"OOF rows: "
            f"{len(test_indices)}"
        )

        print(
            f"Training classes: "
            f"{len(training_classes)}"
        )

        if len(training_classes) < 2:

            print(
                "Skipping fold: "
                "training data has fewer than 2 classes."
            )

            continue

        models = create_models(
            class_count
        )

        for name, model in models.items():

            print(
                f"Generating {name} OOF predictions..."
            )

            try:

                model.fit(
                    X_train,
                    y_train
                )

                probabilities = (
                    model.predict_proba(
                        X_test
                    )
                )

                # -----------------------------------
                # ALIGN PROBABILITIES
                # -----------------------------------

                aligned_probabilities = np.zeros(
                    (
                        len(test_indices),
                        class_count
                    )
                )

                model_classes = np.asarray(
                    model.classes_
                )

                for local_index, model_class in enumerate(
                    model_classes
                ):

                    global_index = int(
                        model_class
                    )

                    aligned_probabilities[
                        :,
                        global_index
                    ] = probabilities[
                        :,
                        local_index
                    ]

                predictions = (
                    aligned_probabilities.argmax(
                        axis=1
                    )
                )

                for position, row_index in enumerate(
                    test_indices
                ):

                    oof_probabilities[
                        name
                    ][row_index] = (
                        aligned_probabilities[
                            position
                        ]
                    )

                    oof_predictions[
                        name
                    ][row_index] = (
                        predictions[position]
                    )

                    used_rows.add(
                        row_index
                    )

                print(
                    f"{name} completed."
                )

            except Exception as error:

                print(
                    f"{name} failed:"
                )

                print(error)

    # -----------------------------------
    # BUILD OOF DATAFRAME
    # -----------------------------------

    valid_rows = sorted(
        list(used_rows)
    )

    if not valid_rows:

        print(
            "\nNo valid OOF predictions generated."
        )

        return

    oof_data = pd.DataFrame({

        "record_date":
            dataset.loc[
                valid_rows,
                "record_date"
            ].values,

        "actual_target":
            y.loc[
                valid_rows
            ].values
    })

    # -----------------------------------
    # MODEL PREDICTIONS
    # -----------------------------------

    for name in model_names:

        prediction_values = (
            oof_predictions[name][
                valid_rows
            ]
        )

        oof_data[
            f"{name}_prediction"
        ] = prediction_values.astype(
            int
        )

    # -----------------------------------
    # PROBABILITY FEATURES
    # -----------------------------------

    for name in model_names:

        probability_matrix = (
            oof_probabilities[name][
                valid_rows
            ]
        )

        for class_index in range(
            class_count
        ):

            actual_class = (
                index_to_class[
                    class_index
                ]
            )

            oof_data[
                f"{name}_prob_{actual_class}"
            ] = (
                probability_matrix[
                    :,
                    class_index
                ]
            )

    # -----------------------------------
    # SAVE OOF DATA
    # -----------------------------------

    output_file = (
        "ml/oof_predictions.csv"
    )

    oof_data.to_csv(
        output_file,
        index=False
    )

    # -----------------------------------
    # DISPLAY RESULTS
    # -----------------------------------

    print("\n===================================")
    print("OOF PREDICTION SUMMARY")
    print("===================================")

    print(
        f"OOF rows generated: "
        f"{len(oof_data)}"
    )

    print(
        f"OOF columns generated: "
        f"{len(oof_data.columns)}"
    )

    print(
        f"Saved to: "
        f"{output_file}"
    )

    print("\n-----------------------------------")
    print("OOF SAMPLE")
    print("-----------------------------------")

    print(
        oof_data.head(
            10
        ).to_string(
            index=False
        )
    )

    # -----------------------------------
    # MODEL OOF ACCURACY
    # -----------------------------------

    print("\n-----------------------------------")
    print("OOF MODEL PERFORMANCE")
    print("-----------------------------------")

    for name in model_names:

        prediction_column = (
            f"{name}_prediction"
        )

        predictions_for_model = (
            oof_data[
                prediction_column
            ].astype(int)
        )

        actual_values = [
            class_to_index[
                value
            ]
            for value in oof_data[
                "actual_target"
            ]
        ]

        accuracy = np.mean(
            predictions_for_model.values
            ==
            np.array(
                actual_values
            )
        )

        print(
            f"{name}: "
            f"Accuracy={accuracy:.4f}"
        )

    print("\n-----------------------------------")
    print("OOF DATA QUALITY")
    print("-----------------------------------")

    missing_oof = (
        oof_data.isna()
        .sum()
        .sum()
    )

    print(
        f"Missing OOF values: "
        f"{missing_oof}"
    )

    print(
        "Future-target leakage: "
        "PREVENTED by chronological folds"
    )

    print(
        "Same-row training prediction leakage: "
        "PREVENTED for generated OOF rows"
    )

    # -----------------------------------
    # FINAL MESSAGE
    # -----------------------------------

    print("\n===================================")
    print("     OOF PREDICTIONS COMPLETED")
    print("===================================")

    print(
        "\nOOF predictions are now ready "
        "for the stacking stage."
    )

    print(
        "===================================\n"
    )


if __name__ == "__main__":

    run_oof_predictions(
        target_column="col1"
    )