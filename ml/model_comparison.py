import time

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ml.training_data import build_training_dataset
from ml.time_split import time_aware_split


def run_model_comparison(target_column="col1"):

    print("\n===================================")
    print("         MODEL COMPARISON")
    print("===================================")

    print(f"Target column: {target_column}")

    dataset = build_training_dataset(target_column)

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
        if column not in ["target", "record_date"]
    ]

    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]

    y_train = train_data["target"]
    y_test = test_data["target"]

    classes = sorted(y_train.unique())
    class_count = len(classes)

    print("\n-----------------------------------")
    print("DATA")
    print("-----------------------------------")

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Classes: {class_count}")

    if class_count < 2:
        print("At least 2 training classes are required.")
        return

    # -----------------------------------
    # CLASS ENCODING
    # -----------------------------------

    class_to_index = {
        value: index
        for index, value in enumerate(classes)
    }

    y_train_encoded = y_train.map(class_to_index)

    # -----------------------------------
    # MODELS
    # -----------------------------------

    models = {}

    models["Logistic Regression"] = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    models["Decision Tree"] = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
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

    results = []

    # -----------------------------------
    # TRAIN AND EVALUATE
    # -----------------------------------

    print("\n===================================")
    print("TRAINING MODELS")
    print("===================================")

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        start_time = time.perf_counter()

        try:

            if model_name in [
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

            # Convert test values to encoded classes.
            # If a test class was never present in training,
            # log loss cannot be calculated safely.

            y_test_encoded = y_test.map(
                class_to_index
            )

            if y_test_encoded.isna().any():

                loss = None

            else:

                try:

                    loss = log_loss(
                        y_test_encoded,
                        probabilities,
                        labels=list(
                            range(class_count)
                        )
                    )

                except ValueError:

                    loss = None

            results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Balanced Accuracy": balanced_accuracy,
                "Macro F1": macro_f1,
                "Log Loss": loss,
                "Training Time": training_time
            })

            print(
                f"Completed: {model_name}"
            )

        except Exception as error:

            print(
                f"ERROR in {model_name}: "
                f"{error}"
            )

    # -----------------------------------
    # RESULTS
    # -----------------------------------

    print("\n===================================")
    print("MODEL RESULTS")
    print("===================================")

    if not results:

        print("No models completed successfully.")
        return

    print(
        "\n"
        f"{'Model':<22}"
        f"{'Accuracy':>12}"
        f"{'Balanced':>12}"
        f"{'Macro F1':>12}"
        f"{'Log Loss':>12}"
    )

    print("-" * 70)

    for result in results:

        accuracy = (
            f"{result['Accuracy'] * 100:.2f}%"
        )

        balanced = (
            f"{result['Balanced Accuracy'] * 100:.2f}%"
        )

        f1 = (
            f"{result['Macro F1'] * 100:.2f}%"
        )

        if result["Log Loss"] is None:

            loss = "N/A"

        else:

            loss = f"{result['Log Loss']:.4f}"

        print(
            f"{result['Model']:<22}"
            f"{accuracy:>12}"
            f"{balanced:>12}"
            f"{f1:>12}"
            f"{loss:>12}"
        )

    # -----------------------------------
    # TRAINING TIME
    # -----------------------------------

    print("\n-----------------------------------")
    print("TRAINING TIME")
    print("-----------------------------------")

    for result in results:

        print(
            f"{result['Model']:<22}"
            f"{result['Training Time']:.4f} seconds"
        )

    print("\n===================================")
    print("MODEL COMPARISON COMPLETED")
    print("===================================\n")


if __name__ == "__main__":
    run_model_comparison("col1")