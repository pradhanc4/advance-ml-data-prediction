import numpy as np
import pandas as pd


from ml.model_loader import load_all_models
from ml.prediction_engine import prepare_prediction_features


# ===================================
# CONFIGURATION
# ===================================

TARGET_COLUMN = "col1"

EXPECTED_FEATURE_COUNT = 108


# ===================================
# NORMALIZE MODEL PREDICTION
# ===================================

def normalize_prediction(prediction):

    """
    Convert different model prediction
    formats into one scalar Python value.

    Examples:

        2
        np.int64(2)
        np.array([2])

    become:

        2
    """

    if isinstance(
        prediction,
        np.ndarray
    ):

        if prediction.size == 0:

            raise ValueError(
                "Model returned an empty prediction."
            )

        prediction = (
            prediction
            .flatten()[0]
        )

    elif isinstance(
        prediction,
        (list, tuple)
    ):

        if len(prediction) == 0:

            raise ValueError(
                "Model returned an empty prediction."
            )

        prediction = prediction[0]

    # Convert NumPy scalar
    # into normal Python value.

    if isinstance(
        prediction,
        np.generic
    ):

        prediction = prediction.item()

    return prediction


# ===================================
# PREDICT WITH ONE MODEL
# ===================================

def predict_with_model(
    model_name,
    bundle,
    prediction_features
):

    model = bundle["model"]

    feature_names = list(
        bundle["feature_names"]
    )

    print("\n-----------------------------------")
    print(model_name)
    print("-----------------------------------")

    # ===================================
    # FEATURE SCHEMA CHECK
    # ===================================

    missing_features = [
        feature
        for feature in feature_names
        if feature not in prediction_features.columns
    ]

    if missing_features:

        raise ValueError(
            f"{model_name}: Missing "
            f"features: {missing_features[:10]}"
        )

    # ===================================
    # EXACT FEATURE ORDER
    # ===================================

    X = prediction_features[
        feature_names
    ].copy()

    # ===================================
    # FEATURE COUNT
    # ===================================

    if X.shape[1] != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            f"{model_name}: Expected "
            f"{EXPECTED_FEATURE_COUNT} features, "
            f"got {X.shape[1]}"
        )

    # ===================================
    # NUMERIC VALIDATION
    # ===================================

    X = X.select_dtypes(
        include=["number"]
    )

    if X.shape[1] != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            f"{model_name}: Numeric feature "
            f"count mismatch."
        )

    # ===================================
    # NULL VALIDATION
    # ===================================

    null_count = (
        X.isna()
        .sum()
        .sum()
    )

    if null_count > 0:

        raise ValueError(
            f"{model_name}: Prediction input "
            f"contains {null_count} NULL values."
        )

    # ===================================
    # INFINITE VALUE VALIDATION
    # ===================================

    infinite_count = np.isinf(
        X.to_numpy()
    ).sum()

    if infinite_count > 0:

        raise ValueError(
            f"{model_name}: Prediction input "
            f"contains infinite values."
        )

    # ===================================
    # MODEL PREDICTION
    # ===================================

    try:

        prediction = model.predict(X)

    except Exception as error:

        raise RuntimeError(
            f"{model_name}: Prediction failed: "
            f"{error}"
        )

    # ===================================
    # NORMALIZE PREDICTION
    # ===================================

    predicted_value = (
        normalize_prediction(
            prediction
        )
    )

    print(
        f"Predicted {TARGET_COLUMN}: "
        f"{predicted_value}"
    )

    # ===================================
    # PROBABILITY PREDICTION
    # ===================================

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        try:

            probabilities = (
                model.predict_proba(X)
            )

        except Exception as error:

            print(
                "Probability prediction "
                f"unavailable: {error}"
            )

            probabilities = None

    # ===================================
    # PROBABILITY DICTIONARY
    # ===================================

    probability_dict = {}

    if probabilities is not None:

        probability_array = np.asarray(
            probabilities
        )

        # Make sure the result is
        # two-dimensional.

        if probability_array.ndim == 1:

            probability_row = (
                probability_array
            )

        else:

            probability_row = (
                probability_array[0]
            )

        classes = getattr(
            model,
            "classes_",
            None
        )

        # -----------------------------------
        # MODEL CLASSES AVAILABLE
        # -----------------------------------

        if classes is not None:

            classes_array = np.asarray(
                classes
            ).flatten()

            if len(classes_array) != len(
                probability_row
            ):

                print(
                    "WARNING: Number of classes "
                    "does not match probability "
                    "columns."
                )

            else:

                for (
                    index,
                    class_value
                ) in enumerate(
                    classes_array
                ):

                    # Convert NumPy scalar
                    # to normal Python value.

                    if isinstance(
                        class_value,
                        np.generic
                    ):

                        class_value = (
                            class_value.item()
                        )

                    probability_dict[
                        class_value
                    ] = float(
                        probability_row[index]
                    )

        else:

            print(
                "Model classes metadata "
                "not available."
            )

    # ===================================
    # PROBABILITY VALIDATION
    # ===================================

    if probability_dict:

        probability_sum = sum(
            probability_dict.values()
        )

        print(
            f"Probability sum: "
            f"{probability_sum:.6f}"
        )

        if not np.isclose(
            probability_sum,
            1.0,
            atol=0.01
        ):

            print(
                "WARNING: Probability "
                "sum is not approximately 1."
            )

    # ===================================
    # TOP PROBABILITIES
    # ===================================

    if probability_dict:

        sorted_probabilities = sorted(
            probability_dict.items(),
            key=lambda item: item[1],
            reverse=True
        )

        print("\nTop probabilities:")

        for (
            class_value,
            probability
        ) in sorted_probabilities[:5]:

            print(
                f"  {class_value}: "
                f"{probability * 100:.2f}%"
            )

    else:

        print(
            "Probability information: "
            "NOT AVAILABLE"
        )

    # ===================================
    # RETURN RESULT
    # ===================================

    return {
        "model_name": model_name,
        "prediction": predicted_value,
        "probabilities": probability_dict
    }


# ===================================
# RUN INDIVIDUAL PREDICTIONS
# ===================================

def run_individual_predictions():

    print("\n===================================")
    print("     INDIVIDUAL MODEL PREDICTIONS")
    print("===================================")

    # ===================================
    # LOAD MODELS
    # ===================================

    models = load_all_models()

    if not models:

        print(
            "No models available."
        )

        return None

    # ===================================
    # REQUIRE ALL FOUR MODELS
    # ===================================

    if len(models) < 4:

        print(
            "\nWARNING:"
        )

        print(
            "Not all four models are "
            "available."
        )

        print(
            "Prediction test cannot "
            "continue."
        )

        return None

    print(
        f"\nModels available: "
        f"{len(models)}"
    )

    # ===================================
    # CREATE PREDICTION FEATURES
    # ===================================

    print("\n-----------------------------------")
    print("PREPARING PREDICTION FEATURES")
    print("-----------------------------------")

    prediction_features = (
        prepare_prediction_features()
    )

    if prediction_features is None:

        print(
            "Prediction features could "
            "not be created."
        )

        return None

    # ===================================
    # FEATURE INFORMATION
    # ===================================

    print(
        f"Prediction rows: "
        f"{len(prediction_features)}"
    )

    print(
        f"Prediction features: "
        f"{prediction_features.shape[1]}"
    )

    # ===================================
    # RUN ALL MODELS
    # ===================================

    results = []

    for model_name, bundle in models.items():

        try:

            result = predict_with_model(
                model_name,
                bundle,
                prediction_features
            )

            results.append(
                result
            )

        except Exception as error:

            print(
                f"\n{model_name} prediction "
                f"FAILED:"
            )

            print(
                f"Error: {error}"
            )

    # ===================================
    # CHECK RESULTS
    # ===================================

    if not results:

        print(
            "\nNo model predictions "
            "were generated."
        )

        return None

    # ===================================
    # INDIVIDUAL PREDICTION SUMMARY
    # ===================================

    print("\n===================================")
    print("       INDIVIDUAL PREDICTIONS")
    print("===================================")

    for result in results:

        print(
            f"{result['model_name']}: "
            f"{result['prediction']}"
        )

    # ===================================
    # PREDICTION AGREEMENT
    # ===================================

    predictions = [
        result["prediction"]
        for result in results
    ]

    # Make absolutely sure all
    # predictions are hashable.

    normalized_predictions = []

    for prediction in predictions:

        prediction = normalize_prediction(
            prediction
        )

        normalized_predictions.append(
            prediction
        )

    unique_predictions = set(
        normalized_predictions
    )

    print("\n-----------------------------------")
    print("MODEL AGREEMENT")
    print("-----------------------------------")

    print(
        f"Models producing predictions: "
        f"{len(normalized_predictions)}"
    )

    print(
        f"Unique predictions: "
        f"{len(unique_predictions)}"
    )

    if len(unique_predictions) == 1:

        print(
            "All models agree: YES"
        )

    else:

        print(
            "All models agree: NO"
        )

    # ===================================
    # PREDICTION DISTRIBUTION
    # ===================================

    prediction_counts = {}

    for prediction in normalized_predictions:

        prediction_counts[prediction] = (
            prediction_counts.get(
                prediction,
                0
            ) + 1
        )

    print("\n-----------------------------------")
    print("PREDICTION VOTE DISTRIBUTION")
    print("-----------------------------------")

    for (
        prediction,
        count
    ) in sorted(
        prediction_counts.items()
    ):

        print(
            f"{prediction}: "
            f"{count} model(s)"
        )

    # ===================================
    # FINAL STATUS
    # ===================================

    print("\n===================================")
    print(
        "9.5 INDIVIDUAL MODEL "
        "PREDICTIONS COMPLETED"
    )
    print("===================================\n")

    return results


# ===================================
# DIRECT EXECUTION
# ===================================

if __name__ == "__main__":

    run_individual_predictions()