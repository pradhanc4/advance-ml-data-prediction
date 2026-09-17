import numpy as np


from ml.model_loader import load_all_models
from ml.prediction_engine import prepare_prediction_features
from ml.individual_predictions import (
    predict_with_model
)


# ===================================
# CONFIGURATION
# ===================================

EXPECTED_FEATURE_COUNT = 108

TARGET_COLUMN = "col1"


# ===================================
# CREATE ENSEMBLE PROBABILITIES
# ===================================

def calculate_ensemble_probabilities(
    results
):

    if not results:

        raise ValueError(
            "No model prediction results "
            "available."
        )

    # ===================================
    # COLLECT ALL CLASSES
    # ===================================

    all_classes = set()

    for result in results:

        probabilities = (
            result["probabilities"]
        )

        all_classes.update(
            probabilities.keys()
        )

    if not all_classes:

        raise ValueError(
            "No probability information "
            "available from models."
        )

    all_classes = sorted(
        all_classes
    )

    # ===================================
    # BUILD PROBABILITY MATRIX
    # ===================================

    probability_matrix = []

    model_names = []

    for result in results:

        model_name = (
            result["model_name"]
        )

        probabilities = (
            result["probabilities"]
        )

        model_names.append(
            model_name
        )

        row = []

        for class_value in all_classes:

            probability = (
                probabilities.get(
                    class_value,
                    0.0
                )
            )

            row.append(
                float(probability)
            )

        probability_matrix.append(
            row
        )

    probability_matrix = np.array(
        probability_matrix,
        dtype=float
    )

    # ===================================
    # AVERAGE MODEL PROBABILITIES
    # ===================================

    ensemble_probabilities = (
        probability_matrix.mean(
            axis=0
        )
    )

    # ===================================
    # NORMALIZE
    # ===================================

    probability_sum = (
        ensemble_probabilities.sum()
    )

    if probability_sum <= 0:

        raise ValueError(
            "Ensemble probability sum "
            "is zero."
        )

    ensemble_probabilities = (
        ensemble_probabilities
        / probability_sum
    )

    # ===================================
    # CREATE RESULT DICTIONARY
    # ===================================

    ensemble_dict = {}

    for index, class_value in enumerate(
        all_classes
    ):

        ensemble_dict[
            class_value
        ] = float(
            ensemble_probabilities[index]
        )

    return (
        ensemble_dict,
        probability_matrix,
        model_names
    )


# ===================================
# GET ENSEMBLE PREDICTION
# ===================================

def get_ensemble_prediction(
    ensemble_probabilities
):

    if not ensemble_probabilities:

        raise ValueError(
            "Ensemble probabilities "
            "are empty."
        )

    predicted_class = max(
        ensemble_probabilities,
        key=ensemble_probabilities.get
    )

    predicted_probability = (
        ensemble_probabilities[
            predicted_class
        ]
    )

    return (
        predicted_class,
        predicted_probability
    )


# ===================================
# MODEL AGREEMENT
# ===================================

def calculate_model_agreement(
    results
):

    predictions = []

    for result in results:

        prediction = (
            result["prediction"]
        )

        # Normalize NumPy values.

        if isinstance(
            prediction,
            np.ndarray
        ):

            prediction = (
                prediction
                .flatten()[0]
            )

        if isinstance(
            prediction,
            np.generic
        ):

            prediction = (
                prediction.item()
            )

        predictions.append(
            prediction
        )

    if not predictions:

        return {
            "total_models": 0,
            "unique_predictions": 0,
            "agreement_count": 0,
            "agreement_ratio": 0.0
        }

    prediction_counts = {}

    for prediction in predictions:

        prediction_counts[
            prediction
        ] = (
            prediction_counts.get(
                prediction,
                0
            ) + 1
        )

    max_agreement = max(
        prediction_counts.values()
    )

    total_models = len(
        predictions
    )

    agreement_ratio = (
        max_agreement
        / total_models
    )

    return {
        "total_models": total_models,
        "unique_predictions": len(
            prediction_counts
        ),
        "agreement_count": max_agreement,
        "agreement_ratio": agreement_ratio,
        "prediction_counts": prediction_counts
    }


# ===================================
# RUN ENSEMBLE PREDICTION
# ===================================

def run_ensemble_prediction():

    print("\n===================================")
    print("         ENSEMBLE PREDICTION")
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

    if len(models) < 4:

        print(
            "ERROR: All four models "
            "are required."
        )

        return None

    print(
        f"\nModels loaded: "
        f"{len(models)}"
    )

    # ===================================
    # PREPARE FEATURES
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
    # FEATURE VALIDATION
    # ===================================

    if (
        prediction_features.shape[1]
        != EXPECTED_FEATURE_COUNT
    ):

        print(
            "ERROR: Prediction feature "
            "count mismatch."
        )

        print(
            f"Expected: "
            f"{EXPECTED_FEATURE_COUNT}"
        )

        print(
            f"Received: "
            f"{prediction_features.shape[1]}"
        )

        return None

    # ===================================
    # RUN INDIVIDUAL MODELS
    # ===================================

    print("\n-----------------------------------")
    print("RUNNING INDIVIDUAL MODELS")
    print("-----------------------------------")

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
                f"{model_name} failed:"
            )

            print(
                f"Error: {error}"
            )

    # ===================================
    # CHECK MODEL RESULTS
    # ===================================

    if len(results) < 4:

        print(
            "ERROR: Not all models "
            "produced predictions."
        )

        return None

    # ===================================
    # CALCULATE ENSEMBLE
    # ===================================

    print("\n-----------------------------------")
    print("CALCULATING ENSEMBLE")
    print("-----------------------------------")

    (
        ensemble_probabilities,
        probability_matrix,
        model_names
    ) = calculate_ensemble_probabilities(
        results
    )

    (
        ensemble_prediction,
        ensemble_probability
    ) = get_ensemble_prediction(
        ensemble_probabilities
    )

    # ===================================
    # MODEL AGREEMENT
    # ===================================

    agreement = (
        calculate_model_agreement(
            results
        )
    )

    # ===================================
    # DISPLAY ENSEMBLE
    # ===================================

    print("\n===================================")
    print("       ENSEMBLE RESULT")
    print("===================================")

    print(
        f"Ensemble {TARGET_COLUMN}: "
        f"{ensemble_prediction}"
    )

    print(
        f"Ensemble probability: "
        f"{ensemble_probability * 100:.2f}%"
    )

    # ===================================
    # ENSEMBLE PROBABILITIES
    # ===================================

    print("\n-----------------------------------")
    print("ENSEMBLE PROBABILITIES")
    print("-----------------------------------")

    sorted_ensemble = sorted(
        ensemble_probabilities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for (
        class_value,
        probability
    ) in sorted_ensemble:

        print(
            f"  {class_value}: "
            f"{probability * 100:.2f}%"
        )

    # ===================================
    # TOP 5
    # ===================================

    print("\n-----------------------------------")
    print("TOP 5 ENSEMBLE PREDICTIONS")
    print("-----------------------------------")

    for (
        rank,
        (
            class_value,
            probability
        )
    ) in enumerate(
        sorted_ensemble[:5],
        start=1
    ):

        print(
            f"{rank}. "
            f"{class_value} -> "
            f"{probability * 100:.2f}%"
        )

    # ===================================
    # MODEL AGREEMENT
    # ===================================

    print("\n-----------------------------------")
    print("MODEL AGREEMENT")
    print("-----------------------------------")

    print(
        f"Total models: "
        f"{agreement['total_models']}"
    )

    print(
        f"Unique predictions: "
        f"{agreement['unique_predictions']}"
    )

    print(
        f"Highest agreement: "
        f"{agreement['agreement_count']}/"
        f"{agreement['total_models']}"
    )

    print(
        f"Agreement ratio: "
        f"{agreement['agreement_ratio'] * 100:.2f}%"
    )

    print("\nVote distribution:")

    for (
        prediction,
        count
    ) in sorted(
        agreement["prediction_counts"].items()
    ):

        print(
            f"  {prediction}: "
            f"{count} model(s)"
        )

    # ===================================
    # PROBABILITY VALIDATION
    # ===================================

    ensemble_sum = sum(
        ensemble_probabilities.values()
    )

    print("\n-----------------------------------")
    print("ENSEMBLE VALIDATION")
    print("-----------------------------------")

    print(
        f"Probability sum: "
        f"{ensemble_sum:.6f}"
    )

    if np.isclose(
        ensemble_sum,
        1.0,
        atol=0.000001
    ):

        print(
            "Probability validation: PASSED"
        )

    else:

        print(
            "Probability validation: FAILED"
        )

    print(
        f"Prediction feature count: "
        f"{prediction_features.shape[1]}"
    )

    if (
        prediction_features.shape[1]
        == EXPECTED_FEATURE_COUNT
    ):

        print(
            "Feature schema validation: PASSED"
        )

    else:

        print(
            "Feature schema validation: FAILED"
        )

    # ===================================
    # FINAL STATUS
    # ===================================

    print("\n===================================")
    print(
        "9.6 ENSEMBLE PREDICTION "
        "COMPLETED"
    )
    print("===================================\n")

    return {
        "prediction": ensemble_prediction,
        "probability": ensemble_probability,
        "probabilities": ensemble_probabilities,
        "individual_results": results,
        "agreement": agreement
    }


# ===================================
# DIRECT EXECUTION
# ===================================

if __name__ == "__main__":

    run_ensemble_prediction()