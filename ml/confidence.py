import math
import numpy as np


from ml.ensemble_prediction import (
    run_ensemble_prediction
)


# ===================================
# CONFIGURATION
# ===================================

MIN_PROBABILITY = 0.0
MAX_PROBABILITY = 1.0


# ===================================
# VALIDATE PROBABILITIES
# ===================================

def validate_probabilities(
    probabilities
):

    if not probabilities:

        raise ValueError(
            "Probability dictionary "
            "is empty."
        )

    cleaned = {}

    for class_value, probability in (
        probabilities.items()
    ):

        probability = float(
            probability
        )

        if not np.isfinite(
            probability
        ):

            raise ValueError(
                f"Invalid probability "
                f"for class {class_value}"
            )

        if (
            probability
            < MIN_PROBABILITY
            or probability
            > MAX_PROBABILITY
        ):

            raise ValueError(
                f"Probability outside "
                f"valid range for "
                f"class {class_value}"
            )

        cleaned[class_value] = (
            probability
        )

    total = sum(
        cleaned.values()
    )

    if total <= 0:

        raise ValueError(
            "Probability total "
            "must be greater than zero."
        )

    # Normalize defensively.

    normalized = {
        class_value:
        probability / total
        for (
            class_value,
            probability
        ) in cleaned.items()
    }

    return normalized


# ===================================
# PROBABILITY MARGIN
# ===================================

def calculate_probability_margin(
    probabilities
):

    sorted_probabilities = sorted(
        probabilities.values(),
        reverse=True
    )

    if len(
        sorted_probabilities
    ) == 1:

        return 1.0

    first_probability = (
        sorted_probabilities[0]
    )

    second_probability = (
        sorted_probabilities[1]
    )

    margin = (
        first_probability
        - second_probability
    )

    return float(
        margin
    )


# ===================================
# ENTROPY
# ===================================

def calculate_entropy(
    probabilities
):

    values = [
        probability
        for probability
        in probabilities.values()
        if probability > 0
    ]

    if not values:

        return 0.0

    entropy = 0.0

    for probability in values:

        entropy -= (
            probability
            * math.log(
                probability
            )
        )

    return float(
        entropy
    )


# ===================================
# NORMALIZED ENTROPY
# ===================================

def calculate_normalized_entropy(
    probabilities
):

    class_count = len(
        probabilities
    )

    if class_count <= 1:

        return 0.0

    entropy = (
        calculate_entropy(
            probabilities
        )
    )

    maximum_entropy = math.log(
        class_count
    )

    if maximum_entropy == 0:

        return 0.0

    normalized_entropy = (
        entropy
        / maximum_entropy
    )

    return float(
        normalized_entropy
    )


# ===================================
# MODEL AGREEMENT SCORE
# ===================================

def calculate_agreement_score(
    agreement
):

    total_models = int(
        agreement.get(
            "total_models",
            0
        )
    )

    agreement_count = int(
        agreement.get(
            "agreement_count",
            0
        )
    )

    if total_models <= 0:

        return 0.0

    score = (
        agreement_count
        / total_models
    )

    return float(
        score
    )


# ===================================
# CONFIDENCE SCORE
# ===================================

def calculate_confidence_score(
    top_probability,
    agreement_score,
    probability_margin,
    normalized_entropy
):

    # --------------------------------
    # Probability component
    # --------------------------------

    probability_component = (
        top_probability
    )

    # --------------------------------
    # Agreement component
    # --------------------------------

    agreement_component = (
        agreement_score
    )

    # --------------------------------
    # Margin component
    # --------------------------------

    # Scale margin from 0-1.
    # A 25 percentage-point gap
    # is treated as a strong margin.

    margin_component = min(
        probability_margin / 0.25,
        1.0
    )

    # --------------------------------
    # Entropy component
    # --------------------------------

    entropy_component = (
        1.0
        - normalized_entropy
    )

    # --------------------------------
    # Combined score
    # --------------------------------

    confidence_score = (
        0.35
        * probability_component

        + 0.30
        * agreement_component

        + 0.20
        * margin_component

        + 0.15
        * entropy_component
    )

    confidence_score = max(
        0.0,
        min(
            confidence_score,
            1.0
        )
    )

    return float(
        confidence_score
    )


# ===================================
# CONFIDENCE CATEGORY
# ===================================

def classify_confidence(
    confidence_score
):

    if confidence_score >= 0.70:

        return "HIGH"

    if confidence_score >= 0.45:

        return "MODERATE"

    return "LOW"


# ===================================
# COMPLETE CONFIDENCE ANALYSIS
# ===================================

def analyze_confidence(
    ensemble_result
):

    if not ensemble_result:

        raise ValueError(
            "Ensemble result is empty."
        )

    probabilities = (
        ensemble_result[
            "probabilities"
        ]
    )

    probabilities = (
        validate_probabilities(
            probabilities
        )
    )

    # --------------------------------
    # Top prediction
    # --------------------------------

    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_class = (
        sorted_probabilities[0][0]
    )

    top_probability = (
        sorted_probabilities[0][1]
    )

    # --------------------------------
    # Probability margin
    # --------------------------------

    probability_margin = (
        calculate_probability_margin(
            probabilities
        )
    )

    # --------------------------------
    # Entropy
    # --------------------------------

    entropy = (
        calculate_entropy(
            probabilities
        )
    )

    normalized_entropy = (
        calculate_normalized_entropy(
            probabilities
        )
    )

    # --------------------------------
    # Model agreement
    # --------------------------------

    agreement = (
        ensemble_result[
            "agreement"
        ]
    )

    agreement_score = (
        calculate_agreement_score(
            agreement
        )
    )

    # --------------------------------
    # Final confidence
    # --------------------------------

    confidence_score = (
        calculate_confidence_score(
            top_probability,
            agreement_score,
            probability_margin,
            normalized_entropy
        )
    )

    confidence_category = (
        classify_confidence(
            confidence_score
        )
    )

    return {
        "prediction": top_class,
        "top_probability": top_probability,
        "probability_margin": probability_margin,
        "entropy": entropy,
        "normalized_entropy":
            normalized_entropy,
        "agreement_score":
            agreement_score,
        "confidence_score":
            confidence_score,
        "confidence_category":
            confidence_category
    }


# ===================================
# RUN CONFIDENCE ENGINE
# ===================================

def run_confidence_engine():

    print("\n===================================")
    print("       PROBABILITY & CONFIDENCE")
    print("===================================")

    # =================================
    # RUN ENSEMBLE
    # =================================

    ensemble_result = (
        run_ensemble_prediction()
    )

    if ensemble_result is None:

        print(
            "Ensemble prediction "
            "failed."
        )

        return None

    # =================================
    # ANALYZE CONFIDENCE
    # =================================

    print("\n-----------------------------------")
    print("ANALYZING CONFIDENCE")
    print("-----------------------------------")

    confidence = (
        analyze_confidence(
            ensemble_result
        )
    )

    # =================================
    # DISPLAY RESULTS
    # =================================

    print("\n===================================")
    print("       CONFIDENCE RESULT")
    print("===================================")

    print(
        f"Prediction: "
        f"{confidence['prediction']}"
    )

    print(
        f"Top probability: "
        f"{confidence['top_probability'] * 100:.2f}%"
    )

    print(
        f"Probability margin: "
        f"{confidence['probability_margin'] * 100:.2f}%"
    )

    print(
        f"Entropy: "
        f"{confidence['entropy']:.6f}"
    )

    print(
        f"Normalized entropy: "
        f"{confidence['normalized_entropy']:.6f}"
    )

    print(
        f"Model agreement: "
        f"{confidence['agreement_score'] * 100:.2f}%"
    )

    print(
        f"Confidence score: "
        f"{confidence['confidence_score'] * 100:.2f}%"
    )

    print(
        f"Confidence category: "
        f"{confidence['confidence_category']}"
    )

    # =================================
    # VALIDATION
    # =================================

    print("\n-----------------------------------")
    print("CONFIDENCE VALIDATION")
    print("-----------------------------------")

    validation_passed = True

    if not (
        0.0
        <= confidence[
            "top_probability"
        ]
        <= 1.0
    ):

        print(
            "Top probability validation: FAILED"
        )

        validation_passed = False

    else:

        print(
            "Top probability validation: PASSED"
        )

    if not (
        0.0
        <= confidence[
            "probability_margin"
        ]
        <= 1.0
    ):

        print(
            "Probability margin validation: FAILED"
        )

        validation_passed = False

    else:

        print(
            "Probability margin validation: PASSED"
        )

    if not (
        0.0
        <= confidence[
            "agreement_score"
        ]
        <= 1.0
    ):

        print(
            "Agreement validation: FAILED"
        )

        validation_passed = False

    else:

        print(
            "Agreement validation: PASSED"
        )

    if not (
        0.0
        <= confidence[
            "confidence_score"
        ]
        <= 1.0
    ):

        print(
            "Confidence score validation: FAILED"
        )

        validation_passed = False

    else:

        print(
            "Confidence score validation: PASSED"
        )

    valid_categories = {
        "LOW",
        "MODERATE",
        "HIGH"
    }

    if (
        confidence[
            "confidence_category"
        ]
        not in valid_categories
    ):

        print(
            "Confidence category validation: FAILED"
        )

        validation_passed = False

    else:

        print(
            "Confidence category validation: PASSED"
        )

    print(
        f"\nOverall validation: "
        f"{'PASSED' if validation_passed else 'FAILED'}"
    )

    print("\n===================================")

    if validation_passed:

        print(
            "9.7 PROBABILITY & CONFIDENCE "
            "COMPLETED"
        )

    else:

        print(
            "9.7 PROBABILITY & CONFIDENCE "
            "FAILED"
        )

    print("===================================\n")

    return confidence


# ===================================
# DIRECT EXECUTION
# ===================================

if __name__ == "__main__":

    run_confidence_engine()