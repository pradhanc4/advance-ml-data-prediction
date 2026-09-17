import json
from datetime import datetime

import numpy as np

from ml.confidence import (
    run_confidence_engine
)


# ===================================
# CONFIGURATION
# ===================================

TARGET_COLUMN = "col1"


# ===================================
# NORMALIZE VALUE
# ===================================

def normalize_value(value):

    if isinstance(
        value,
        np.ndarray
    ):

        if value.size == 0:
            return None

        value = value.flatten()[0]

    if isinstance(
        value,
        np.generic
    ):

        value = value.item()

    return value


# ===================================
# FORMAT PERCENTAGE
# ===================================

def format_percentage(
    value
):

    return round(
        float(value) * 100,
        2
    )


# ===================================
# BUILD MODEL RESULTS
# ===================================

def format_model_results(
    individual_results
):

    formatted_models = []

    for result in individual_results:

        model_name = (
            result["model_name"]
        )

        prediction = normalize_value(
            result["prediction"]
        )

        probabilities = (
            result.get(
                "probabilities",
                {}
            )
        )

        formatted_probabilities = {}

        for (
            class_value,
            probability
        ) in probabilities.items():

            class_value = normalize_value(
                class_value
            )

            formatted_probabilities[
                str(class_value)
            ] = format_percentage(
                probability
            )

        formatted_models.append(
            {
                "model_name": model_name,
                "prediction": prediction,
                "probabilities":
                    formatted_probabilities
            }
        )

    return formatted_models


# ===================================
# BUILD TOP PREDICTIONS
# ===================================

def format_top_predictions(
    probabilities,
    limit=5
):

    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_predictions = []

    for rank, (
        class_value,
        probability
    ) in enumerate(
        sorted_probabilities[:limit],
        start=1
    ):

        class_value = normalize_value(
            class_value
        )

        top_predictions.append(
            {
                "rank": rank,
                "value": class_value,
                "probability":
                    format_percentage(
                        probability
                    )
            }
        )

    return top_predictions


# ===================================
# BUILD PREDICTION RESULT
# ===================================

def build_prediction_result(
    confidence_result,
    ensemble_result
):

    if not confidence_result:

        raise ValueError(
            "Confidence result is empty."
        )

    if not ensemble_result:

        raise ValueError(
            "Ensemble result is empty."
        )

    prediction = normalize_value(
        confidence_result[
            "prediction"
        ]
    )

    probabilities = (
        ensemble_result[
            "probabilities"
        ]
    )

    agreement = (
        ensemble_result[
            "agreement"
        ]
    )

    individual_results = (
        ensemble_result[
            "individual_results"
        ]
    )

    # ===================================
    # BUILD RESULT
    # ===================================

    result = {

        "result_version":
            "prediction_v1",

        "generated_at":
            datetime.now().isoformat(),

        "target_column":
            TARGET_COLUMN,

        "prediction":
            prediction,

        "ensemble": {

            "probability":
                format_percentage(
                    confidence_result[
                        "top_probability"
                    ]
                ),

            "probabilities":
                {
                    str(
                        normalize_value(
                            class_value
                        )
                    ):
                    format_percentage(
                        probability
                    )
                    for (
                        class_value,
                        probability
                    )
                    in probabilities.items()
                },

            "top_predictions":
                format_top_predictions(
                    probabilities
                )
        },

        "confidence": {

            "score":
                format_percentage(
                    confidence_result[
                        "confidence_score"
                    ]
                ),

            "category":
                confidence_result[
                    "confidence_category"
                ],

            "probability_margin":
                format_percentage(
                    confidence_result[
                        "probability_margin"
                    ]
                ),

            "entropy":
                round(
                    float(
                        confidence_result[
                            "entropy"
                        ]
                    ),
                    6
                ),

            "normalized_entropy":
                round(
                    float(
                        confidence_result[
                            "normalized_entropy"
                        ]
                    ),
                    6
                )
        },

        "model_agreement": {

            "total_models":
                agreement[
                    "total_models"
                ],

            "agreement_count":
                agreement[
                    "agreement_count"
                ],

            "agreement_ratio":
                format_percentage(
                    agreement[
                        "agreement_ratio"
                    ]
                ),

            "unique_predictions":
                agreement[
                    "unique_predictions"
                ],

            "vote_distribution":
                {
                    str(
                        normalize_value(
                            prediction_value
                        )
                    ):
                    count

                    for (
                        prediction_value,
                        count
                    )
                    in agreement[
                        "prediction_counts"
                    ].items()
                }
        },

        "models":
            format_model_results(
                individual_results
            )
    }

    return result


# ===================================
# VALIDATE RESULT
# ===================================

def validate_prediction_result(
    result
):

    required_keys = [

        "result_version",
        "generated_at",
        "target_column",
        "prediction",
        "ensemble",
        "confidence",
        "model_agreement",
        "models"

    ]

    missing_keys = [

        key
        for key in required_keys
        if key not in result

    ]

    if missing_keys:

        print(
            "Missing result fields:"
        )

        print(
            missing_keys
        )

        return False

    # ---------------------------------
    # Ensemble validation
    # ---------------------------------

    ensemble_probability = (
        result[
            "ensemble"
        ][
            "probability"
        ]
    )

    if not (
        0 <= ensemble_probability <= 100
    ):

        print(
            "Ensemble probability "
            "validation: FAILED"
        )

        return False

    # ---------------------------------
    # Confidence validation
    # ---------------------------------

    confidence_score = (
        result[
            "confidence"
        ][
            "score"
        ]
    )

    if not (
        0 <= confidence_score <= 100
    ):

        print(
            "Confidence score "
            "validation: FAILED"
        )

        return False

    # ---------------------------------
    # Model validation
    # ---------------------------------

    models = result[
        "models"
    ]

    if len(models) != 4:

        print(
            "Model count validation: FAILED"
        )

        print(
            f"Expected: 4"
        )

        print(
            f"Received: {len(models)}"
        )

        return False

    # ---------------------------------
    # Probability validation
    # ---------------------------------

    probabilities = result[
        "ensemble"
    ][
        "probabilities"
    ]

    probability_sum = sum(
        probabilities.values()
    )

    if not np.isclose(
        probability_sum,
        100.0,
        atol=0.01
    ):

        print(
            "Formatted probability "
            "sum validation: FAILED"
        )

        print(
            f"Sum: {probability_sum:.2f}%"
        )

        return False

    return True


# ===================================
# PRINT RESULT
# ===================================

def print_prediction_result(
    result
):

    print("\n===================================")
    print("       FORMATTED PREDICTION")
    print("===================================")

    print(
        f"Result version: "
        f"{result['result_version']}"
    )

    print(
        f"Generated at: "
        f"{result['generated_at']}"
    )

    print(
        f"Target: "
        f"{result['target_column']}"
    )

    print(
        f"Prediction: "
        f"{result['prediction']}"
    )

    # ---------------------------------
    # Ensemble
    # ---------------------------------

    ensemble = result[
        "ensemble"
    ]

    print("\n-----------------------------------")
    print("ENSEMBLE")
    print("-----------------------------------")

    print(
        f"Probability: "
        f"{ensemble['probability']:.2f}%"
    )

    print("\nTop predictions:")

    for item in (
        ensemble[
            "top_predictions"
        ]
    ):

        print(
            f"{item['rank']}. "
            f"{item['value']} -> "
            f"{item['probability']:.2f}%"
        )

    # ---------------------------------
    # Confidence
    # ---------------------------------

    confidence = result[
        "confidence"
    ]

    print("\n-----------------------------------")
    print("CONFIDENCE")
    print("-----------------------------------")

    print(
        f"Score: "
        f"{confidence['score']:.2f}%"
    )

    print(
        f"Category: "
        f"{confidence['category']}"
    )

    print(
        f"Probability margin: "
        f"{confidence['probability_margin']:.2f}%"
    )

    print(
        f"Entropy: "
        f"{confidence['entropy']:.6f}"
    )

    print(
        f"Normalized entropy: "
        f"{confidence['normalized_entropy']:.6f}"
    )

    # ---------------------------------
    # Agreement
    # ---------------------------------

    agreement = result[
        "model_agreement"
    ]

    print("\n-----------------------------------")
    print("MODEL AGREEMENT")
    print("-----------------------------------")

    print(
        f"Models: "
        f"{agreement['total_models']}"
    )

    print(
        f"Agreement: "
        f"{agreement['agreement_count']}/"
        f"{agreement['total_models']}"
    )

    print(
        f"Agreement ratio: "
        f"{agreement['agreement_ratio']:.2f}%"
    )

    print(
        f"Unique predictions: "
        f"{agreement['unique_predictions']}"
    )

    print("\nVote distribution:")

    for (
        prediction,
        count
    ) in agreement[
        "vote_distribution"
    ].items():

        print(
            f"  {prediction}: "
            f"{count} model(s)"
        )

    # ---------------------------------
    # Individual models
    # ---------------------------------

    print("\n-----------------------------------")
    print("INDIVIDUAL MODELS")
    print("-----------------------------------")

    for model in result[
        "models"
    ]:

        print(
            f"{model['model_name']}: "
            f"{model['prediction']}"
        )

    print("\n===================================")


# ===================================
# RUN RESULT FORMATTER
# ===================================

def run_prediction_result_formatter():

    print("\n===================================")
    print("     PREDICTION RESULT FORMATTER")
    print("===================================")

    # =================================
    # RUN CONFIDENCE ENGINE
    # =================================

    confidence_result = (
        run_confidence_engine()
    )

    if confidence_result is None:

        print(
            "Confidence engine failed."
        )

        return None

    # =================================
    # GET ENSEMBLE RESULT
    # =================================
    #
    # The confidence engine internally
    # runs the ensemble. To keep this
    # stage reliable and independent,
    # run it again and capture the
    # complete ensemble object.
    #

    from ml.ensemble_prediction import (
        run_ensemble_prediction
    )

    ensemble_result = (
        run_ensemble_prediction()
    )

    if ensemble_result is None:

        print(
            "Ensemble result unavailable."
        )

        return None

    # =================================
    # BUILD RESULT
    # =================================

    result = build_prediction_result(
        confidence_result,
        ensemble_result
    )

    # =================================
    # VALIDATE
    # =================================

    print("\n-----------------------------------")
    print("RESULT VALIDATION")
    print("-----------------------------------")

    validation_passed = (
        validate_prediction_result(
            result
        )
    )

    print(
        "Result structure validation: "
        f"{'PASSED' if validation_passed else 'FAILED'}"
    )

    if not validation_passed:

        print(
            "\n9.8 RESULT FORMATTER FAILED"
        )

        return None

    # =================================
    # DISPLAY
    # =================================

    print_prediction_result(
        result
    )

    # =================================
    # JSON PREVIEW
    # =================================

    print("\n-----------------------------------")
    print("JSON RESULT PREVIEW")
    print("-----------------------------------")

    print(
        json.dumps(
            result,
            indent=2,
            default=str
        )
    )

    # =================================
    # FINAL
    # =================================

    print("\n===================================")
    print(
        "9.8 PREDICTION RESULT FORMATTING "
        "COMPLETED"
    )
    print("===================================\n")

    return result


# ===================================
# DIRECT EXECUTION
# ===================================

if __name__ == "__main__":

    run_prediction_result_formatter()