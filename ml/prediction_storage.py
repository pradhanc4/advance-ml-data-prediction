import json

from database.connection import SessionLocal
from database.models import PredictionRecord


def save_prediction_to_sql(
    prediction_result,
    prediction_date
):

    if not prediction_result:
        raise ValueError(
            "Prediction result is empty."
        )

    db = SessionLocal()

    try:

        prediction = prediction_result[
            "prediction"
        ]

        ensemble = prediction_result[
            "ensemble"
        ]

        confidence = prediction_result[
            "confidence"
        ]

        agreement = prediction_result[
            "model_agreement"
        ]

        models = prediction_result[
            "models"
        ]

        target_column = prediction_result[
            "target_column"
        ]

        # -----------------------------------
        # CHECK EXISTING PENDING PREDICTION
        # -----------------------------------

        existing = (
            db.query(PredictionRecord)
            .filter(
                PredictionRecord.prediction_date
                == prediction_date
            )
            .filter(
                PredictionRecord.target_column
                == target_column
            )
            .filter(
                PredictionRecord.prediction_status
                == "PENDING"
            )
            .first()
        )

        if existing:

            print(
                "Prediction already exists "
                "for this date."
            )

            print(
                f"Prediction ID: {existing.id}"
            )

            return existing

        # -----------------------------------
        # CREATE SQL RECORD
        # -----------------------------------

        prediction_record = PredictionRecord(

            prediction_date=prediction_date,

            target_column=target_column,

            predicted_value=int(
                prediction
            ),

            ensemble_probability=float(
                ensemble["probability"]
            ),

            confidence_score=float(
                confidence["score"]
            ),

            confidence_category=(
                confidence["category"]
            ),

            probability_margin=float(
                confidence["probability_margin"]
            ),

            entropy=float(
                confidence["entropy"]
            ),

            normalized_entropy=float(
                confidence["normalized_entropy"]
            ),

            agreement_count=int(
                agreement["agreement_count"]
            ),

            total_models=int(
                agreement["total_models"]
            ),

            agreement_ratio=float(
                agreement["agreement_ratio"]
            ),

            unique_predictions=int(
                agreement["unique_predictions"]
            ),

            ensemble_probabilities=json.dumps(
                ensemble["probabilities"]
            ),

            top_predictions=json.dumps(
                ensemble["top_predictions"]
            ),

            model_predictions=json.dumps(
                models
            ),

            result_json=json.dumps(
                prediction_result,
                default=str
            ),

            actual_value=None,

            actual_record_date=None,

            prediction_status="PENDING"
        )

        # -----------------------------------
        # SAVE
        # -----------------------------------

        db.add(
            prediction_record
        )

        db.commit()

        db.refresh(
            prediction_record
        )

        print("\n-----------------------------------")
        print("SQL PREDICTION STORAGE")
        print("-----------------------------------")

        print(
            "Prediction saved: YES"
        )

        print(
            f"Prediction ID: "
            f"{prediction_record.id}"
        )

        print(
            f"Prediction date: "
            f"{prediction_record.prediction_date}"
        )

        print(
            f"Target: "
            f"{prediction_record.target_column}"
        )

        print(
            f"Prediction: "
            f"{prediction_record.predicted_value}"
        )

        print(
            f"Probability: "
            f"{prediction_record.ensemble_probability:.2f}%"
        )

        print(
            f"Confidence: "
            f"{prediction_record.confidence_score:.2f}%"
        )

        print(
            f"Status: "
            f"{prediction_record.prediction_status}"
        )

        return prediction_record

    except Exception:

        db.rollback()

        raise

    finally:

        db.close()
        