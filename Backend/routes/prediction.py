from datetime import timedelta

from flask import Blueprint, jsonify

from ml.prediction_result import (
    run_prediction_result_formatter
)

from ml.prediction_engine import (
    prepare_prediction_input
)

from ml.prediction_storage import (
    save_prediction_to_sql
)

from database.connection import SessionLocal
from database.models import PredictionRecord


prediction_bp = Blueprint(
    "prediction",
    __name__,
    url_prefix="/api/prediction"
)


# ===================================
# PREDICTION API HEALTH
# ===================================

@prediction_bp.route(
    "/health",
    methods=["GET"]
)
def prediction_health():

    return jsonify({
        "status": "success",
        "message": "Prediction API is running",
        "service": "prediction"
    })


# ===================================
# CREATE PREDICTION
# ===================================

@prediction_bp.route(
    "",
    methods=["POST"]
)
def create_prediction():

    try:

        # -----------------------------------
        # STEP 1: CREATE PREDICTION RESULT
        # -----------------------------------

        result = run_prediction_result_formatter()

        if result is None:

            return jsonify({
                "status": "error",
                "message": (
                    "Prediction result creation failed."
                )
            }), 500

        # -----------------------------------
        # STEP 2: LOAD HISTORICAL DATA
        # -----------------------------------

        historical_data = (
            prepare_prediction_input()
        )

        if historical_data is None:

            return jsonify({
                "status": "error",
                "message": (
                    "Historical prediction input "
                    "is unavailable."
                )
            }), 500

        # -----------------------------------
        # STEP 3: DETERMINE PREDICTION DATE
        # -----------------------------------

        latest_date = (
            historical_data[
                "record_date"
            ].max()
        )

        prediction_date = (
            latest_date
            + timedelta(days=1)
        )

        # -----------------------------------
        # STEP 4: SAVE TO SQL
        # -----------------------------------

        prediction_record = (
            save_prediction_to_sql(
                result,
                prediction_date
            )
        )

        if prediction_record is None:

            return jsonify({
                "status": "error",
                "message": (
                    "Prediction could not be "
                    "stored in SQL."
                )
            }), 500

        # -----------------------------------
        # STEP 5: API RESPONSE
        # -----------------------------------

        return jsonify({

            "status": "success",

            "message": (
                "Prediction generated "
                "successfully."
            ),

            "prediction": {

                "id": prediction_record.id,

                "prediction_date": str(
                    prediction_record.prediction_date
                ),

                "target_column": (
                    prediction_record.target_column
                ),

                "predicted_value": (
                    prediction_record.predicted_value
                ),

                "ensemble_probability": (
                    prediction_record
                    .ensemble_probability
                ),

                "confidence_score": (
                    prediction_record
                    .confidence_score
                ),

                "confidence_category": (
                    prediction_record
                    .confidence_category
                ),

                "prediction_status": (
                    prediction_record
                    .prediction_status
                )
            },

            "result": result
        })

    except Exception as error:

        return jsonify({

            "status": "error",

            "message": (
                "Prediction API failed."
            ),

            "error": str(error)

        }), 500


# ===================================
# PREDICTION HISTORY
# ===================================

@prediction_bp.route(
    "/history",
    methods=["GET"]
)
def prediction_history():

    db = SessionLocal()

    try:

        records = (
            db.query(PredictionRecord)
            .order_by(
                PredictionRecord
                .prediction_date
                .desc()
            )
            .all()
        )

        history = []

        for record in records:

            history.append({

                "id": record.id,

                "prediction_date": str(
                    record.prediction_date
                ),

                "target_column": (
                    record.target_column
                ),

                "predicted_value": (
                    record.predicted_value
                ),

                "ensemble_probability": (
                    record.ensemble_probability
                ),

                "confidence_score": (
                    record.confidence_score
                ),

                "confidence_category": (
                    record.confidence_category
                ),

                "actual_value": (
                    record.actual_value
                ),

                "prediction_status": (
                    record.prediction_status
                ),

                "created_at": (
                    str(record.created_at)
                    if record.created_at
                    else None
                )
            })

        return jsonify({

            "status": "success",

            "total_records": len(
                history
            ),

            "predictions": history

        })

    except Exception as error:

        return jsonify({

            "status": "error",

            "message": (
                "Prediction history "
                "retrieval failed."
            ),

            "error": str(error)

        }), 500

    finally:

        db.close()