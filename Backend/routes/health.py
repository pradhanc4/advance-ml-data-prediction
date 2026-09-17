from flask import Blueprint

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health", methods=["GET"])
def health_check():
    return {
        "status": "healthy",
        "service": "ML Prediction Platform"
    }