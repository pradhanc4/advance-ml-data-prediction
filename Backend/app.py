from flask import Flask
from flask_cors import CORS

from Backend.routes.health import health_bp
from Backend.routes.prediction import prediction_bp


def create_app():
    app = Flask(__name__)

    CORS(app)

    app.register_blueprint(health_bp)
    app.register_blueprint(prediction_bp)

    @app.route("/")
    def home():
        return {
            "message": "ML Prediction Platform API is running",
            "status": "success"
        }

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)