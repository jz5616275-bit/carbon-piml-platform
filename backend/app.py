from flask import Flask, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
from pathlib import Path
from backend.globals import db
from backend.api.uploads_api import uploads_blueprint
from backend.api.predictions_api import predictions_blueprint


def create_app() -> Flask:
    # Always load .env from project root
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    app = Flask(__name__)
    CORS(
        app,
        allow_headers=["Content-Type", "Authorization"],
        expose_headers=["Content-Type"],
    )

    app.register_blueprint(uploads_blueprint, url_prefix="/api")
    app.register_blueprint(predictions_blueprint, url_prefix="/api")

    @app.route("/api/health", methods=["GET"])
    def health():
        return make_response(jsonify({"status": "ok"}), 200)

    @app.route("/api/health/db", methods=["GET"])
    def health_db():
        try:
            ping = db.command("ping")
            return make_response(
                jsonify({"db": "connected", "ping": ping.get("ok", 0)}),
                200,
            )
        except Exception as e:
            return make_response(
                jsonify({"db": "error", "message": str(e)}),
                500,
            )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )





