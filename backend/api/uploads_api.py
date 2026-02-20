from __future__ import annotations
import io
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
from bson import ObjectId
from flask import Blueprint, jsonify, make_response, request
from backend.api.auth_utils import get_user_from_request
from backend.globals import db
from backend.utils.data_validator import validate_and_prepare_upload

uploads_blueprint = Blueprint("uploads_blueprint", __name__)


@uploads_blueprint.route("/uploads", methods=["POST"])
def create_upload():
    try:
        user = get_user_from_request(request)
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 401)

    if "file" not in request.files:
        return make_response(jsonify({"error": "Missing file field 'file'."}), 400)

    file = request.files["file"]
    if not file or file.filename == "":
        return make_response(jsonify({"error": "No file selected."}), 400)

    if not file.filename.lower().endswith(".csv"):
        return make_response(jsonify({"error": "Only .csv files are supported."}), 400)

    dataset_name = request.form.get("dataset_name", file.filename)
    target_override: Optional[str] = request.form.get("target_col")

    try:
        raw_bytes = file.read()
        df_raw = pd.read_csv(io.BytesIO(raw_bytes))

        clean, points = validate_and_prepare_upload(
            df_raw=df_raw,
            dataset_name=dataset_name,
            target_override=target_override,
        )

        doc: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "created_at": datetime.utcnow(),
            "owner": user,
            "schema": {
                "time_col": clean.time_col,
                "target_col": clean.target_col,
                "feature_cols": clean.feature_cols,
            },
            "scale_detected": clean.scale_detected,
            "scale_used": clean.scale_used,
            "mode_detected": clean.mode_detected,
            "warnings": clean.warnings,
            "stats": clean.stats,
            "data": points,
        }

        ins = db.uploads.insert_one(doc)

        return make_response(
            jsonify(
                {
                    "message": "Upload stored successfully.",
                    "upload_id": str(ins.inserted_id),
                    "dataset_name": dataset_name,
                    "schema": doc["schema"],
                    "scale_detected": doc["scale_detected"],
                    "scale_used": doc["scale_used"],
                    "mode_detected": doc["mode_detected"],
                    "warnings": doc["warnings"],
                    "stats": doc["stats"],
                    "owner": user,
                }
            ),
            201,
        )

    except ValueError as e:
        return make_response(
            jsonify({"error": str(e), "hint": "Check your time column, numeric target, and missing values."}),
            400,
        )
    except Exception as e:
        return make_response(jsonify({"error": "Failed to process CSV.", "details": str(e)}), 500)


@uploads_blueprint.route("/uploads/<upload_id>", methods=["GET"])
def get_upload(upload_id: str):
    try:
        user = get_user_from_request(request)
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 401)

    try:
        doc = db.uploads.find_one({"_id": ObjectId(upload_id), "owner.user_id": user["user_id"]})
        if not doc:
            return make_response(jsonify({"error": "Upload not found."}), 404)

        return make_response(
            jsonify(
                {
                    "upload_id": str(doc["_id"]),
                    "dataset_name": doc.get("dataset_name"),
                    "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
                    "schema": doc.get("schema"),
                    "scale_detected": doc.get("scale_detected"),
                    "scale_used": doc.get("scale_used"),
                    "mode_detected": doc.get("mode_detected"),
                    "warnings": doc.get("warnings", []),
                    "stats": doc.get("stats"),
                    "owner": doc.get("owner"),
                }
            ),
            200,
        )
    except Exception as e:
        return make_response(jsonify({"error": "Failed to fetch upload.", "details": str(e)}), 500)


