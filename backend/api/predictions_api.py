from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
from bson import ObjectId
from dateutil.relativedelta import relativedelta
from flask import Blueprint, jsonify, make_response, request
from backend.api.auth_utils import get_user_from_request
from backend.globals import db
from backend.ml.piml_correction import apply_physics_corrections
from backend.utils.data_validator import validate_horizon
predictions_blueprint = Blueprint("predictions_blueprint", __name__)


def _detect_mode_from_upload(upload: Dict[str, Any]) -> str:
    feature_cols = (upload.get("schema", {}) or {}).get("feature_cols", []) or []
    return "advanced" if len(feature_cols) > 0 else "basic"


def _parse_predict_payload() -> Tuple[str, int, str | None, Dict[str, Any]]:
    payload = request.get_json(silent=True) or {}
    upload_id = payload.get("upload_id")
    if not upload_id:
        raise ValueError("Missing field: upload_id")

    horizon = payload.get("horizon_months", 12)
    try:
        horizon = int(horizon)
    except Exception:
        raise ValueError("horizon_months must be an integer")

    mode_override = payload.get("mode_override")
    if mode_override is not None and mode_override not in ("basic", "advanced"):
        raise ValueError("mode_override must be 'basic' or 'advanced'")

    physics_mode = (payload.get("physics_mode") or "none").strip().lower()
    if physics_mode not in ("none", "non_negative", "smoothness", "cap", "full"):
        raise ValueError("physics_mode must be one of: none, non_negative, smoothness, cap, full")

    physics = payload.get("physics") or {}
    if physics is None:
        physics = {}

    non_negative = bool(physics.get("non_negative", True))
    max_change_rate = physics.get("max_change_rate", 0.25)
    try:
        max_change_rate = float(max_change_rate)
    except Exception:
        raise ValueError("physics.max_change_rate must be a number")

    cap_value = physics.get("cap_value", None)
    if cap_value is not None:
        try:
            cap_value = float(cap_value)
        except Exception:
            raise ValueError("physics.cap_value must be a number or null")

    physics_params = {
        "physics_mode": physics_mode,
        "non_negative": non_negative,
        "max_change_rate": max_change_rate,
        "cap_value": cap_value,
    }

    return upload_id, horizon, mode_override, physics_params


def _month_add(iso_date: str, k: int) -> str:
    d = datetime.fromisoformat(iso_date).date()
    return (d + relativedelta(months=+k)).isoformat()


def _basic_predict_monthly(points: List[Dict[str, Any]], horizon: int) -> List[Dict[str, Any]]:
    n = len(points)
    x = np.arange(n, dtype=float)
    y = np.array([float(p["y"]) for p in points], dtype=float)
    a, b = np.polyfit(x, y, 1)
    fitted = (a * x + b).tolist()
    future_x = np.arange(n, n + horizon, dtype=float)
    future_y = (a * future_x + b).tolist()
    last_date = points[-1]["date"]
    future_dates = [_month_add(last_date, i) for i in range(1, horizon + 1)]
    out: List[Dict[str, Any]] = []
    for i in range(n):
        out.append({"date": points[i]["date"], "value": fitted[i], "kind": "fitted"})
    for i in range(horizon):
        out.append({"date": future_dates[i], "value": future_y[i], "kind": "forecast"})
    return out


def _advanced_predict_monthly(points: List[Dict[str, Any]], feature_cols: List[str], horizon: int) -> List[Dict[str, Any]]:
    X_rows = []
    y_vals = []
    dates = []

    for p in points:
        feats = p.get("features", {}) or {}
        row = []
        valid = True
        for col in feature_cols:
            v = feats.get(col)
            if v is None:
                valid = False
                break
            try:
                row.append(float(v))
            except Exception:
                valid = False
                break
        if not valid:
            continue

        dates.append(p["date"])
        X_rows.append(row)
        y_vals.append(float(p["y"]))

    if len(X_rows) < 6:
        raise ValueError("Not enough valid feature rows for advanced prediction (need >= 6).")

    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=float)
    Xb = np.hstack([np.ones((X.shape[0], 1), dtype=float), X])
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    y_hat = (Xb @ w).tolist()
    last_feats = X[-1, :]
    future_X = np.tile(last_feats, (horizon, 1))
    future_Xb = np.hstack([np.ones((horizon, 1), dtype=float), future_X])
    future_y = (future_Xb @ w).tolist()
    last_date = dates[-1]
    future_dates = [_month_add(last_date, i) for i in range(1, horizon + 1)]
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(dates):
        out.append({"date": d, "value": y_hat[i], "kind": "fitted"})
    for i in range(horizon):
        out.append({"date": future_dates[i], "value": future_y[i], "kind": "forecast"})
    return out


def _compute_comparison(baseline: List[Dict[str, Any]], piml: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not baseline or not piml or len(baseline) != len(piml):
        return {"mean_abs_adjustment": 0.0, "max_abs_adjustment": 0.0, "adjusted_ratio": 0.0}

    diffs = [abs(float(b["value"]) - float(p["value"])) for b, p in zip(baseline, piml)]
    mean_abs = float(sum(diffs) / len(diffs)) if diffs else 0.0
    max_abs = float(max(diffs)) if diffs else 0.0
    changed = sum(1 for b, p in zip(baseline, piml) if float(b["value"]) != float(p["value"]))
    ratio = float(changed) / float(len(piml)) if piml else 0.0

    return {"mean_abs_adjustment": mean_abs, "max_abs_adjustment": max_abs, "adjusted_ratio": ratio}


@predictions_blueprint.route("/predict", methods=["POST"])
def create_prediction():
    try:
        user = get_user_from_request(request)
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 401)

    try:
        upload_id, horizon, mode_override, physics_params = _parse_predict_payload()
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 400)

    try:
        upload = db.uploads.find_one({"_id": ObjectId(upload_id), "owner.user_id": user["user_id"]})
        if not upload:
            return make_response(jsonify({"error": "Upload not found."}), 404)
    except Exception as e:
        return make_response(jsonify({"error": "Invalid upload_id.", "details": str(e)}), 400)

    schema = upload.get("schema", {}) or {}
    feature_cols = schema.get("feature_cols", []) or []
    mode_detected = upload.get("mode_detected") or _detect_mode_from_upload(upload)

    if mode_override == "advanced" and mode_detected == "basic":
        return make_response(
            jsonify({"error": "mode_override='advanced' not allowed for a basic upload (no feature columns)."}),
            400,
        )

    mode_used = mode_override if mode_override is not None else mode_detected
    points = upload.get("data", []) or []
    if len(points) < 2:
        return make_response(jsonify({"error": "Not enough data points to predict."}), 400)

    try:
        validate_horizon(horizon=horizon, n_points=len(points))
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 400)

    try:
        points_sorted = sorted(
            [{"date": p["date"], "y": p["y"], "features": p.get("features", {})} for p in points],
            key=lambda x: x["date"],
        )
    except Exception as e:
        return make_response(jsonify({"error": "Failed to parse upload data.", "details": str(e)}), 500)

    try:
        if mode_used == "basic":
            baseline = _basic_predict_monthly(points_sorted, horizon)
            method = "baseline_basic_time_trend_monthly"
        else:
            baseline = _advanced_predict_monthly(points_sorted, feature_cols, horizon)
            method = "baseline_advanced_linear_regression_monthly"
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 400)
    except Exception as e:
        return make_response(jsonify({"error": "Model fit/predict failed.", "details": str(e)}), 500)

    try:
        piml, correction_summary = apply_physics_corrections(
            baseline,
            physics_mode=physics_params["physics_mode"],
            non_negative=physics_params["non_negative"],
            max_change_rate=physics_params["max_change_rate"],
            cap_value=physics_params["cap_value"],
        )
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 400)
    except Exception as e:
        return make_response(jsonify({"error": "PIML correction failed.", "details": str(e)}), 500)

    comparison = _compute_comparison(baseline, piml)

    record = {
        "upload_id": ObjectId(upload_id),
        "owner": user,
        "created_at": datetime.utcnow(),
        "scale_used": upload.get("scale_used", "monthly"),
        "mode_detected": mode_detected,
        "mode_used": mode_used,
        "method": method,
        "params": {"horizon_months": horizon, "mode_override": mode_override, "physics": physics_params},
        "outputs": {
            "baseline": baseline,
            "piml": piml,
            "comparison": comparison,
            "correction_summary": correction_summary,
        },
        "limitations": [
            "Advanced baseline holds the last feature vector constant for forecasting.",
            "Yearly data is rejected; daily data is resampled to monthly.",
            "PIML is implemented as a user-selectable post-processing correction layer (not retraining the baseline).",
        ],
    }

    try:
        db.predictions.delete_many({"upload_id": ObjectId(upload_id), "owner.user_id": user["user_id"]})
        ins = db.predictions.insert_one(record)
    except Exception as e:
        return make_response(jsonify({"error": "Failed to save prediction record.", "details": str(e)}), 500)

    return make_response(
        jsonify(
            {
                "message": "Prediction created.",
                "prediction_id": str(ins.inserted_id),
                "upload_id": upload_id,
                "scale_used": record["scale_used"],
                "mode_detected": mode_detected,
                "mode_used": mode_used,
                "method": method,
                "params": record["params"],
                "baseline": baseline,
                "piml": piml,
                "comparison": comparison,
                "correction_summary": correction_summary,
                "limitations": record["limitations"],
            }
        ),
        201,
    )


@predictions_blueprint.route("/predictions/<prediction_id>", methods=["GET"])
def get_prediction(prediction_id: str):
    try:
        user = get_user_from_request(request)
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 401)

    try:
        doc = db.predictions.find_one({"_id": ObjectId(prediction_id), "owner.user_id": user["user_id"]})
        if not doc:
            return make_response(jsonify({"error": "Prediction not found."}), 404)

        return make_response(
            jsonify(
                {
                    "prediction_id": str(doc["_id"]),
                    "upload_id": str(doc["upload_id"]),
                    "scale_used": doc.get("scale_used"),
                    "mode_detected": doc.get("mode_detected"),
                    "mode_used": doc.get("mode_used"),
                    "method": doc.get("method"),
                    "params": doc.get("params"),
                    "outputs": doc.get("outputs"),
                    "limitations": doc.get("limitations", []),
                    "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
                }
            ),
            200,
        )
    except Exception as e:
        return make_response(jsonify({"error": "Failed to fetch prediction.", "details": str(e)}), 500)






