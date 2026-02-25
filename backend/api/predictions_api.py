from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
from bson import ObjectId
from dateutil.relativedelta import relativedelta
from flask import Blueprint, jsonify, make_response, request
from backend.api.auth_utils import get_user_from_request
from backend.globals import db
from backend.ml.disturbance import apply_basic_disturbance, build_advanced_future_features
from backend.ml.evaluation import evaluate_history, physical_metrics
from backend.ml.piml_correction import apply_physics_corrections
from backend.utils.data_validator import validate_disturbance, validate_evaluation, validate_horizon

predictions_blueprint = Blueprint("predictions_blueprint", __name__)


def _detect_mode_from_upload(upload: Dict[str, Any]) -> str:
    feature_cols = (upload.get("schema", {}) or {}).get("feature_cols", []) or []
    return "advanced" if len(feature_cols) > 0 else "basic"


def _parse_predict_payload() -> Tuple[str, int, str | None, Dict[str, Any], Dict[str, Any] | None, Dict[str, Any] | None]:
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
        if cap_value < 0:
            raise ValueError("cap_value must be >= 0")

    apply_to = (physics.get("apply_to") or "test_forecast").strip().lower()
    if apply_to not in ("forecast", "test_forecast"):
        raise ValueError("physics.apply_to must be 'forecast' or 'test_forecast'")

    physics_params = {
        "physics_mode": physics_mode,
        "non_negative": non_negative,
        "max_change_rate": max_change_rate,
        "cap_value": cap_value,
        "apply_to": apply_to,
    }

    disturbance = payload.get("disturbance")
    evaluation = payload.get("evaluation")

    return upload_id, horizon, mode_override, physics_params, disturbance, evaluation


def _month_add(iso_date: str, k: int) -> str:
    d = datetime.fromisoformat(iso_date).date()
    return (d + relativedelta(months=+k)).isoformat()


def _day_add(iso_date: str, k: int) -> str:
    d = datetime.fromisoformat(iso_date).date()
    return (d + relativedelta(days=+k)).isoformat()


def _fourier_block(t: np.ndarray, period: float) -> np.ndarray:
    # Returns 2 columns
    ang = (2.0 * np.pi / float(period)) * t
    return np.column_stack([np.sin(ang), np.cos(ang)])


def _basic_design(t: np.ndarray, *, scale_used: str) -> np.ndarray:
    # Basic baseline
    ones = np.ones((t.shape[0], 1), dtype=float)
    trend = t.reshape(-1, 1)

    if scale_used == "monthly":
        s12 = _fourier_block(t, 12.0)
        s6 = _fourier_block(t, 6.0)
        return np.hstack([ones, trend, s12, s6])
    # daily
    s30 = _fourier_block(t, 30.0)
    s7 = _fourier_block(t, 7.0)
    return np.hstack([ones, trend, s30, s7])


def _basic_predict_monthly(points: List[Dict[str, Any]], horizon_months: int) -> List[Dict[str, Any]]:
    n = len(points)
    t = np.arange(n, dtype=float)
    y = np.array([float(p["y"]) for p in points], dtype=float)
    X = _basic_design(t, scale_used="monthly")
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = (X @ w).tolist()
    future_t = np.arange(n, n + horizon_months, dtype=float)
    future_X = _basic_design(future_t, scale_used="monthly")
    future_y = (future_X @ w).tolist()
    last_date = points[-1]["date"]
    future_dates = [_month_add(last_date, i) for i in range(1, horizon_months + 1)]
    out: List[Dict[str, Any]] = []
    for i in range(n):
        out.append({"date": points[i]["date"], "value": float(fitted[i]), "kind": "fitted"})
    for i in range(horizon_months):
        out.append({"date": future_dates[i], "value": float(future_y[i]), "kind": "forecast"})
    return out


def _basic_predict_daily(points: List[Dict[str, Any]], horizon_days: int) -> List[Dict[str, Any]]:
    n = len(points)
    t = np.arange(n, dtype=float)
    y = np.array([float(p["y"]) for p in points], dtype=float)
    X = _basic_design(t, scale_used="daily")
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = (X @ w).tolist()
    future_t = np.arange(n, n + horizon_days, dtype=float)
    future_X = _basic_design(future_t, scale_used="daily")
    future_y = (future_X @ w).tolist()
    last_date = points[-1]["date"]
    future_dates = [_day_add(last_date, i) for i in range(1, horizon_days + 1)]
    out: List[Dict[str, Any]] = []
    for i in range(n):
        out.append({"date": points[i]["date"], "value": float(fitted[i]), "kind": "fitted"})
    for i in range(horizon_days):
        out.append({"date": future_dates[i], "value": float(future_y[i]), "kind": "forecast"})
    return out


def _advanced_design(X: np.ndarray, t: np.ndarray, *, scale_used: str) -> np.ndarray:
    # Advanced baseline
    ones = np.ones((X.shape[0], 1), dtype=float)
    trend = t.reshape(-1, 1)

    if scale_used == "monthly":
        s12 = _fourier_block(t, 12.0)
        s6 = _fourier_block(t, 6.0)
        return np.hstack([ones, X, trend, s12, s6])

    s30 = _fourier_block(t, 30.0)
    s7 = _fourier_block(t, 7.0)
    return np.hstack([ones, X, trend, s30, s7])


def _advanced_fit(
    points: List[Dict[str, Any]],
    feature_cols: List[str],
    *,
    scale_used: str,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    X_rows: List[List[float]] = []
    y_vals: List[float] = []
    dates: List[str] = []

    for p in points:
        feats = p.get("features", {}) or {}
        row: List[float] = []
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
    t = np.arange(X.shape[0], dtype=float)
    Z = _advanced_design(X, t, scale_used=scale_used)
    w, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return w, dates, X


def _advanced_predict_monthly(points: List[Dict[str, Any]], feature_cols: List[str], horizon_months: int) -> List[Dict[str, Any]]:
    w, dates, X = _advanced_fit(points, feature_cols, scale_used="monthly")
    t = np.arange(X.shape[0], dtype=float)
    Z = _advanced_design(X, t, scale_used="monthly")
    y_hat = (Z @ w).tolist()
    last_feats = X[-1, :]
    future_X = np.tile(last_feats, (horizon_months, 1))
    future_t = np.arange(X.shape[0], X.shape[0] + horizon_months, dtype=float)
    future_Z = _advanced_design(future_X, future_t, scale_used="monthly")
    future_y = (future_Z @ w).tolist()
    last_date = dates[-1]
    future_dates = [_month_add(last_date, i) for i in range(1, horizon_months + 1)]
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(dates):
        out.append({"date": d, "value": float(y_hat[i]), "kind": "fitted"})
    for i in range(horizon_months):
        out.append({"date": future_dates[i], "value": float(future_y[i]), "kind": "forecast"})
    return out


def _advanced_predict_daily(points: List[Dict[str, Any]], feature_cols: List[str], horizon_days: int) -> List[Dict[str, Any]]:
    w, dates, X = _advanced_fit(points, feature_cols, scale_used="daily")
    t = np.arange(X.shape[0], dtype=float)
    Z = _advanced_design(X, t, scale_used="daily")
    y_hat = (Z @ w).tolist()
    last_feats = X[-1, :]
    future_X = np.tile(last_feats, (horizon_days, 1))
    future_t = np.arange(X.shape[0], X.shape[0] + horizon_days, dtype=float)
    future_Z = _advanced_design(future_X, future_t, scale_used="daily")
    future_y = (future_Z @ w).tolist()
    last_date = dates[-1]
    future_dates = [_day_add(last_date, i) for i in range(1, horizon_days + 1)]
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(dates):
        out.append({"date": d, "value": float(y_hat[i]), "kind": "fitted"})
    for i in range(horizon_days):
        out.append({"date": future_dates[i], "value": float(future_y[i]), "kind": "forecast"})
    return out


def _advanced_predict_monthly_with_future_features(
    points: List[Dict[str, Any]],
    feature_cols: List[str],
    horizon_months: int,
    future_features: Dict[str, float],
) -> List[Dict[str, Any]]:
    w, dates, X = _advanced_fit(points, feature_cols, scale_used="monthly")
    t = np.arange(X.shape[0], dtype=float)
    Z = _advanced_design(X, t, scale_used="monthly")
    y_hat = (Z @ w).tolist()
    last_feats = X[-1, :].copy()
    for i, col in enumerate(feature_cols):
        if col in future_features:
            last_feats[i] = float(future_features[col])

    future_X = np.tile(last_feats, (horizon_months, 1))
    future_t = np.arange(X.shape[0], X.shape[0] + horizon_months, dtype=float)
    future_Z = _advanced_design(future_X, future_t, scale_used="monthly")
    future_y = (future_Z @ w).tolist()
    last_date = dates[-1]
    future_dates = [_month_add(last_date, i) for i in range(1, horizon_months + 1)]
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(dates):
        out.append({"date": d, "value": float(y_hat[i]), "kind": "fitted"})
    for i in range(horizon_months):
        out.append({"date": future_dates[i], "value": float(future_y[i]), "kind": "forecast"})
    return out


def _advanced_predict_daily_with_future_features(
    points: List[Dict[str, Any]],
    feature_cols: List[str],
    horizon_days: int,
    future_features: Dict[str, float],
) -> List[Dict[str, Any]]:
    w, dates, X = _advanced_fit(points, feature_cols, scale_used="daily")
    t = np.arange(X.shape[0], dtype=float)
    Z = _advanced_design(X, t, scale_used="daily")
    y_hat = (Z @ w).tolist()
    last_feats = X[-1, :].copy()
    for i, col in enumerate(feature_cols):
        if col in future_features:
            last_feats[i] = float(future_features[col])

    future_X = np.tile(last_feats, (horizon_days, 1))
    future_t = np.arange(X.shape[0], X.shape[0] + horizon_days, dtype=float)
    future_Z = _advanced_design(future_X, future_t, scale_used="daily")
    future_y = (future_Z @ w).tolist()
    last_date = dates[-1]
    future_dates = [_day_add(last_date, i) for i in range(1, horizon_days + 1)]
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(dates):
        out.append({"date": d, "value": float(y_hat[i]), "kind": "fitted"})
    for i in range(horizon_days):
        out.append({"date": future_dates[i], "value": float(future_y[i]), "kind": "forecast"})
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


def _build_delta_series(baseline: List[Dict[str, Any]], piml: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not baseline or not piml or len(baseline) != len(piml):
        return out
    for b, p in zip(baseline, piml):
        out.append({"date": b["date"], "kind": b.get("kind", ""), "delta": float(p["value"]) - float(b["value"])})
    return out


def _annotate_fit_test_kinds(baseline: List[Dict[str, Any]], *, n_history: int, n_test: int) -> List[Dict[str, Any]]:
    if not baseline or n_history <= 0 or n_test <= 0:
        return baseline

    split_idx = max(0, n_history - n_test)
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(baseline):
        kind = p.get("kind", "")
        if i < n_history:
            kind = "test_pred" if i >= split_idx else "fit_pred"
        out.append({"date": p["date"], "value": float(p["value"]), "kind": kind})
    return out


def _run_physics(
    baseline: List[Dict[str, Any]],
    physics_params: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    piml, correction_summary = apply_physics_corrections(
        baseline,
        physics_mode=physics_params["physics_mode"],
        non_negative=physics_params["non_negative"],
        max_change_rate=physics_params["max_change_rate"],
        cap_value=physics_params["cap_value"],
        apply_to=physics_params.get("apply_to", "test_forecast"),
    )
    comparison = _compute_comparison(baseline, piml)
    delta_series = _build_delta_series(baseline, piml)
    return piml, correction_summary, comparison, delta_series


@predictions_blueprint.route("/predict", methods=["POST"])
def create_prediction():
    try:
        user = get_user_from_request(request)
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 401)

    try:
        upload_id, horizon_raw, mode_override, physics_params, disturbance_raw, evaluation_raw = _parse_predict_payload()
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
    scale_used = (upload.get("scale_used") or "monthly").strip().lower()

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
        validate_horizon(horizon=horizon_raw, n_points=len(points), scale_used=scale_used)
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
        disturbance = validate_disturbance(disturbance_raw, mode_used=mode_used, feature_cols=feature_cols)
        evaluation = validate_evaluation(evaluation_raw, n_points=len(points_sorted))
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 400)

    n_history = int(len(points_sorted))
    observed = [{"date": p["date"], "value": float(p["y"]), "kind": "observed"} for p in points_sorted]
    horizon_unit = "months" if scale_used == "monthly" else "days"
    horizon_steps = int(horizon_raw)

    try:
        if scale_used == "monthly":
            if mode_used == "basic":
                baseline_original = _basic_predict_monthly(points_sorted, horizon_steps)
                method = "baseline_basic_trend_fourier_monthly"
            else:
                baseline_original = _advanced_predict_monthly(points_sorted, feature_cols, horizon_steps)
                method = "baseline_advanced_lr_features_time_fourier_monthly"
        elif scale_used == "daily":
            if mode_used == "basic":
                baseline_original = _basic_predict_daily(points_sorted, horizon_steps)
                method = "baseline_basic_trend_fourier_daily"
            else:
                baseline_original = _advanced_predict_daily(points_sorted, feature_cols, horizon_steps)
                method = "baseline_advanced_lr_features_time_fourier_daily"
        else:
            return make_response(jsonify({"error": "Unsupported scale. Use daily or monthly data."}), 400)
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 400)
    except Exception as e:
        return make_response(jsonify({"error": "Model fit/predict failed.", "details": str(e)}), 500)

    evaluation_out: Dict[str, Any] | None = None
    n_test = 0
    n_train = n_history
    if evaluation.get("enabled") and evaluation.get("split"):
        try:
            evaluation_out = evaluate_history(
                points_sorted=points_sorted,
                mode_used=mode_used,
                feature_cols=feature_cols,
                split_cfg=evaluation["split"],
                physics_params=physics_params,
                apply_physics_fn=apply_physics_corrections,
            )
            n_test = int(evaluation_out.get("n_test") or 0)
            n_train = int(evaluation_out.get("n_train") or (n_history - n_test))
        except ValueError as e:
            return make_response(jsonify({"error": str(e)}), 400)
        except Exception as e:
            return make_response(jsonify({"error": "Evaluation failed.", "details": str(e)}), 500)

    baseline_original = _annotate_fit_test_kinds(baseline_original, n_history=n_history, n_test=n_test)
    try:
        piml_original, correction_original, comparison_original, delta_original = _run_physics(baseline_original, physics_params)
        violations_original = correction_original.get("violations_series") or []
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 400)
    except Exception as e:
        return make_response(jsonify({"error": "PIML correction failed.", "details": str(e)}), 500)

    prev_anchor = float(points_sorted[-1]["y"])
    original_physics = {
        "baseline": physical_metrics(
            baseline_original,
            non_negative=bool(physics_params["non_negative"]),
            max_change_rate=float(physics_params["max_change_rate"]),
            cap_value=physics_params["cap_value"],
            prev_value=prev_anchor,
            correction_summary=None,
        ),
        "piml": physical_metrics(
            piml_original,
            non_negative=bool(physics_params["non_negative"]),
            max_change_rate=float(physics_params["max_change_rate"]),
            cap_value=physics_params["cap_value"],
            prev_value=prev_anchor,
            correction_summary=correction_original,
        ),
    }

    meta = {
        "scale_used": scale_used,
        "apply_to": physics_params.get("apply_to", "test_forecast"),
        "n_history": n_history,
        "n_train": n_train,
        "n_test": n_test,
        "horizon_steps": horizon_steps,
        "horizon_unit": horizon_unit,
    }

    outputs: Dict[str, Any] = {
        "observed": observed,
        "original": {
            "meta": meta,
            "baseline": baseline_original,
            "piml": piml_original,
            "delta_series": delta_original,
            "violations_series": violations_original,
            "comparison": comparison_original,
            "correction_summary": correction_original,
            "physics": original_physics,
        },
    }

    disturbance_summary: Dict[str, Any] | None = None

    if disturbance["enabled"]:
        try:
            if mode_used == "basic":
                disturbed_points = apply_basic_disturbance(points_sorted, float(disturbance["global_pct"] or 0.0))
                if scale_used == "monthly":
                    baseline_disturbed = _basic_predict_monthly(disturbed_points, horizon_steps)
                else:
                    baseline_disturbed = _basic_predict_daily(disturbed_points, horizon_steps)
                disturbance_summary = {
                    "enabled": True,
                    "mode": "basic",
                    "global_pct": float(disturbance["global_pct"] or 0.0),
                }
            else:
                last_feats = points_sorted[-1].get("features", {}) or {}
                future_feats, disturbance_summary = build_advanced_future_features(
                    last_features=last_feats,
                    feature_cols=feature_cols,
                    feature_pct=disturbance["feature_pct"] or {},
                )
                disturbance_summary["enabled"] = True

                if scale_used == "monthly":
                    baseline_disturbed = _advanced_predict_monthly_with_future_features(
                        points_sorted,
                        feature_cols,
                        horizon_steps,
                        future_features=future_feats,
                    )
                else:
                    baseline_disturbed = _advanced_predict_daily_with_future_features(
                        points_sorted,
                        feature_cols,
                        horizon_steps,
                        future_features=future_feats,
                    )

            baseline_disturbed = _annotate_fit_test_kinds(baseline_disturbed, n_history=n_history, n_test=n_test)
            piml_disturbed, correction_disturbed, comparison_disturbed, delta_disturbed = _run_physics(baseline_disturbed, physics_params)
            violations_disturbed = correction_disturbed.get("violations_series") or []
            disturbed_physics = {
                "baseline": physical_metrics(
                    baseline_disturbed,
                    non_negative=bool(physics_params["non_negative"]),
                    max_change_rate=float(physics_params["max_change_rate"]),
                    cap_value=physics_params["cap_value"],
                    prev_value=prev_anchor,
                    correction_summary=None,
                ),
                "piml": physical_metrics(
                    piml_disturbed,
                    non_negative=bool(physics_params["non_negative"]),
                    max_change_rate=float(physics_params["max_change_rate"]),
                    cap_value=physics_params["cap_value"],
                    prev_value=prev_anchor,
                    correction_summary=correction_disturbed,
                ),
            }

            outputs["disturbed"] = {
                "meta": meta,
                "baseline": baseline_disturbed,
                "piml": piml_disturbed,
                "delta_series": delta_disturbed,
                "violations_series": violations_disturbed,
                "comparison": comparison_disturbed,
                "correction_summary": correction_disturbed,
                "physics": disturbed_physics,
            }

        except ValueError as e:
            return make_response(jsonify({"error": str(e)}), 400)
        except Exception as e:
            return make_response(jsonify({"error": "Disturbance pipeline failed.", "details": str(e)}), 500)

    record = {
        "upload_id": ObjectId(upload_id),
        "owner": user,
        "created_at": datetime.utcnow(),
        "scale_used": scale_used,
        "mode_detected": mode_detected,
        "mode_used": mode_used,
        "method": method,
        "params": {
            "horizon_months": horizon_raw,
            "horizon_unit": horizon_unit,
            "mode_override": mode_override,
            "physics": physics_params,
            "disturbance": disturbance,
            "evaluation": evaluation,
        },
        "outputs": outputs,
        "evaluation": evaluation_out,
        "disturbance_summary": disturbance_summary,
        "limitations": [
            "Horizon uses months for monthly data and days for daily data (API field name kept for compatibility).",
            "Advanced baseline uses the last feature vector for forecasting; disturbance applies to the future feature vector.",
            "Yearly data is rejected; daily and monthly are kept as is.",
            "PIML is implemented as a post-processing correction layer (no retraining).",
            "Disturbed scenarios are for what-if comparison, not accuracy scoring (no ground truth).",
        ],
    }

    try:
        db.predictions.delete_many({"upload_id": ObjectId(upload_id), "owner.user_id": user["user_id"]})
        ins = db.predictions.insert_one(record)
    except Exception as e:
        return make_response(jsonify({"error": "Failed to save prediction record.", "details": str(e)}), 500)

    resp = {
        "message": "Prediction created.",
        "prediction_id": str(ins.inserted_id),
        "upload_id": upload_id,
        "scale_used": record["scale_used"],
        "mode_detected": mode_detected,
        "mode_used": mode_used,
        "method": method,
        "params": record["params"],
        "outputs": outputs,
        "evaluation": evaluation_out,
        "disturbance_summary": disturbance_summary,
        "limitations": record["limitations"],
        "baseline": outputs["original"]["baseline"],
        "piml": outputs["original"]["piml"],
        "comparison": outputs["original"]["comparison"],
        "correction_summary": outputs["original"]["correction_summary"],
    }

    return make_response(jsonify(resp), 201)


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
                    "evaluation": doc.get("evaluation"),
                    "disturbance_summary": doc.get("disturbance_summary"),
                    "limitations": doc.get("limitations", []),
                    "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
                }
            ),
            200,
        )
    except Exception as e:
        return make_response(jsonify({"error": "Failed to fetch prediction.", "details": str(e)}), 500)