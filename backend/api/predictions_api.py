from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from bson import ObjectId
from dateutil.relativedelta import relativedelta
from flask import Blueprint, jsonify, make_response, request
from backend.api.auth_utils import get_user_from_request
from backend.globals import db
from backend.ml.disturbance import (
    apply_basic_disturbance_to_series,
    build_advanced_future_features,
    build_basic_observed_whatif_for_dates,
)
from backend.ml.evaluation import evaluate_history, physical_metrics
from backend.ml.piml_correction import apply_physics_corrections
from backend.utils.data_validator import validate_disturbance, validate_evaluation, validate_horizon

predictions_blueprint = Blueprint("predictions_blueprint", __name__)

def _detect_mode_from_upload(upload: Dict[str, Any]) -> str:
    feature_cols = (upload.get("schema", {}) or {}).get("feature_cols", []) or []
    return "advanced" if len(feature_cols) > 0 else "basic"


def _month_add(iso_date: str, k: int) -> str:
    d = datetime.fromisoformat(iso_date).date()
    return (d + relativedelta(months=+k)).isoformat()


def _day_add(iso_date: str, k: int) -> str:
    d = datetime.fromisoformat(iso_date).date()
    return (d + relativedelta(days=+k)).isoformat()


def _year_add(iso_date: str, k: int) -> str:
    d = datetime.fromisoformat(iso_date).date()
    return (d + relativedelta(years=+k)).isoformat()


def _ridge_solve(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    XtX = X.T @ X
    Xty = X.T @ y
    I = np.eye(X.shape[1], dtype=float)
    a = float(alpha)
    for _ in range(6):
        try:
            return np.linalg.solve(XtX + a * I, Xty)
        except np.linalg.LinAlgError:
            a *= 10.0

    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w


def _time_basis(t: np.ndarray, periods: List[float], *, trend_degree: int = 1) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    feats: List[np.ndarray] = [np.ones_like(t)]
    if trend_degree >= 1:
        feats.append(t)
    if trend_degree >= 2:
        feats.append(t**2)

    for p in periods:
        if p <= 0:
            continue
        ang = 2.0 * np.pi * t / float(p)
        feats.append(np.sin(ang))
        feats.append(np.cos(ang))

    return np.vstack(feats).T


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _compute_test_kpi(
    observed: List[Dict[str, Any]],
    pred_series: List[Dict[str, Any]],
    *,
    n_history: int,
    n_test: int,
) -> Dict[str, Any]:
    if not observed or n_history <= 0 or n_test <= 0:
        return {"rmse": None, "mae": None, "mape": None, "n": 0}

    dates = [p.get("date") for p in observed if p.get("date")]
    if len(dates) < n_history:
        n_history = len(dates)

    split_idx = max(0, n_history - n_test)
    test_dates = set(dates[split_idx:n_history])
    obs_map = {p["date"]: _safe_float(p.get("value")) for p in observed if p.get("date") in test_dates}
    pred_map = {p["date"]: _safe_float(p.get("value")) for p in pred_series if p.get("date") in test_dates and (p.get("kind") or "").lower() == "test_pred"}
    ys = []
    yh = []
    for d, y in obs_map.items():
        if d in pred_map:
            ys.append(float(y))
            yh.append(float(pred_map[d]))

    if not ys:
        return {"rmse": None, "mae": None, "mape": None, "n": 0}

    y_np = np.asarray(ys, dtype=float)
    yh_np = np.asarray(yh, dtype=float)
    err = yh_np - y_np
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(y_np), 1e-9)
    mape = float(np.mean(np.abs(err) / denom)) * 100.0

    return {"rmse": rmse, "mae": mae, "mape": mape, "n": int(len(ys))}


def _basic_fit_predict_monthly(points: List[Dict[str, Any]], n_train: int, n_test: int, horizon_months: int) -> List[Dict[str, Any]]:
    n = len(points)
    t_all = np.arange(n, dtype=float)
    y_all = np.array([float(p["y"]) for p in points], dtype=float)
    n_train = int(max(2, min(n_train, n)))
    X_train = _time_basis(t_all[:n_train], periods=[12.0], trend_degree=1)
    w = _ridge_solve(X_train, y_all[:n_train], alpha=1.0)
    X_fit = _time_basis(t_all[:n_train], periods=[12.0], trend_degree=1)
    fit_pred = (X_fit @ w).tolist()
    X_test = _time_basis(t_all[n_train:n], periods=[12.0], trend_degree=1) if n_train < n else np.zeros((0, X_train.shape[1]))
    test_pred = (X_test @ w).tolist() if len(X_test) else []
    ft = np.arange(n, n + horizon_months, dtype=float)
    Xf = _time_basis(ft, periods=[12.0], trend_degree=1)
    forecast = (Xf @ w).tolist()
    last_date = points[-1]["date"]
    future_dates = [_month_add(last_date, i) for i in range(1, horizon_months + 1)]
    out: List[Dict[str, Any]] = []
    for i in range(n_train):
        out.append({"date": points[i]["date"], "value": float(fit_pred[i]), "kind": "fit_pred"})
    for i in range(n_train, n):
        out.append({"date": points[i]["date"], "value": float(test_pred[i - n_train]), "kind": "test_pred"})
    for i in range(horizon_months):
        out.append({"date": future_dates[i], "value": float(forecast[i]), "kind": "forecast"})
    return out


def _basic_fit_predict_yearly(points: List[Dict[str, Any]], n_train: int, n_test: int, horizon_years: int) -> List[Dict[str, Any]]:
    n = len(points)
    t_all = np.arange(n, dtype=float)
    y_all = np.array([float(p["y"]) for p in points], dtype=float)
    n_train = int(max(2, min(n_train, n)))
    X_train = _time_basis(t_all[:n_train], periods=[], trend_degree=1)
    w = _ridge_solve(X_train, y_all[:n_train], alpha=1.0)
    X_fit = _time_basis(t_all[:n_train], periods=[], trend_degree=1)
    fit_pred = (X_fit @ w).tolist()
    X_test = _time_basis(t_all[n_train:n], periods=[], trend_degree=1) if n_train < n else np.zeros((0, X_train.shape[1]))
    test_pred = (X_test @ w).tolist() if len(X_test) else []
    ft = np.arange(n, n + horizon_years, dtype=float)
    Xf = _time_basis(ft, periods=[], trend_degree=1)
    forecast = (Xf @ w).tolist()
    last_date = points[-1]["date"]
    future_dates = [_year_add(last_date, i) for i in range(1, horizon_years + 1)]
    out: List[Dict[str, Any]] = []
    for i in range(n_train):
        out.append({"date": points[i]["date"], "value": float(fit_pred[i]), "kind": "fit_pred"})
    for i in range(n_train, n):
        out.append({"date": points[i]["date"], "value": float(test_pred[i - n_train]), "kind": "test_pred"})
    for i in range(horizon_years):
        out.append({"date": future_dates[i], "value": float(forecast[i]), "kind": "forecast"})
    return out


def _basic_fit_predict_daily(points: List[Dict[str, Any]], n_train: int, n_test: int, horizon_days: int) -> List[Dict[str, Any]]:
    n = len(points)
    y_all = np.array([float(p["y"]) for p in points], dtype=float)
    n_train = int(max(10, min(n_train, n)))
    lags = [1, 7, 14]
    max_lag = max(lags)

    if n_train <= max_lag + 5:
        t_all = np.arange(n, dtype=float)
        X_train = _time_basis(t_all[:n_train], periods=[7.0, 30.0, 365.25], trend_degree=1)
        w = _ridge_solve(X_train, y_all[:n_train], alpha=50.0)
        fit_pred = (X_train @ w).tolist()
        X_test = _time_basis(t_all[n_train:n], periods=[7.0, 30.0, 365.25], trend_degree=1) if n_train < n else np.zeros((0, X_train.shape[1]))
        test_pred = (X_test @ w).tolist() if len(X_test) else []
        ft = np.arange(n, n + horizon_days, dtype=float)
        Xf = _time_basis(ft, periods=[7.0, 30.0, 365.25], trend_degree=1)
        forecast = (Xf @ w).tolist()
        last_date = points[-1]["date"]
        future_dates = [_day_add(last_date, i) for i in range(1, horizon_days + 1)]
        out: List[Dict[str, Any]] = []
        for i in range(n_train):
            out.append({"date": points[i]["date"], "value": float(fit_pred[i]), "kind": "fit_pred"})
        for i in range(n_train, n):
            out.append({"date": points[i]["date"], "value": float(test_pred[i - n_train]), "kind": "test_pred"})
        for i in range(horizon_days):
            out.append({"date": future_dates[i], "value": float(forecast[i]), "kind": "forecast"})
        return out

    # Build training design
    idx = np.arange(max_lag, n_train, dtype=int)
    t = idx.astype(float)
    t0 = float(max_lag)
    t_scale = float(max(1, (n_train - 1) - max_lag))
    t_norm = (t - t0) / t_scale
    t_season = (idx - max_lag).astype(float)
    X_time = _time_basis(t_season, periods=[7.0, 30.0, 365.25], trend_degree=0)
    X_trend = np.vstack([np.ones_like(t_norm), t_norm]).T

    def _roll_mean(arr: np.ndarray, end_i: int, win: int) -> float:
        s = end_i - win
        if s < 0:
            s = 0
        return float(np.mean(arr[s:end_i]))

    X_ar: List[List[float]] = []
    y_train: List[float] = []
    for i in idx:
        row = [
            float(y_all[i - 1]),
            float(y_all[i - 7]),
            float(y_all[i - 14]),
            _roll_mean(y_all, i, 7),
        ]
        X_ar.append(row)
        y_train.append(float(y_all[i]))

    X_ar_np = np.asarray(X_ar, dtype=float)
    y_np = np.asarray(y_train, dtype=float)
    X = np.hstack([X_trend, X_time, X_ar_np])
    w = _ridge_solve(X, y_np, alpha=5.0)

    # Fit preds on TRAIN
    fit_vals = [float(y_all[i]) for i in range(n_train)]
    for k, i in enumerate(idx):
        fit_vals[i] = float((X[k, :] @ w))
    total_forward = int((n - n_train) + horizon_days)
    buf = list(y_all[:n_train].tolist())
    forward_vals: List[float] = []
    for step in range(1, total_forward + 1):
        i = (n_train - 1) + step
        t_seas = float(i - max_lag)
        t_norm_f = (float(i) - t0) / t_scale
        X_time_f = _time_basis(np.array([t_seas]), periods=[7.0, 30.0, 365.25], trend_degree=0)
        X_trend_f = np.array([[1.0, t_norm_f]], dtype=float)
        lag1 = float(buf[-1])
        lag7 = float(buf[-7]) if len(buf) >= 7 else float(buf[-1])
        lag14 = float(buf[-14]) if len(buf) >= 14 else float(buf[-1])
        rm7 = float(np.mean(buf[-7:])) if len(buf) >= 7 else float(np.mean(buf))
        X_ar_f = np.array([[lag1, lag7, lag14, rm7]], dtype=float)
        Xf = np.hstack([X_trend_f, X_time_f, X_ar_f])
        y_next = float((Xf @ w).reshape(-1)[0])
        forward_vals.append(y_next)
        buf.append(y_next)

    test_len = int(n - n_train)
    test_pred = forward_vals[:test_len]
    forecast = forward_vals[test_len:]
    last_date = points[-1]["date"]
    future_dates = [_day_add(last_date, i) for i in range(1, horizon_days + 1)]
    out: List[Dict[str, Any]] = []
    for i in range(n_train):
        out.append({"date": points[i]["date"], "value": float(fit_vals[i]), "kind": "fit_pred"})
    for i in range(n_train, n):
        out.append({"date": points[i]["date"], "value": float(test_pred[i - n_train]), "kind": "test_pred"})
    for i in range(horizon_days):
        out.append({"date": future_dates[i], "value": float(forecast[i]), "kind": "forecast"})
    return out


def _advanced_fit(
    points: List[Dict[str, Any]],
    feature_cols: List[str],
    *,
    scale_used: str,
    n_train: int,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    X_rows: List[List[float]] = []
    y_vals: List[float] = []
    dates: List[str] = []
    t_vals: List[float] = []
    for i, p in enumerate(points):
        feats = p.get("features", {}) or {}
        row: List[float] = []
        ok = True
        for col in feature_cols:
            v = feats.get(col)
            if v is None:
                ok = False
                break
            try:
                row.append(float(v))
            except Exception:
                ok = False
                break
        if not ok:
            continue

        dates.append(p["date"])
        X_rows.append(row)
        y_vals.append(float(p["y"]))
        t_vals.append(float(i))

    if len(X_rows) < 6:
        raise ValueError("Not enough valid feature rows for advanced prediction (need >= 6).")

    X_feat_all = np.array(X_rows, dtype=float)
    y_all = np.array(y_vals, dtype=float)
    t_all = np.array(t_vals, dtype=float)
    n_train = int(max(2, min(n_train, len(y_all))))
    scale = (scale_used or "monthly").strip().lower()
    if scale == "daily":
        X_time_all = _time_basis(t_all, periods=[7.0, 30.0, 365.25], trend_degree=1)
        alpha = 50.0
    elif scale == "yearly":
        X_time_all = _time_basis(t_all, periods=[], trend_degree=1)
        alpha = 1.0
    else:
        X_time_all = _time_basis(t_all, periods=[12.0], trend_degree=1)
        alpha = 1.0

    X_all = np.hstack([X_time_all, X_feat_all])
    w = _ridge_solve(X_all[:n_train, :], y_all[:n_train], alpha=alpha)
    return w, dates, X_feat_all, t_all


def _advanced_fit_predict(
    points: List[Dict[str, Any]],
    feature_cols: List[str],
    *,
    scale_used: str,
    n_train: int,
    horizon_steps: int,
    future_features: Optional[Dict[str, float]] = None,
    test_feature_pct: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    w, dates, X_feat_all, t_all = _advanced_fit(points, feature_cols, scale_used=scale_used, n_train=n_train)
    n = len(dates)
    scale = (scale_used or "monthly").strip().lower()
    if scale == "daily":
        X_time_all = _time_basis(t_all, periods=[7.0, 30.0, 365.25], trend_degree=1)
        Xf_time = _time_basis(np.arange(n, n + horizon_steps, dtype=float), periods=[7.0, 30.0, 365.25], trend_degree=1)
    elif scale == "yearly":
        X_time_all = _time_basis(t_all, periods=[], trend_degree=1)
        Xf_time = _time_basis(np.arange(n, n + horizon_steps, dtype=float), periods=[], trend_degree=1)
    else:
        X_time_all = _time_basis(t_all, periods=[12.0], trend_degree=1)
        Xf_time = _time_basis(np.arange(n, n + horizon_steps, dtype=float), periods=[12.0], trend_degree=1)

    # apply test feature disturbance
    X_feat_used = X_feat_all.copy()
    if test_feature_pct:
        pct_map = {str(k): float(v) for k, v in (test_feature_pct or {}).items()}
        for j, col in enumerate(feature_cols):
            pct = float(pct_map.get(col, 0.0))
            if abs(pct) <= 1e-12:
                continue
            X_feat_used[n_train:, j] = X_feat_used[n_train:, j] * (1.0 + pct)

    X_all = np.hstack([X_time_all, X_feat_used])
    y_hat_all = (X_all @ w).tolist()
    last_feats = X_feat_all[-1, :].copy()
    if future_features:
        for j, col in enumerate(feature_cols):
            if col in future_features:
                last_feats[j] = float(future_features[col])

    Xf_feat = np.tile(last_feats, (horizon_steps, 1))
    Xf = np.hstack([Xf_time, Xf_feat])
    future_y = (Xf @ w).tolist()
    last_date = dates[-1]
    if scale == "daily":
        future_dates = [_day_add(last_date, i) for i in range(1, horizon_steps + 1)]
    elif scale == "yearly":
        future_dates = [_year_add(last_date, i) for i in range(1, horizon_steps + 1)]
    else:
        future_dates = [_month_add(last_date, i) for i in range(1, horizon_steps + 1)]

    out: List[Dict[str, Any]] = []
    for i in range(min(n_train, n)):
        out.append({"date": dates[i], "value": float(y_hat_all[i]), "kind": "fit_pred"})
    for i in range(n_train, n):
        out.append({"date": dates[i], "value": float(y_hat_all[i]), "kind": "test_pred"})
    for i in range(horizon_steps):
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


def _parse_predict_payload() -> Tuple[
    str,
    int,
    Optional[str],
    str,
    Dict[str, Any],
    Dict[str, Any] | None,
    Dict[str, Any] | None,
    str,
]:
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
    max_change_rate = None
    if "max_change_rate" in physics and physics.get("max_change_rate") is not None:
        try:
            max_change_rate = float(physics.get("max_change_rate"))
        except Exception:
            raise ValueError("physics.max_change_rate must be a number")

    cap_value = None
    if "cap_value" in physics and physics.get("cap_value") is not None:
        try:
            cap_value = float(physics.get("cap_value"))
        except Exception:
            raise ValueError("physics.cap_value must be a number or null")
        if cap_value < 0:
            raise ValueError("cap_value must be >= 0")

    apply_to = (physics.get("apply_to") or "test_forecast").strip().lower()
    if apply_to not in ("forecast", "test_forecast"):
        raise ValueError("physics.apply_to must be 'forecast' or 'test_forecast'")

    physics_user = {
        "non_negative": non_negative,
        "max_change_rate": max_change_rate,
        "cap_value": cap_value,
        "apply_to": apply_to,
    }

    disturbance = payload.get("disturbance")
    evaluation = payload.get("evaluation")
    scenario_name = str(payload.get("scenario_name") or "").strip()

    return upload_id, horizon, mode_override, physics_mode, physics_user, disturbance, evaluation, scenario_name


def _derive_physics_effective(
    *,
    physics_mode: str,
    physics_user: Dict[str, Any],
    scale_used: str,
    history_y: List[float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    mode = (physics_mode or "none").strip().lower()
    scale = (scale_used or "monthly").strip().lower()
    non_negative = bool(physics_user.get("non_negative", True))
    mcr = physics_user.get("max_change_rate", None)
    if mcr is None:
        if scale == "daily":
            mcr = 0.08
            mcr_src = "auto_daily_default_0.08"
        elif scale == "yearly":
            mcr = 0.35
            mcr_src = "auto_yearly_default_0.35"
        else:
            mcr = 0.20
            mcr_src = "auto_monthly_default_0.20"
    else:
        mcr_src = "user"

    try:
        mcr = float(mcr)
    except Exception:
        mcr = 0.20 if scale not in ("daily", "yearly") else (0.08 if scale == "daily" else 0.35)
        mcr_src = "auto_fallback"
    if mcr < 0:
        mcr = abs(mcr)
        mcr_src = f"{mcr_src}_abs"
    if mcr > 1.0:
        mcr = 1.0
        mcr_src = f"{mcr_src}_clamped_1.0"
    cap_value = physics_user.get("cap_value", None)
    cap_src = "user" if cap_value is not None else None
    need_cap = mode in ("cap", "full")
    if need_cap and cap_value is None:
        ys = [float(v) for v in (history_y or []) if v is not None and np.isfinite(float(v))]
        if len(ys) >= 3:
            if scale == "daily":
                cap_value = float(np.percentile(ys, 99.0) * 1.05)
                cap_src = "auto_p99_x1.05"
            elif scale == "yearly":
                cap_value = float(np.percentile(ys, 95.0) * 1.05)
                cap_src = "auto_p95_x1.05"
            else:
                cap_value = float(np.percentile(ys, 98.0) * 1.05)
                cap_src = "auto_p98_x1.05"
        else:
            cap_value = None
            cap_src = "auto_skipped_insufficient_history"

    if cap_value is not None:
        try:
            cap_value = float(cap_value)
            if cap_value < 0:
                cap_value = abs(cap_value)
                cap_src = f"{cap_src}_abs" if cap_src else "abs"
        except Exception:
            cap_value = None
            cap_src = "auto_invalid_cap_value"

    apply_to = (physics_user.get("apply_to") or "test_forecast").strip().lower()
    if apply_to not in ("forecast", "test_forecast"):
        apply_to = "test_forecast"

    physics_effective = {
        "physics_mode": mode,
        "non_negative": non_negative,
        "max_change_rate": float(mcr),
        "cap_value": cap_value,
        "apply_to": apply_to,
    }
    physics_sources = {
        "non_negative": "user_or_default_true",
        "max_change_rate": mcr_src,
        "cap_value": cap_src if cap_src is not None else "not_used",
        "apply_to": "user_or_default_test_forecast",
    }
    return physics_effective, physics_sources


def _run_physics(
    baseline: List[Dict[str, Any]],
    physics_effective: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Optional[str]]:
    piml, correction_summary = apply_physics_corrections(
        baseline,
        physics_mode=physics_effective["physics_mode"],
        non_negative=physics_effective["non_negative"],
        max_change_rate=physics_effective["max_change_rate"],
        cap_value=physics_effective.get("cap_value"),
        apply_to=physics_effective.get("apply_to", "test_forecast"),
    )
    comparison = _compute_comparison(baseline, piml)
    delta_series = _build_delta_series(baseline, piml)
    why_no = None
    try:
        if int((correction_summary or {}).get("num_adjusted") or 0) == 0:
            why_no = "No constraint violations under current physics params; correction skipped."
    except Exception:
        pass

    return piml, correction_summary, comparison, delta_series, why_no


@predictions_blueprint.route("/predictions", methods=["GET"])
def list_predictions():
    try:
        user = get_user_from_request(request)
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 401)

    try:
        raw_limit = request.args.get("limit", "5")
        try:
            limit = int(raw_limit)
        except Exception:
            limit = 5
        limit = max(1, min(50, limit))

        cursor = (
            db.predictions.find({"owner.user_id": user["user_id"]})
            .sort("created_at", -1)
            .limit(limit)
        )

        items: List[Dict[str, Any]] = []
        for doc in cursor:
            params = doc.get("params") or {}
            scenario_name = str(params.get("scenario_name") or "").strip()
            physics_eff = params.get("physics_effective") or {}
            phys_mode = (
                str(physics_eff.get("physics_mode") or params.get("physics_mode") or "")
                .strip()
                .lower()
            )
            outputs = doc.get("outputs") or {}
            orig = outputs.get("original") or {}
            cs = orig.get("correction_summary") or {}
            num_adj = cs.get("num_adjusted")
            try:
                num_adj_i = int(num_adj) if num_adj is not None else 0
            except Exception:
                num_adj_i = 0

            items.append(
                {
                    "prediction_id": str(doc["_id"]),
                    "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
                    "scenario_name": scenario_name,
                    "scale_used": doc.get("scale_used"),
                    "mode_used": doc.get("mode_used"),
                    "method": doc.get("method"),
                    "physics_mode": phys_mode or None,
                    "num_adjusted": num_adj_i,
                }
            )

        return make_response(jsonify({"items": items, "limit": limit}), 200)

    except Exception as e:
        return make_response(jsonify({"error": "Failed to list predictions.", "details": str(e)}), 500)


@predictions_blueprint.route("/predict", methods=["POST"])
def create_prediction():
    try:
        user = get_user_from_request(request)
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 401)

    try:
        upload_id, horizon_raw, mode_override, physics_mode, physics_user, disturbance_raw, evaluation_raw, scenario_name = _parse_predict_payload()
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
        evaluation = validate_evaluation(evaluation_raw, n_points=len(points_sorted), scale_used=scale_used)
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 400)

    n_history = int(len(points_sorted))
    observed = [{"date": p["date"], "value": float(p["y"]), "kind": "observed"} for p in points_sorted]
    horizon_steps = int(horizon_raw)
    horizon_unit = "days" if scale_used == "daily" else ("years" if scale_used == "yearly" else "months")
    history_y = [float(p["y"]) for p in points_sorted]
    physics_effective, physics_sources = _derive_physics_effective(
        physics_mode=physics_mode,
        physics_user=physics_user,
        scale_used=scale_used,
        history_y=history_y,
    )

    # evaluation
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
                physics_effective=physics_effective,
                apply_physics_fn=apply_physics_corrections,
                scale_used=scale_used,
            )
            n_test = int(evaluation_out.get("n_test") or 0)
            n_train = int(evaluation_out.get("n_train") or (n_history - n_test))
        except ValueError as e:
            return make_response(jsonify({"error": str(e)}), 400)
        except Exception as e:
            return make_response(jsonify({"error": "Evaluation failed.", "details": str(e)}), 500)

    n_train = int(max(2, min(n_train, n_history)))
    n_test = int(max(0, min(n_test, n_history - n_train)))

    # baseline fit/predict
    try:
        if scale_used == "monthly":
            if mode_used == "basic":
                baseline_original = _basic_fit_predict_monthly(points_sorted, n_train, n_test, horizon_steps)
                method = "baseline_basic_timebasis_ridge_monthly"
            else:
                baseline_original = _advanced_fit_predict(
                    points_sorted, feature_cols,
                    scale_used="monthly",
                    n_train=n_train,
                    horizon_steps=horizon_steps,
                )
                method = "baseline_advanced_ridge_timebasis_monthly"
        elif scale_used == "daily":
            if mode_used == "basic":
                baseline_original = _basic_fit_predict_daily(points_sorted, n_train, n_test, horizon_steps)
                method = "baseline_basic_timebasis_ridge_daily"
            else:
                baseline_original = _advanced_fit_predict(
                    points_sorted, feature_cols,
                    scale_used="daily",
                    n_train=n_train,
                    horizon_steps=horizon_steps,
                )
                method = "baseline_advanced_ridge_timebasis_daily"
        elif scale_used == "yearly":
            if mode_used == "basic":
                baseline_original = _basic_fit_predict_yearly(points_sorted, n_train, n_test, horizon_steps)
                method = "baseline_basic_timebasis_ridge_yearly"
            else:
                baseline_original = _advanced_fit_predict(
                    points_sorted, feature_cols,
                    scale_used="yearly",
                    n_train=n_train,
                    horizon_steps=horizon_steps,
                )
                method = "baseline_advanced_ridge_timebasis_yearly"
        else:
            return make_response(jsonify({"error": "Unsupported scale. Use daily, monthly, or yearly data."}), 400)
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 400)
    except Exception as e:
        return make_response(jsonify({"error": "Model fit/predict failed.", "details": str(e)}), 500)

    # physics correction
    try:
        piml_original, correction_original, comparison_original, delta_original, why_no_original = _run_physics(
            baseline_original, physics_effective
        )
        violations_original = correction_original.get("violations_series") or []
    except ValueError as e:
        return make_response(jsonify({"error": str(e)}), 400)
    except Exception as e:
        return make_response(jsonify({"error": "PIML correction failed.", "details": str(e)}), 500)

    prev_anchor = float(points_sorted[-1]["y"])
    original_physics = {
        "baseline": physical_metrics(
            baseline_original,
            non_negative=bool(physics_effective["non_negative"]),
            max_change_rate=float(physics_effective["max_change_rate"]),
            cap_value=physics_effective.get("cap_value"),
            prev_value=prev_anchor,
            correction_summary=None,
        ),
        "piml": physical_metrics(
            piml_original,
            non_negative=bool(physics_effective["non_negative"]),
            max_change_rate=float(physics_effective["max_change_rate"]),
            cap_value=physics_effective.get("cap_value"),
            prev_value=prev_anchor,
            correction_summary=correction_original,
        ),
    }

    meta = {
        "scale_used": scale_used,
        "apply_to": physics_effective.get("apply_to", "test_forecast"),
        "n_history": n_history,
        "n_train": n_train,
        "n_test": n_test,
        "horizon_steps": horizon_steps,
        "horizon_unit": horizon_unit,
    }

    # KPI only on TEST
    kpi_test_original_base = _compute_test_kpi(observed, baseline_original, n_history=n_history, n_test=n_test)
    kpi_test_original_piml = _compute_test_kpi(observed, piml_original, n_history=n_history, n_test=n_test)
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
            "physics_effective": physics_effective,
            "physics_sources": physics_sources,
            "why_no_correction": why_no_original,
            "kpi_test": {
                "baseline": kpi_test_original_base,
                "piml": kpi_test_original_piml,
            },
        },
    }

    # disturbance branch
    disturbance_summary: Dict[str, Any] | None = None
    if disturbance["enabled"]:
        try:
            disturbance_note = (
                "Disturbed is a what-if scenario. Real observed history is unchanged; "
                "the disturbance affects ONLY test + forecast outputs (basic) or ONLY test + forecast inputs (advanced). "
                "Accuracy is not scored for forecast because there is no ground truth."
            )

            observed_disturbed_series: List[Dict[str, Any]] | None = None
            observed_for_kpi = observed

            if mode_used == "basic":
                baseline_disturbed, disturbance_summary = apply_basic_disturbance_to_series(
                    baseline_original,
                    float(disturbance["global_pct"] or 0.0),
                    apply_kinds=("test_pred", "forecast"),
                )
                disturbance_summary["enabled"] = True
                if n_test > 0:
                    dates = [p.get("date") for p in observed if p.get("date")]
                    split_idx = max(0, len(dates) - int(n_test))
                    test_dates = set(dates[split_idx:len(dates)])
                    observed_disturbed_series = build_basic_observed_whatif_for_dates(
                        observed,
                        float(disturbance["global_pct"] or 0.0),
                        test_dates,
                    )
                    m = 1.0 + float(disturbance["global_pct"] or 0.0)
                    dm = {p["date"]: float(p["value"]) for p in observed_disturbed_series}
                    observed_for_kpi = [
                        {"date": p["date"], "value": (dm[p["date"]] if p["date"] in dm else float(p["value"])), "kind": p.get("kind", "observed")}
                        for p in observed
                    ]

            else:
                last_feats = points_sorted[-1].get("features", {}) or {}
                future_feats, disturbance_summary = build_advanced_future_features(
                    last_features=last_feats,
                    feature_cols=feature_cols,
                    feature_pct=disturbance["feature_pct"] or {},
                )
                disturbance_summary["enabled"] = True
                test_feature_pct = disturbance["feature_pct"] or {}
                baseline_disturbed = _advanced_fit_predict(
                    points_sorted,
                    feature_cols,
                    scale_used=scale_used,
                    n_train=n_train,
                    horizon_steps=horizon_steps,
                    future_features=future_feats,
                    test_feature_pct=test_feature_pct,
                )

            piml_disturbed, correction_disturbed, comparison_disturbed, delta_disturbed, why_no_disturbed = _run_physics(
                baseline_disturbed, physics_effective
            )
            violations_disturbed = correction_disturbed.get("violations_series") or []
            disturbed_physics = {
                "baseline": physical_metrics(
                    baseline_disturbed,
                    non_negative=bool(physics_effective["non_negative"]),
                    max_change_rate=float(physics_effective["max_change_rate"]),
                    cap_value=physics_effective.get("cap_value"),
                    prev_value=prev_anchor,
                    correction_summary=None,
                ),
                "piml": physical_metrics(
                    piml_disturbed,
                    non_negative=bool(physics_effective["non_negative"]),
                    max_change_rate=float(physics_effective["max_change_rate"]),
                    cap_value=physics_effective.get("cap_value"),
                    prev_value=prev_anchor,
                    correction_summary=correction_disturbed,
                ),
            }

            # KPI only on TEST
            kpi_test_dist_base = _compute_test_kpi(observed_for_kpi, baseline_disturbed, n_history=n_history, n_test=n_test)
            kpi_test_dist_piml = _compute_test_kpi(observed_for_kpi, piml_disturbed, n_history=n_history, n_test=n_test)

            outputs["disturbed"] = {
                "meta": meta,
                "baseline": baseline_disturbed,
                "piml": piml_disturbed,
                "delta_series": delta_disturbed,
                "violations_series": violations_disturbed,
                "comparison": comparison_disturbed,
                "correction_summary": correction_disturbed,
                "physics": disturbed_physics,
                "physics_effective": physics_effective,
                "physics_sources": physics_sources,
                "why_no_correction": why_no_disturbed,
                "disturbance_note": disturbance_note,
                "kpi_test": {
                    "baseline": kpi_test_dist_base,
                    "piml": kpi_test_dist_piml,
                },
            }

            if observed_disturbed_series is not None:
                outputs["disturbed"]["observed_disturbed"] = observed_disturbed_series

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
            "scenario_name": scenario_name,
            "horizon_months": horizon_raw,
            "horizon_unit": horizon_unit,
            "mode_override": mode_override,
            "physics_user": {"physics_mode": physics_mode, **physics_user},
            "physics_effective": physics_effective,
            "physics_sources": physics_sources,
            "disturbance": disturbance,
            "evaluation": evaluation,
        },
        "outputs": outputs,
        "evaluation": evaluation_out,
        "disturbance_summary": disturbance_summary,
        "limitations": [
            "Horizon uses months for monthly data, days for daily data, and years for yearly data (API field name kept for compatibility).",
            "Training is always ridge regression; physics never participates in training; disturbance never retrains.",
            "Physics correction is post-processing and affects ONLY test_pred + forecast (fit_pred is never corrected).",
            "KPI is computed ONLY on test segment; forecast is never included in KPI.",
        ],
    }

    try:
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