from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple, Callable
import numpy as np


def split_points(points: List[Dict[str, Any]], split: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = len(points)
    mode = (split.get("mode") or "ratio").strip().lower()

    if mode in ("lastn", "last12"):
        k_default = 12 if mode == "last12" else 6
        k = int(split.get("test_points", k_default))
        if k < 2:
            k = 2
        if n <= k:
            raise ValueError("Not enough points for lastN split.")
        return points[: n - k], points[n - k :]

    # ratio
    r = float(split.get("test_ratio", 0.2))
    if r <= 0 or r >= 0.8:
        raise ValueError("evaluation.test_ratio must be in (0, 0.8).")
    k = int(max(2, math.floor(n * r)))
    if n <= k:
        raise ValueError("Not enough points for ratio split.")
    return points[: n - k], points[n - k :]


def _rmse(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return 0.0
    e = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]
    return float(math.sqrt(sum(e) / len(e)))


def _mae(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return 0.0
    e = [abs(a - b) for a, b in zip(y_true, y_pred)]
    return float(sum(e) / len(e))


def _mape(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return 0.0
    eps = 1e-9
    vals = []
    for a, b in zip(y_true, y_pred):
        denom = abs(a) if abs(a) > eps else eps
        vals.append(abs(a - b) / denom)
    return float(sum(vals) / len(vals))


def accuracy_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
    }


def _summarize_corrections(correction_summary: Dict[str, Any] | None) -> Dict[str, Any]:
    if not correction_summary:
        return {"num_adjusted": 0, "by_rule": {}, "max_abs_adjustment": 0.0}

    adjustments = correction_summary.get("adjustments") or []
    by_rule: Dict[str, int] = {}
    max_abs = 0.0

    for a in adjustments:
        rule = str(a.get("rule") or "unknown")
        by_rule[rule] = int(by_rule.get(rule, 0) + 1)

        try:
            f = float(a.get("from"))
            t = float(a.get("to"))
            max_abs = max(max_abs, abs(t - f))
        except Exception:
            pass

    return {
        "num_adjusted": int(correction_summary.get("num_adjusted") or len(adjustments) or 0),
        "by_rule": by_rule,
        "max_abs_adjustment": float(max_abs),
    }


def physical_metrics(
    series: List[Dict[str, Any]],
    *,
    non_negative: bool,
    max_change_rate: float,
    cap_value: float | None,
    prev_value: float | None = None,
    correction_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Two sources:
      - series values -> violations + max magnitudes (works for baseline/piml)
      - correction_summary -> "how many times we clipped" (important for smoothness)
    """
    n = len(series) if series else 0
    if n == 0:
        return {
            "n": 0,
            "violations": {"negatives": 0, "cap": 0, "jump": 0},
            "ratios": {"negatives": 0.0, "cap": 0.0, "jump": 0.0},
            "max": {"cap_excess": 0.0, "jump_excess": 0.0, "abs_change": 0.0, "change_rate": 0.0},
            "corrections": {"num_adjusted": 0, "by_rule": {}, "max_abs_adjustment": 0.0},
        }

    neg = 0
    cap_v = 0
    jump = 0
    max_cap_excess = 0.0
    max_jump_excess = 0.0
    max_abs_change = 0.0
    max_rate_observed = 0.0
    prev = prev_value
    r = float(max_change_rate)

    for p in series:
        y = float(p["value"])

        if non_negative and y < 0:
            neg += 1

        if cap_value is not None:
            capf = float(cap_value)
            if y > capf:
                cap_v += 1
                max_cap_excess = max(max_cap_excess, y - capf)

        if prev is not None:
            abs_change = abs(y - prev)
            max_abs_change = max(max_abs_change, abs_change)

            base = abs(prev) if abs(prev) > 1e-9 else 1.0
            allowed = r * base

            # observed rate: abs change / base
            max_rate_observed = max(max_rate_observed, float(abs_change / base))

            if abs_change > allowed:
                jump += 1
                max_jump_excess = max(max_jump_excess, abs_change - allowed)

        prev = y

    corr_agg = _summarize_corrections(correction_summary)

    def _ratio(c: int) -> float:
        return float(c) / float(n) if n > 0 else 0.0

    return {
        "n": int(n),
        "violations": {"negatives": int(neg), "cap": int(cap_v), "jump": int(jump)},
        "ratios": {"negatives": _ratio(neg), "cap": _ratio(cap_v), "jump": _ratio(jump)},
        "max": {
            "cap_excess": float(max_cap_excess),
            "jump_excess": float(max_jump_excess),
            "abs_change": float(max_abs_change),
            "change_rate": float(max_rate_observed),
        },
        "corrections": corr_agg,
    }


def predict_basic_on_test(train: List[Dict[str, Any]], test: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n_train = len(train)
    x = np.arange(n_train, dtype=float)
    y = np.array([float(p["y"]) for p in train], dtype=float)
    a, b = np.polyfit(x, y, 1)
    n_test = len(test)
    xt = np.arange(n_train, n_train + n_test, dtype=float)
    yt = (a * xt + b).tolist()

    out: List[Dict[str, Any]] = []
    for i, p in enumerate(test):
        out.append({"date": p["date"], "value": float(yt[i]), "kind": "test_pred"})
    return out


def _fit_advanced(train: List[Dict[str, Any]], feature_cols: List[str]) -> np.ndarray:
    X_rows: List[List[float]] = []
    y_vals: List[float] = []

    for p in train:
        feats = p.get("features", {}) or {}
        row: List[float] = []
        ok = True
        for c in feature_cols:
            v = feats.get(c)
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
        X_rows.append(row)
        y_vals.append(float(p["y"]))

    if len(X_rows) < 6:
        raise ValueError("Not enough valid feature rows for advanced evaluation (need >= 6).")

    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=float)
    Xb = np.hstack([np.ones((X.shape[0], 1), dtype=float), X])
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return w


def predict_advanced_on_test(train: List[Dict[str, Any]], test: List[Dict[str, Any]], feature_cols: List[str]) -> List[Dict[str, Any]]:
    w = _fit_advanced(train, feature_cols)

    X_rows: List[List[float]] = []
    dates: List[str] = []

    for p in test:
        feats = p.get("features", {}) or {}
        row: List[float] = []
        ok = True
        for c in feature_cols:
            v = feats.get(c)
            if v is None:
                ok = False
                break
            try:
                row.append(float(v))
            except Exception:
                ok = False
                break
        if not ok:
            raise ValueError("Test split contains missing/invalid feature values.")
        X_rows.append(row)
        dates.append(p["date"])

    X = np.array(X_rows, dtype=float)
    Xb = np.hstack([np.ones((X.shape[0], 1), dtype=float), X])
    y_hat = (Xb @ w).tolist()
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(dates):
        out.append({"date": d, "value": float(y_hat[i]), "kind": "test_pred"})
    return out


def evaluate_history(
    *,
    points_sorted: List[Dict[str, Any]],
    mode_used: str,
    feature_cols: List[str],
    split_cfg: Dict[str, Any],
    physics_params: Dict[str, Any],
    apply_physics_fn: Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]],
) -> Dict[str, Any]:
    train, test = split_points(points_sorted, split_cfg)
    y_true = [float(p["y"]) for p in test]
    prev_anchor = float(train[-1]["y"])
    if mode_used == "basic":
        baseline_test = predict_basic_on_test(train, test)
    else:
        baseline_test = predict_advanced_on_test(train, test, feature_cols)

    piml_test, corr = apply_physics_fn(
        baseline_test,
        physics_mode=physics_params["physics_mode"],
        non_negative=physics_params["non_negative"],
        max_change_rate=physics_params["max_change_rate"],
        cap_value=physics_params["cap_value"],
    )

    y_pred_base = [float(p["value"]) for p in baseline_test]
    y_pred_piml = [float(p["value"]) for p in piml_test]

    base_acc = accuracy_metrics(y_true, y_pred_base)
    piml_acc = accuracy_metrics(y_true, y_pred_piml)

    # baseline has no corrections; piml has corr summary
    base_phys = physical_metrics(
        baseline_test,
        non_negative=bool(physics_params["non_negative"]),
        max_change_rate=float(physics_params["max_change_rate"]),
        cap_value=physics_params["cap_value"],
        prev_value=prev_anchor,
        correction_summary=None,
    )
    piml_phys = physical_metrics(
        piml_test,
        non_negative=bool(physics_params["non_negative"]),
        max_change_rate=float(physics_params["max_change_rate"]),
        cap_value=physics_params["cap_value"],
        prev_value=prev_anchor,
        correction_summary=corr,
    )

    return {
        "split": split_cfg,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_series": {
            "y_true": [{"date": p["date"], "value": float(p["y"]), "kind": "test_true"} for p in test],
            "baseline": baseline_test,
            "piml": piml_test,
        },
        "metrics": {
            "baseline": {"accuracy": base_acc, "physics": base_phys},
            "piml": {"accuracy": piml_acc, "physics": piml_phys},
        },
        "correction_summary": corr,
    }