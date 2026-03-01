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
    return {"rmse": _rmse(y_true, y_pred), "mae": _mae(y_true, y_pred), "mape": _mape(y_true, y_pred)}


def _summarize_corrections(correction_summary: Dict[str, Any] | None) -> Dict[str, Any]:
    if not correction_summary:
        return {
            "num_adjusted": 0,
            "max_abs_adjustment": 0.0,
            "mean_abs_adjustment": 0.0,
            "adjusted_ratio": 0.0,
            "by_rule_counts": {},
            "adjustments": [],
        }

    by_rule_counts = correction_summary.get("by_rule_counts") or correction_summary.get("by_rule") or {}
    adjustments = correction_summary.get("adjustments") or []
    num_adjusted = correction_summary.get("num_adjusted")
    deltas = []
    for a in adjustments:
        try:
            deltas.append(abs(float(a.get("delta", float(a.get("to")) - float(a.get("from"))))))
        except Exception:
            pass

    max_abs = correction_summary.get("max_abs_adjustment")
    if max_abs is None:
        max_abs = float(max(deltas)) if deltas else 0.0

    mean_abs = correction_summary.get("mean_abs_adjustment")
    if mean_abs is None:
        mean_abs = float(sum(deltas) / len(deltas)) if deltas else 0.0

    if num_adjusted is None:
        num_adjusted = len(adjustments)

    adjusted_ratio = correction_summary.get("adjusted_ratio")
    if adjusted_ratio is None:
        changed = set()
        for a in adjustments:
            changed.add((a.get("date"), a.get("kind")))
        denom = max(1, len(set((p.get("date"), p.get("kind")) for p in adjustments))) if adjustments else 1
        adjusted_ratio = float(len(changed) / denom) if denom else 0.0

    return {
        "num_adjusted": int(num_adjusted or 0),
        "max_abs_adjustment": float(max_abs or 0.0),
        "mean_abs_adjustment": float(mean_abs or 0.0),
        "adjusted_ratio": float(adjusted_ratio or 0.0),
        "by_rule_counts": dict(by_rule_counts or {}),
        "adjustments": adjustments,
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
    n = len(series) if series else 0
    if n == 0:
        return {
            "n": 0,
            "violations": {"negatives": 0, "cap": 0, "jump": 0},
            "ratios": {"negatives": 0.0, "cap": 0.0, "jump": 0.0},
            "max": {"cap_excess": 0.0, "jump_excess": 0.0, "abs_change": 0.0, "change_rate": 0.0},
            "corrections": _summarize_corrections(correction_summary),
            "negatives": 0,
            "cap_violations": 0,
            "jump_violations": 0,
            "negatives_ratio": 0.0,
            "cap_violations_ratio": 0.0,
            "jump_violations_ratio": 0.0,
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
            max_rate_observed = max(max_rate_observed, float(abs_change / base))
            if abs_change > allowed:
                jump += 1
                max_jump_excess = max(max_jump_excess, abs_change - allowed)

        prev = y

    corr_agg = _summarize_corrections(correction_summary)
    def _ratio(c: int) -> float:
        return float(c) / float(n) if n > 0 else 0.0

    neg_r = _ratio(neg)
    cap_r = _ratio(cap_v)
    jump_r = _ratio(jump)

    return {
        "n": int(n),
        "violations": {"negatives": int(neg), "cap": int(cap_v), "jump": int(jump)},
        "ratios": {"negatives": float(neg_r), "cap": float(cap_r), "jump": float(jump_r)},
        "max": {
            "cap_excess": float(max_cap_excess),
            "jump_excess": float(max_jump_excess),
            "abs_change": float(max_abs_change),
            "change_rate": float(max_rate_observed),
        },
        "corrections": corr_agg,
        "negatives": int(neg),
        "cap_violations": int(cap_v),
        "jump_violations": int(jump),
        "negatives_ratio": float(neg_r),
        "cap_violations_ratio": float(cap_r),
        "jump_violations_ratio": float(jump_r),
    }


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


def predict_basic_on_test(train: List[Dict[str, Any]], test: List[Dict[str, Any]], *, scale_used: str) -> List[Dict[str, Any]]:
    n_train = len(train)
    n_test = len(test)
    if n_train < 2 or n_test < 1:
        return [{"date": p["date"], "value": float(train[-1]["y"]) if train else 0.0, "kind": "test_pred"} for p in test]

    scale = (scale_used or "monthly").strip().lower()
    t = np.arange(n_train, dtype=float)
    y = np.array([float(p["y"]) for p in train], dtype=float)

    if scale == "daily":
        X = _time_basis(t, periods=[7.0, 30.0, 365.25], trend_degree=1)
        w = _ridge_solve(X, y, alpha=50.0)
        ft = np.arange(n_train, n_train + n_test, dtype=float)
        Xf = _time_basis(ft, periods=[7.0, 30.0, 365.25], trend_degree=1)
    elif scale == "yearly":
        X = _time_basis(t, periods=[], trend_degree=1)
        w = _ridge_solve(X, y, alpha=1.0)
        ft = np.arange(n_train, n_train + n_test, dtype=float)
        Xf = _time_basis(ft, periods=[], trend_degree=1)
    else:
        X = _time_basis(t, periods=[12.0], trend_degree=1)
        w = _ridge_solve(X, y, alpha=1.0)
        ft = np.arange(n_train, n_train + n_test, dtype=float)
        Xf = _time_basis(ft, periods=[12.0], trend_degree=1)

    y_hat = (Xf @ w).tolist()
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(test):
        out.append({"date": p["date"], "value": float(y_hat[i]), "kind": "test_pred"})
    return out


def predict_advanced_on_test(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    feature_cols: List[str],
    *,
    scale_used: str,
) -> List[Dict[str, Any]]:
    n_train = len(train)
    n_test = len(test)
    if n_train < 6 or n_test < 1:
        raise ValueError("Not enough valid feature rows for advanced evaluation (need >= 6).")

    scale = (scale_used or "monthly").strip().lower()
    X_train_rows: List[List[float]] = []
    y_train_vals: List[float] = []
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
        X_train_rows.append(row)
        y_train_vals.append(float(p["y"]))

    if len(X_train_rows) < 6:
        raise ValueError("Not enough valid feature rows for advanced evaluation (need >= 6).")

    X_feat = np.array(X_train_rows, dtype=float)
    y = np.array(y_train_vals, dtype=float)
    t_train = np.arange(len(X_feat), dtype=float)
    if scale == "daily":
        X_time = _time_basis(t_train, periods=[7.0, 30.0, 365.25], trend_degree=1)
        alpha = 50.0
    elif scale == "yearly":
        X_time = _time_basis(t_train, periods=[], trend_degree=1)
        alpha = 1.0
    else:
        X_time = _time_basis(t_train, periods=[12.0], trend_degree=1)
        alpha = 1.0

    X_train = np.hstack([X_time, X_feat])
    w = _ridge_solve(X_train, y, alpha=alpha)

    X_test_rows: List[List[float]] = []
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
        X_test_rows.append(row)
        dates.append(p["date"])

    Xf_feat = np.array(X_test_rows, dtype=float)
    t_test = np.arange(len(X_feat), len(X_feat) + n_test, dtype=float)
    if scale == "daily":
        Xf_time = _time_basis(t_test, periods=[7.0, 30.0, 365.25], trend_degree=1)
    elif scale == "yearly":
        Xf_time = _time_basis(t_test, periods=[], trend_degree=1)
    else:
        Xf_time = _time_basis(t_test, periods=[12.0], trend_degree=1)

    X_test = np.hstack([Xf_time, Xf_feat])
    y_hat = (X_test @ w).tolist()

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
    physics_effective: Dict[str, Any],
    apply_physics_fn: Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    scale_used: str = "monthly",
) -> Dict[str, Any]:
    train, test = split_points(points_sorted, split_cfg)
    y_true = [float(p["y"]) for p in test]
    prev_anchor = float(train[-1]["y"])

    if mode_used == "basic":
        baseline_test = predict_basic_on_test(train, test, scale_used=scale_used)
    else:
        baseline_test = predict_advanced_on_test(train, test, feature_cols, scale_used=scale_used)

    apply_to_eff = (physics_effective.get("apply_to") or "test_forecast").strip().lower()
    piml_test, corr = apply_physics_fn(
        baseline_test,
        physics_mode=physics_effective["physics_mode"],
        non_negative=physics_effective["non_negative"],
        max_change_rate=physics_effective["max_change_rate"],
        cap_value=physics_effective.get("cap_value"),
        apply_to=apply_to_eff,
        prev_value=prev_anchor,
    )
    corr_sum = _summarize_corrections(corr)
    why_no = None
    try:
        if int(corr_sum.get("num_adjusted") or 0) == 0:
            why_no = "No constraint violations under current physics params; correction skipped."
    except Exception:
        pass

    y_pred_base = [float(p["value"]) for p in baseline_test]
    y_pred_piml = [float(p["value"]) for p in piml_test]
    base_acc = accuracy_metrics(y_true, y_pred_base)
    piml_acc = accuracy_metrics(y_true, y_pred_piml)

    base_phys = physical_metrics(
        baseline_test,
        non_negative=bool(physics_effective["non_negative"]),
        max_change_rate=float(physics_effective["max_change_rate"]),
        cap_value=physics_effective.get("cap_value"),
        prev_value=prev_anchor,
        correction_summary=None,
    )
    piml_phys = physical_metrics(
        piml_test,
        non_negative=bool(physics_effective["non_negative"]),
        max_change_rate=float(physics_effective["max_change_rate"]),
        cap_value=physics_effective.get("cap_value"),
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
        "correction_summary": corr_sum,
        "physics_effective": physics_effective,
        "why_no_correction": why_no,
    }