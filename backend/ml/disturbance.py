from __future__ import annotations
from typing import Any, Dict, List, Tuple


def apply_basic_disturbance(points: List[Dict[str, Any]], global_pct: float) -> List[Dict[str, Any]]:
    m = 1.0 + float(global_pct)
    if m <= 0:
        raise ValueError("disturbance.global_pct too small (1 + pct must be > 0).")

    out: List[Dict[str, Any]] = []
    for p in points:
        out.append({"date": p["date"], "y": float(p["y"]) * m, "features": p.get("features", {}) or {}})
    return out


def build_basic_observed_whatif(points: List[Dict[str, Any]], global_pct: float) -> List[Dict[str, Any]]:
    disturbed = apply_basic_disturbance(points, global_pct)
    return [{"date": p["date"], "value": float(p["y"]), "kind": "observed_disturbed"} for p in disturbed]


def build_advanced_future_features(
    last_features: Dict[str, Any],
    feature_cols: List[str],
    feature_pct: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    applied: Dict[str, Any] = {}
    future: Dict[str, float] = {}

    for col in feature_cols:
        v0_any = (last_features or {}).get(col)
        if v0_any is None:
            continue
        v0 = float(v0_any)
        pct = float(feature_pct.get(col, 0.0))
        v1 = v0 * (1.0 + pct)
        future[col] = v1
        if abs(pct) > 1e-12:
            applied[col] = {"from": v0, "pct": pct, "to": v1}

    summary = {"mode": "advanced", "feature_pct": {k: float(v) for k, v in feature_pct.items()}, "applied": applied}
    return future, summary