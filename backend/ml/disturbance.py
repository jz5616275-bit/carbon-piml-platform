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


def apply_basic_disturbance_to_series(
    series: List[Dict[str, Any]],
    global_pct: float,
    *,
    apply_kinds: Tuple[str, ...] = ("test_pred", "forecast"),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    m = 1.0 + float(global_pct)
    if m <= 0:
        raise ValueError("disturbance.global_pct too small (1 + pct must be > 0).")

    out: List[Dict[str, Any]] = []
    applied = 0
    for p in series or []:
        kind = (p.get("kind") or "").strip().lower()
        v = float(p.get("value", 0.0))
        if kind in apply_kinds:
            out.append({"date": p["date"], "value": v * m, "kind": p.get("kind", "")})
            applied += 1
        else:
            out.append({"date": p["date"], "value": v, "kind": p.get("kind", "")})

    summary = {
        "mode": "basic",
        "global_pct": float(global_pct),
        "multiplier": float(m),
        "applied_points": int(applied),
        "apply_kinds": list(apply_kinds),
    }
    return out, summary


def build_basic_observed_whatif(points: List[Dict[str, Any]], global_pct: float) -> List[Dict[str, Any]]:
    disturbed = apply_basic_disturbance(points, global_pct)
    return [{"date": p["date"], "value": float(p["y"]), "kind": "observed_disturbed"} for p in disturbed]


def build_basic_observed_whatif_for_dates(
    observed: List[Dict[str, Any]],
    global_pct: float,
    test_dates: set,
) -> List[Dict[str, Any]]:
    m = 1.0 + float(global_pct)
    if m <= 0:
        raise ValueError("disturbance.global_pct too small (1 + pct must be > 0).")

    out: List[Dict[str, Any]] = []
    for p in observed or []:
        d = p.get("date")
        if not d or d not in test_dates:
            continue
        out.append({"date": d, "value": float(p.get("value", 0.0)) * m, "kind": "observed_disturbed"})
    return out


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


def apply_feature_pct_to_feature_rows(
    feat_rows: List[Dict[str, Any]],
    feature_cols: List[str],
    feature_pct: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    applied_cols: Dict[str, Any] = {}
    out: List[Dict[str, Any]] = []
    pct_map = {str(k): float(v) for k, v in (feature_pct or {}).items()}

    for r in feat_rows or []:
        feats0 = (r.get("features") or {}).copy()
        feats1 = dict(feats0)
        for col in feature_cols or []:
            if col not in feats0:
                continue
            pct = float(pct_map.get(col, 0.0))
            if abs(pct) <= 1e-12:
                continue
            v0 = float(feats0[col])
            v1 = v0 * (1.0 + pct)
            feats1[col] = v1
            if col not in applied_cols:
                applied_cols[col] = {"pct": pct}
        out.append({"date": r.get("date"), "features": feats1})

    summary = {"mode": "advanced", "feature_pct": pct_map, "applied_cols": applied_cols}
    return out, summary