from __future__ import annotations
from typing import Any, Dict, List, Tuple


def _apply_non_negative(series: List[Dict[str, Any]], summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in series:
        y0 = float(p["value"])
        y = y0
        if y < 0:
            y = 0.0
            summary["num_adjusted"] += 1
            summary["adjustments"].append({"rule": "non_negative", "date": p.get("date"), "from": y0, "to": y})
        out.append({"date": p["date"], "value": y, "kind": p.get("kind", "unknown")})
    return out


def _apply_cap(series: List[Dict[str, Any]], cap_value: float, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cap = float(cap_value)
    for p in series:
        y0 = float(p["value"])
        y = y0
        if y > cap:
            y = cap
            summary["num_adjusted"] += 1
            summary["adjustments"].append({"rule": "cap_value", "date": p.get("date"), "from": y0, "to": y})
        out.append({"date": p["date"], "value": y, "kind": p.get("kind", "unknown")})
    return out


def _apply_smoothness(series: List[Dict[str, Any]], max_change_rate: float, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    prev: float | None = None
    r = float(max_change_rate)

    for p in series:
        y0 = float(p["value"])
        y = y0
        if prev is not None:
            base = abs(prev) if abs(prev) > 1e-9 else 1.0
            allowed = r * base
            if abs(y - prev) > allowed:
                y = prev + allowed if y > prev else prev - allowed
                summary["num_adjusted"] += 1
                summary["adjustments"].append(
                    {"rule": "smoothness", "date": p.get("date"), "from": y0, "to": y, "prev": prev, "allowed": allowed}
                )

        out.append({"date": p["date"], "value": y, "kind": p.get("kind", "unknown")})
        prev = y

    return out


def apply_physics_corrections(
    baseline_series: List[Dict[str, Any]],
    *,
    physics_mode: str = "none",
    non_negative: bool = True,
    max_change_rate: float = 0.25,
    cap_value: float | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not baseline_series:
        raise ValueError("baseline_series is empty")

    mode = (physics_mode or "none").strip().lower()
    if mode not in ("none", "non_negative", "smoothness", "cap", "full"):
        raise ValueError("physics_mode must be one of: none, non_negative, smoothness, cap, full")

    if mode == "cap" and cap_value is None:
        raise ValueError("physics_mode='cap' requires physics.cap_value")

    if max_change_rate < 0 or max_change_rate > 2:
        raise ValueError("max_change_rate must be between 0 and 2")

    if cap_value is not None and float(cap_value) < 0:
        raise ValueError("cap_value must be >= 0")

    summary: Dict[str, Any] = {
        "physics_mode": mode,
        "non_negative": bool(non_negative),
        "max_change_rate": float(max_change_rate),
        "cap_value": float(cap_value) if cap_value is not None else None,
        "num_adjusted": 0,
        "rules_applied": [],
        "adjustments": [],
    }

    if mode == "none":
        summary["rules_applied"] = ["none"]
        out = [{"date": p["date"], "value": float(p["value"]), "kind": p.get("kind", "unknown")} for p in baseline_series]
        return out, summary

    series = baseline_series

    if mode == "non_negative":
        summary["rules_applied"] = ["non_negative"]
        series = _apply_non_negative(series, summary)
        summary["non_negative"] = True
        return series, summary

    if mode == "smoothness":
        rules = []
        if non_negative:
            rules.append("non_negative")
            series = _apply_non_negative(series, summary)
        rules.append("smoothness")
        series = _apply_smoothness(series, max_change_rate, summary)
        summary["rules_applied"] = rules
        return series, summary

    if mode == "cap":
        rules = []
        if non_negative:
            rules.append("non_negative")
            series = _apply_non_negative(series, summary)
        rules.append("cap_value")
        series = _apply_cap(series, float(cap_value), summary)
        summary["rules_applied"] = rules
        return series, summary

    # full: fixed order for selected rules
    rules = []
    if non_negative:
        rules.append("non_negative")
        series = _apply_non_negative(series, summary)

    if cap_value is not None:
        rules.append("cap_value")
        series = _apply_cap(series, float(cap_value), summary)

    rules.append("smoothness")
    series = _apply_smoothness(series, max_change_rate, summary)

    summary["rules_applied"] = rules
    return series, summary


