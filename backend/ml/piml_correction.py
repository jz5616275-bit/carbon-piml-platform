from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _is_target_kind(kind: str, apply_to: str) -> bool:
    k = (kind or "").strip().lower()
    a = (apply_to or "test_forecast").strip().lower()

    if a == "forecast":
        return k == "forecast"
    return k in ("test_pred", "forecast")


def apply_physics_corrections(
    series: List[Dict[str, Any]],
    *,
    physics_mode: str = "none",
    non_negative: bool = True,
    max_change_rate: float = 0.25,
    cap_value: Optional[float] = None,
    apply_to: str = "test_forecast",
    prev_value: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode = (physics_mode or "none").strip().lower()
    a2 = (apply_to or "test_forecast").strip().lower()

    if mode == "none":
        out = [{"date": p["date"], "value": _safe_float(p["value"]), "kind": p.get("kind", "")} for p in series]
        return out, {
            "num_adjusted": 0,
            "max_abs_adjustment": 0.0,
            "mean_abs_adjustment": 0.0,
            "adjusted_ratio": 0.0,
            "by_rule_counts": {},
            "adjustments": [],
            "violations_series": [],
        }

    use_nonneg = non_negative and mode in ("non_negative", "cap", "smoothness", "full")
    use_cap = cap_value is not None and mode in ("cap", "full")
    use_rate = mode in ("smoothness", "full")
    corrected: List[Dict[str, Any]] = []
    adjustments: List[Dict[str, Any]] = []
    by_rule: Dict[str, int] = {}
    violations_series: List[Dict[str, Any]] = []
    prev = float(prev_value) if prev_value is not None else None

    for p in series:
        d = p["date"]
        kind = p.get("kind", "")
        y0 = _safe_float(p["value"])
        y = float(y0)
        rules_hit: List[str] = []

        if _is_target_kind(kind, a2):

            # non-negative
            if use_nonneg and y < 0:
                rules_hit.append("non_negative")
                by_rule["non_negative"] = by_rule.get("non_negative", 0) + 1
                adjustments.append({"date": d, "rule": "non_negative", "from": y, "to": 0.0})
                y = 0.0

            # cap
            if use_cap and cap_value is not None and y > cap_value:
                rules_hit.append("cap")
                by_rule["cap"] = by_rule.get("cap", 0) + 1
                adjustments.append({"date": d, "rule": "cap", "from": y, "to": float(cap_value)})
                y = float(cap_value)

            # smoothness (rate limit)
            if use_rate and prev is not None and max_change_rate > 0:
                base = abs(prev) if abs(prev) > 1e-9 else 1.0
                allowed = max_change_rate * base
                diff = y - prev
                if abs(diff) > allowed:
                    y_clip = prev + (allowed if diff > 0 else -allowed)
                    rules_hit.append("rate_limit")
                    by_rule["rate_limit"] = by_rule.get("rate_limit", 0) + 1
                    adjustments.append({"date": d, "rule": "rate_limit", "from": y, "to": y_clip})
                    y = y_clip

        corrected.append({"date": d, "value": float(y), "kind": kind})
        violations_series.append({"date": d, "kind": kind, "rules_hit": rules_hit})
        prev = y

    deltas = [abs(a["to"] - a["from"]) for a in adjustments]
    mean_abs = sum(deltas) / len(deltas) if deltas else 0.0
    max_abs = max(deltas) if deltas else 0.0
    changed = len({a["date"] for a in adjustments})
    denom = max(1, len([p for p in series if _is_target_kind(p.get("kind", ""), a2)]))
    ratio = changed / denom if denom else 0.0
    by_rule.pop("spike_clip", None)
    summary = {
        "num_adjusted": len(adjustments),
        "max_abs_adjustment": float(max_abs),
        "mean_abs_adjustment": float(mean_abs),
        "adjusted_ratio": float(ratio),
        "by_rule_counts": by_rule,
        "adjustments": adjustments,
        "violations_series": violations_series,
    }

    return corrected, summary