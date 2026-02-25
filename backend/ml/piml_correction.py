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
    # default: test_forecast
    return k in ("test_pred", "forecast")


def _rolling_median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    m = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[m])
    return float((s[m - 1] + s[m]) / 2.0)


def _mad(vals: List[float], med: float) -> float:
    if not vals:
        return 0.0
    dev = [abs(v - med) for v in vals]
    return _rolling_median(dev)


def apply_physics_corrections(
    series: List[Dict[str, Any]],
    *,
    physics_mode: str = "full",
    non_negative: bool = True,
    max_change_rate: float = 0.25,
    cap_value: Optional[float] = None,
    apply_to: str = "test_forecast",
    spike_clip: bool = True,
    spike_window: int = 7,
    spike_z: float = 6.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode = (physics_mode or "none").strip().lower()
    a2 = (apply_to or "test_forecast").strip().lower()
    if a2 not in ("forecast", "test_forecast"):
        a2 = "test_forecast"

    # none -> no correction
    if mode == "none":
        out = []
        for p in series or []:
            out.append({"date": p["date"], "value": _safe_float(p["value"]), "kind": p.get("kind", "")})
        summary = {
            "num_adjusted": 0,
            "max_abs_adjustment": 0.0,
            "mean_abs_adjustment": 0.0,
            "adjusted_ratio": 0.0,
            "by_rule_counts": {},
            "adjustments": [],
            "violations_series": [{"date": p["date"], "kind": p.get("kind", ""), "rules_hit": []} for p in out],
        }
        return out, summary

    use_nonneg = bool(non_negative) and mode in ("non_negative", "full", "cap", "smoothness")
    use_cap = (cap_value is not None) and mode in ("cap", "full")
    use_rate = mode in ("smoothness", "full")
    use_spike = bool(spike_clip) and mode in ("full", "smoothness", "cap")
    r = float(max_change_rate) if max_change_rate is not None else 0.0
    capf = float(cap_value) if cap_value is not None else None
    raw_vals: List[float] = [_safe_float(p.get("value")) for p in (series or [])]
    kinds: List[str] = [str(p.get("kind") or "") for p in (series or [])]

    spike_flag = [False] * len(raw_vals)
    if use_spike and spike_window >= 3:
        w = int(spike_window)
        half = w // 2
        for i in range(len(raw_vals)):
            if not _is_target_kind(kinds[i], a2):
                continue
            lo = max(0, i - half)
            hi = min(len(raw_vals), i + half + 1)
            window_vals = [raw_vals[j] for j in range(lo, hi) if j != i]
            if len(window_vals) < 3:
                continue
            med = _rolling_median(window_vals)
            mad = _mad(window_vals, med)
            # Robust z score (avoid 0)
            denom = mad if mad > 1e-9 else 1e-9
            z = abs(raw_vals[i] - med) / denom
            if z >= float(spike_z):
                spike_flag[i] = True

    adjustments: List[Dict[str, Any]] = []
    by_rule: Dict[str, int] = {}
    viol_series: List[Dict[str, Any]] = []
    corrected: List[Dict[str, Any]] = []
    prev: Optional[float] = None

    for i, p in enumerate(series or []):
        d = str(p.get("date"))
        kind = str(p.get("kind") or "")
        y0 = _safe_float(p.get("value"))
        y = float(y0)
        rules_hit: List[str] = []

        if _is_target_kind(kind, a2):
            # spike_clip first (optional)
            if use_spike and spike_flag[i]:
                w = int(spike_window)
                half = w // 2
                lo = max(0, i - half)
                hi = min(len(raw_vals), i + half + 1)
                window_vals = [raw_vals[j] for j in range(lo, hi) if j != i]
                med = _rolling_median(window_vals)
                if med != y:
                    rules_hit.append("spike_clip")
                    by_rule["spike_clip"] = by_rule.get("spike_clip", 0) + 1
                    adjustments.append(
                        {"date": d, "kind": kind, "rule": "spike_clip", "from": y, "to": float(med), "delta": float(med - y)}
                    )
                    y = float(med)

            # non_negative
            if use_nonneg and y < 0:
                rules_hit.append("non_negative")
                by_rule["non_negative"] = by_rule.get("non_negative", 0) + 1
                adjustments.append({"date": d, "kind": kind, "rule": "non_negative", "from": y, "to": 0.0, "delta": float(0.0 - y)})
                y = 0.0

            # cap
            if use_cap and capf is not None and y > capf:
                rules_hit.append("cap")
                by_rule["cap"] = by_rule.get("cap", 0) + 1
                adjustments.append({"date": d, "kind": kind, "rule": "cap", "from": y, "to": float(capf), "delta": float(capf - y)})
                y = float(capf)

            # rate limit
            if use_rate and prev is not None and r > 0:
                base = abs(prev) if abs(prev) > 1e-9 else 1.0
                allowed = r * base
                diff = y - prev
                if abs(diff) > allowed:
                    y_clip = prev + (allowed if diff > 0 else -allowed)
                    rules_hit.append("rate_limit")
                    by_rule["rate_limit"] = by_rule.get("rate_limit", 0) + 1
                    adjustments.append(
                        {"date": d, "kind": kind, "rule": "rate_limit", "from": y, "to": float(y_clip), "delta": float(y_clip - y)}
                    )
                    y = float(y_clip)

        corrected.append({"date": d, "value": float(y), "kind": kind})
        prev = float(y)

        viol_series.append({"date": d, "kind": kind, "rules_hit": rules_hit})

    deltas = [abs(_safe_float(a.get("delta"))) for a in adjustments]
    num_adj = int(len(adjustments))
    max_abs = float(max(deltas)) if deltas else 0.0
    mean_abs = float(sum(deltas) / len(deltas)) if deltas else 0.0

    changed_dates = set()
    for a in adjustments:
        changed_dates.add((a.get("date"), a.get("kind")))
    denom = max(1, len([1 for i in range(len(series or [])) if _is_target_kind(kinds[i], a2)]))
    adj_ratio = float(len(changed_dates) / denom) if denom else 0.0
    summary = {
        "num_adjusted": int(num_adj),
        "max_abs_adjustment": float(max_abs),
        "mean_abs_adjustment": float(mean_abs),
        "adjusted_ratio": float(adj_ratio),
        "by_rule_counts": by_rule,
        "adjustments": adjustments,
        "violations_series": viol_series,
        "apply_to": a2,
    }
    return corrected, summary


