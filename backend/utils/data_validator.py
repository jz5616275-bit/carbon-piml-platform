def validate_disturbance(
    disturbance: Dict[str, Any] | None,
    *,
    mode_used: str,
    feature_cols: List[str],
) -> Dict[str, Any]:
    if not disturbance:
        return {"enabled": False, "mode": mode_used, "global_pct": None, "feature_pct": None}

    enabled = bool(disturbance.get("enabled", False))
    if not enabled:
        return {"enabled": False, "mode": mode_used, "global_pct": None, "feature_pct": None}

    if mode_used == "basic":
        gp = disturbance.get("global_pct", 0.0)
        try:
            gp = float(gp)
        except Exception:
            raise ValueError("disturbance.global_pct must be a number")

        # Keep the range sane for a demo; avoid negative multiplier.
        if gp <= -0.95:
            raise ValueError("disturbance.global_pct too small (must be > -0.95)")
        return {"enabled": True, "mode": "basic", "global_pct": gp, "feature_pct": None}

    # advanced
    feature_pct = disturbance.get("feature_pct") or {}
    if not isinstance(feature_pct, dict):
        raise ValueError("disturbance.feature_pct must be an object (feature -> pct)")

    normalized: Dict[str, float] = {}
    allowed = set(feature_cols or [])
    
    for k, v in feature_pct.items():
        if k not in allowed:
            raise ValueError(f"disturbance.feature_pct contains unknown feature: {k}")
        try:
            fv = float(v)
        except Exception:
            raise ValueError(f"disturbance.feature_pct[{k}] must be a number")
        if fv <= -0.95:
            raise ValueError(f"disturbance.feature_pct[{k}] too small (must be > -0.95)")
        normalized[k] = fv
    return {"enabled": True, "mode": "advanced", "global_pct": None, "feature_pct": normalized}
