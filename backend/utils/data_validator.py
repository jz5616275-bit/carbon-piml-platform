from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

SUPPORTED_HORIZONS = {3, 6, 12, 24}


@dataclass
class CleanResult:
    time_col: str
    target_col: str
    feature_cols: List[str]
    scale_detected: str  # daily | monthly | yearly
    scale_used: str      # monthly (after resample if needed)
    mode_detected: str   # basic | advanced
    df_clean: pd.DataFrame
    stats: Dict[str, Any]
    warnings: List[str]


def detect_time_column(df: pd.DataFrame) -> str:
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns (time + at least one numeric column).")

    candidates = {str(c).lower().strip(): str(c) for c in df.columns}
    for name in ("date", "time", "timestamp", "month"):
        if name in candidates:
            return candidates[name]

    return str(df.columns[0])


def parse_time_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    s = df[time_col]
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)
    non_empty = s.astype(str).str.strip().ne("")
    if parsed[non_empty].isna().any():
        raise ValueError("Invalid date format in time column. Please use YYYY-MM or YYYY-MM-DD formats.")

    out = df.copy()
    out[time_col] = parsed
    out = out.sort_values(time_col)
    return out


def detect_time_scale(df: pd.DataFrame, time_col: str) -> str:
    ts = df[time_col].dropna().sort_values()
    if len(ts) < 2:
        raise ValueError("Not enough time points (need at least 2 rows).")

    diffs = ts.diff().dropna()
    median_days = diffs.median() / np.timedelta64(1, "D")

    if median_days <= 2:
        return "daily"
    if median_days <= 45:
        return "monthly"
    return "yearly"


def infer_target_column(df: pd.DataFrame, time_col: str, target_override: Optional[str] = None) -> str:
    if target_override:
        if target_override not in df.columns:
            raise ValueError(f"target_col '{target_override}' not found in CSV header.")
        return target_override

    candidates = [c for c in df.columns if c != time_col]
    if not candidates:
        raise ValueError("No candidate target columns found.")

    numeric_cols: List[str] = []
    for c in candidates:
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().sum() > 0:
            numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError("No numeric emission column found.")

    preferred_keywords = ("emission", "emissions", "co2", "carbon")
    lowered = {c: str(c).lower() for c in numeric_cols}
    for kw in preferred_keywords:
        for c in numeric_cols:
            if kw in lowered[c]:
                return c

    return numeric_cols[0]


def infer_feature_columns(df: pd.DataFrame, time_col: str, target_col: str) -> List[str]:
    return [c for c in df.columns if c not in (time_col, target_col)]


def _missing_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 1.0
    return float(series.isna().sum() / len(series))


def _column_stats(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for c in cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        out[str(c)] = {"min": float(s.min()), "max": float(s.max()), "last": float(s.iloc[-1])}
    return out


def clean_and_normalize(
    df_raw: pd.DataFrame,
    time_col: str,
    target_col: str,
    feature_cols: List[str],
    scale_detected: str,
) -> CleanResult:
    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    if scale_detected == "yearly":
        raise ValueError("Yearly data is not supported. Please provide daily or monthly data.")

    df = df_raw.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rows_total = int(df.shape[0])
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col).set_index(time_col)
    scale_used = "monthly"
    if scale_detected == "daily":
        warnings.append("Daily data detected. Automatically resampled to monthly (mean).")
        df = df.resample("M").mean(numeric_only=True)

    if df.shape[0] < 2:
        raise ValueError("Not enough valid time points after preprocessing (need at least 2).")

    target_missing = _missing_rate(df[target_col])
    stats["target_missing_rate"] = target_missing
    if target_missing > 0.2:
        raise ValueError("Too many missing values in target column (>20%). Please clean your dataset and re-upload.")
    if target_missing > 0:
        df[target_col] = df[target_col].interpolate(limit_direction="both")

    mode_detected = "advanced" if len(feature_cols) > 0 else "basic"
    stats["feature_cols"] = feature_cols

    if feature_cols:
        bad_feature_cols: List[str] = []
        for c in feature_cols:
            mr = _missing_rate(df[c])
            stats[f"feature_missing_rate__{c}"] = mr
            if mr > 0.2:
                bad_feature_cols.append(c)

        if bad_feature_cols:
            raise ValueError("Too many missing values in feature columns (>20%): " + ", ".join(bad_feature_cols))
        df[feature_cols] = df[feature_cols].ffill().bfill()

    df = df.dropna(subset=[target_col])
    rows_valid = int(df.shape[0])
    rows_invalid = rows_total - rows_valid

    stats.update(
        {
            "rows_total": rows_total,
            "rows_valid": rows_valid,
            "rows_invalid": rows_invalid,
            "scale_detected": scale_detected,
            "scale_used": scale_used,
            "mode_detected": mode_detected,
        }
    )

    stats["target_stats"] = _column_stats(df, [target_col])
    stats["feature_stats"] = _column_stats(df, feature_cols)

    return CleanResult(
        time_col=time_col,
        target_col=target_col,
        feature_cols=feature_cols,
        scale_detected=scale_detected,
        scale_used=scale_used,
        mode_detected=mode_detected,
        df_clean=df,
        stats=stats,
        warnings=warnings,
    )


def dataframe_to_points(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for ts, row in df.iterrows():
        date_iso = ts.date().isoformat()
        y = float(row[target_col])
        feats: Dict[str, Any] = {}
        for c in feature_cols:
            v = row.get(c)
            feats[str(c)] = None if pd.isna(v) else float(v)
        points.append({"date": date_iso, "y": y, "features": feats})
    return points


def validate_and_prepare_upload(
    df_raw: pd.DataFrame,
    dataset_name: str,
    target_override: Optional[str] = None,
) -> Tuple[CleanResult, List[Dict[str, Any]]]:
    time_col = detect_time_column(df_raw)
    df_parsed = parse_time_column(df_raw, time_col)
    scale_detected = detect_time_scale(df_parsed, time_col)
    target_col = infer_target_column(df_parsed, time_col, target_override=target_override)
    feature_cols = infer_feature_columns(df_parsed, time_col, target_col)
    clean = clean_and_normalize(
        df_raw=df_parsed,
        time_col=time_col,
        target_col=target_col,
        feature_cols=feature_cols,
        scale_detected=scale_detected,
    )

    points = dataframe_to_points(clean.df_clean, clean.target_col, clean.feature_cols)
    if len(points) < 2:
        raise ValueError("Not enough points after preprocessing (need at least 2 monthly points).")

    return clean, points


def validate_horizon(horizon: int, n_points: int) -> None:
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError("horizon_months must be one of: 3, 6, 12, 24")

    if n_points < 6:
        raise ValueError("At least 6 monthly points are required to create a forecast.")

    max_allowed = int(max(3, np.floor(n_points * 0.5)))
    if horizon > max_allowed:
        raise ValueError(
            f"horizon_months too large for available history. "
            f"Got horizon={horizon}, history={n_points}. Try horizon <= {max_allowed}."
        )


def validate_evaluation(evaluation: Dict[str, Any] | None, *, n_points: int) -> Dict[str, Any]:
    # Default: enabled evaluation with 20% holdout
    if not evaluation:
        return {"enabled": True, "split": {"mode": "ratio", "test_ratio": 0.2}}

    enabled = bool(evaluation.get("enabled", True))
    if not enabled:
        return {"enabled": False, "split": None}

    if n_points < 6:
        raise ValueError("Not enough points for evaluation (need >= 6).")

    split = evaluation.get("split") or {}
    mode = str(split.get("mode") or "ratio").strip().lower()

    if mode in ("lastn", "last12"):
        k_default = 12 if mode == "last12" else 6
        k = int(split.get("test_points", k_default))
        if k < 2:
            k = 2
        if n_points <= k:
            raise ValueError("Not enough points for lastN split.")
        return {"enabled": True, "split": {"mode": "lastn", "test_points": k}}

    if mode == "ratio":
        r = split.get("test_ratio", 0.2)
        try:
            r = float(r)
        except Exception:
            raise ValueError("evaluation.split.test_ratio must be a number")
        if r <= 0 or r >= 0.8:
            raise ValueError("evaluation.split.test_ratio must be in (0, 0.8)")
        k = int(max(2, np.floor(n_points * r)))
        if n_points <= k:
            raise ValueError("Not enough points for ratio split.")
        return {"enabled": True, "split": {"mode": "ratio", "test_ratio": r}}

    raise ValueError("evaluation.split.mode must be 'ratio' or 'lastn' (or legacy 'last12')")


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

        if gp <= -0.95:
            raise ValueError("disturbance.global_pct too small (must be > -0.95)")

        return {"enabled": True, "mode": "basic", "global_pct": gp, "feature_pct": None}

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