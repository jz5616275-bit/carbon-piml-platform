from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
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

    candidates = {c.lower().strip(): c for c in df.columns}
    for name in ("date", "time", "timestamp", "month"):
        if name in candidates:
            return candidates[name]

    # fallback: first column
    return str(df.columns[0])


def parse_time_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    s = df[time_col]

    # allow strings/numbers; pandas handles multiple formats
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
    # Convert to days
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

    # Candidate numeric columns excluding time col
    candidates = [c for c in df.columns if c != time_col]
    if not candidates:
        raise ValueError("No candidate target columns found.")

    # Try to coerce numeric for all candidates
    numeric_cols = []
    for c in candidates:
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().sum() > 0:
            numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError("No numeric emission column found.")

    preferred_keywords = ("emission", "emissions", "co2", "carbon")
    lowered = {c: c.lower() for c in numeric_cols}
    for kw in preferred_keywords:
        for c in numeric_cols:
            if kw in lowered[c]:
                return c

    return numeric_cols[0]


def infer_feature_columns(df: pd.DataFrame, time_col: str, target_col: str) -> List[str]:
    cols = [c for c in df.columns if c not in (time_col, target_col)]
    return cols


def _missing_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 1.0
    return float(series.isna().sum() / len(series))


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

    # Coerce numeric target
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rows_total = int(df.shape[0])

    # Remove rows where time is missing 
    df = df.dropna(subset=[time_col])

    # Set index for resampling / sorting
    df = df.sort_values(time_col).set_index(time_col)

    # Resample if daily
    scale_used = "monthly"
    if scale_detected == "daily":
        warnings.append("Daily data detected. Automatically resampled to monthly (mean).")
        df = df.resample("M").mean(numeric_only=True)

    if df.shape[0] < 2:
        raise ValueError("Not enough valid time points after preprocessing (need at least 2).")

    # Missing target handling
    target_missing = _missing_rate(df[target_col])
    stats["target_missing_rate"] = target_missing
    if target_missing > 0.2:
        raise ValueError("Too many missing values in target column (>20%). Please clean your dataset and re-upload.")
    if target_missing > 0:
        df[target_col] = df[target_col].interpolate(limit_direction="both")

    # Feature handling
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
            raise ValueError(
                "Too many missing values in feature columns (>20%): " + ", ".join(bad_feature_cols)
            )

        # Fill gaps
        df[feature_cols] = df[feature_cols].ffill().bfill()

    # Drop any remaining rows where target is still NaN
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
            feats[c] = None if pd.isna(v) else float(v)
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
