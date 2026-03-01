from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

SUPPORTED_HORIZONS_MONTHLY = [3, 6, 12, 24]
SUPPORTED_HORIZONS_DAILY = [30, 60, 90]
SUPPORTED_HORIZONS_YEARLY = [5, 10, 20, 30]


@dataclass
class CleanResult:
    time_col: str
    target_col: str
    feature_cols: List[str]
    scale_detected: str
    scale_used: str
    mode_detected: str
    df_clean: pd.DataFrame
    stats: Dict[str, Any]
    warnings: List[str]


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


def _enforce_format(df: pd.DataFrame, target_override: Optional[str] = None) -> Tuple[str, str, List[str]]:
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns: date/time + target.")

    time_col = str(df.columns[0])
    if target_override:
        if target_override not in df.columns:
            raise ValueError(f"target_col '{target_override}' not found in CSV header.")
        target_col = str(target_override)
    else:
        target_col = str(df.columns[-1])

    if time_col == target_col:
        raise ValueError("Invalid format: time column and target column cannot be the same.")

    feature_cols = [str(c) for c in df.columns[1:-1] if str(c) not in (time_col, target_col)]
    return time_col, target_col, feature_cols


def parse_time_column(df: pd.DataFrame, time_col: str) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    s = df[time_col]
    empty_mask = s.isna() | s.astype(str).str.strip().eq("")
    non_empty = ~empty_mask
    s_non_empty = s.where(non_empty, None)
    parsed: pd.Series
    if pd.api.types.is_numeric_dtype(s_non_empty):
        num = pd.to_numeric(s_non_empty, errors="coerce")
        looks_like_year = num.notna() & (num >= 1800) & (num <= 2200) & ((num % 1) == 0)
        if looks_like_year.sum() >= max(2, int(0.8 * num.notna().sum())):
            year_str = num.round().astype("Int64").astype(str)
            parsed = pd.to_datetime(year_str.where(non_empty, None), errors="coerce", format="%Y", utc=False)
        else:
            parsed = pd.to_datetime(s_non_empty, errors="coerce", utc=False)
    else:
        parsed = pd.to_datetime(s_non_empty, errors="coerce", utc=False)

    if parsed[non_empty].isna().any():
        raise ValueError("Invalid date format in first column. Use YYYY, YYYY-MM, or YYYY-MM-DD.")

    out = df.copy()
    out[time_col] = parsed
    dropped = int(empty_mask.sum())
    if dropped > 0:
        warnings.append(f"Dropped {dropped} rows with missing timestamps in the first column.")

    out = out.dropna(subset=[time_col]).sort_values(time_col)

    if out[time_col].duplicated().any():
        raise ValueError(
            "Duplicated dates detected (likely long-format / multi-entity data). "
            "Please pivot/filter to a single time series before upload."
        )

    return out, warnings


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


def _check_daily_continuity(index: pd.DatetimeIndex) -> None:
    if len(index) < 2:
        return
    diffs = pd.Series(index).diff().dropna()
    gaps = (diffs / np.timedelta64(1, "D")).astype(float)
    if (gaps > 1.0).any():
        max_gap = float(gaps.max())
        raise ValueError(
            f"Daily dates are not continuous (max gap: {max_gap:.0f} days). "
            "Please fill missing dates or provide monthly data."
        )


def _ensure_numeric_columns(df: pd.DataFrame, cols: List[str], kind: str) -> None:
    bad: List[str] = []
    for c in cols:
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().sum() == 0:
            bad.append(str(c))
    if bad:
        raise ValueError(
            f"{kind} columns must be numeric. Invalid columns: {', '.join(bad)}. "
            "Please remove text/categorical columns or convert them to numeric."
        )


def _compute_basic_time_stats(df: pd.DataFrame, time_col: str) -> Dict[str, Any]:
    if df.shape[0] == 0:
        return {"n_points": 0}

    start_ts = df[time_col].iloc[0]
    end_ts = df[time_col].iloc[-1]
    span_days = float((end_ts - start_ts) / np.timedelta64(1, "D")) if end_ts >= start_ts else 0.0
    return {
        "n_points": int(df.shape[0]),
        "start_date": start_ts.date().isoformat(),
        "end_date": end_ts.date().isoformat(),
        "span_days": span_days,
    }


def clean_and_normalize(
    df_raw: pd.DataFrame,
    time_col: str,
    target_col: str,
    feature_cols: List[str],
    scale_detected: str,
    parse_warnings: Optional[List[str]] = None,
) -> CleanResult:
    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    if parse_warnings:
        warnings.extend(parse_warnings)

    df = df_raw.copy()
    _ensure_numeric_columns(df, [target_col], "Target")
    _ensure_numeric_columns(df, feature_cols, "Feature")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rows_total = int(df.shape[0])
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

    scale_used = (scale_detected or "monthly").strip().lower()
    if scale_used not in ("daily", "monthly", "yearly"):
        scale_used = "monthly"

    if scale_detected == "daily":
        _check_daily_continuity(df.index)
        warnings.append("Daily data detected. Kept daily frequency (no resampling).")
    elif scale_detected == "monthly":
        warnings.append("Monthly data detected. Kept monthly frequency (no resampling).")
    else:
        warnings.append("Yearly data detected. Kept yearly frequency (no resampling).")

    if df.shape[0] < 2:
        raise ValueError("Not enough valid time points after preprocessing (need at least 2 points).")

    target_missing = _missing_rate(df[target_col])
    stats["target_missing_rate"] = target_missing
    if target_missing > 0.2:
        raise ValueError("Too many missing values in target column (>20%). Please clean your dataset and re-upload.")
    if target_missing > 0:
        df[target_col] = df[target_col].interpolate(limit_direction="both")
        warnings.append("Missing target values were interpolated (limit <= 20%).")

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

        if any(_missing_rate(df[c]) > 0 for c in feature_cols):
            df[feature_cols] = df[feature_cols].ffill().bfill()
            warnings.append("Missing feature values were forward/back filled (limit <= 20%).")

    df = df.dropna(subset=[target_col])
    rows_valid = int(df.shape[0])
    rows_invalid = rows_total - rows_valid
    tmp = df.reset_index()
    stats.update(_compute_basic_time_stats(tmp, time_col))
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
    _ = dataset_name
    time_col, target_col, feature_cols = _enforce_format(df_raw, target_override=target_override)
    df_parsed, parse_warnings = parse_time_column(df_raw, time_col)
    scale_detected = detect_time_scale(df_parsed, time_col)
    clean = clean_and_normalize(
        df_raw=df_parsed,
        time_col=time_col,
        target_col=target_col,
        feature_cols=feature_cols,
        scale_detected=scale_detected,
        parse_warnings=parse_warnings,
    )

    points = dataframe_to_points(clean.df_clean, clean.target_col, clean.feature_cols)
    if len(points) < 2:
        raise ValueError("Not enough points after preprocessing (need at least 2 points).")

    return clean, points


def get_supported_horizons(scale_used: str) -> List[int]:
    scale = (scale_used or "monthly").strip().lower()
    if scale == "daily":
        return SUPPORTED_HORIZONS_DAILY[:]
    if scale == "yearly":
        return SUPPORTED_HORIZONS_YEARLY[:]
    return SUPPORTED_HORIZONS_MONTHLY[:]


def validate_horizon(horizon: int, n_points: int, *, scale_used: str = "monthly") -> None:
    scale = (scale_used or "monthly").strip().lower()
    supported = get_supported_horizons(scale)

    if horizon not in supported:
        raise ValueError(f"horizon must be one of: {', '.join(str(x) for x in supported)}")

    if scale == "monthly":
        if n_points < 6:
            raise ValueError("At least 6 monthly points are required to create a forecast.")
        max_allowed = int(max(3, np.floor(n_points * 0.5)))
        if horizon > max_allowed:
            allowed = [h for h in supported if h <= max_allowed]
            suggest = allowed[-1] if allowed else supported[0]
            raise ValueError(
                f"horizon too large for available history. "
                f"Got horizon={horizon} months, history={n_points}. "
                f"Try {suggest} (or <= {max_allowed})."
            )
        return

    if scale == "daily":
        if n_points < 30:
            raise ValueError("At least 30 daily points are required to create a forecast.")
        max_allowed = int(max(30, np.floor(n_points * 0.5)))
        if horizon > max_allowed:
            allowed = [h for h in supported if h <= max_allowed]
            suggest = allowed[-1] if allowed else supported[0]
            raise ValueError(
                f"horizon too large for available daily history. "
                f"Got horizon={horizon} days, history={n_points} days. "
                f"Try {suggest} (or <= {max_allowed})."
            )
        return

    if n_points < 30:
        raise ValueError("At least 30 yearly points are required to create a forecast.")
    max_allowed = int(max(5, np.floor(n_points * 0.5)))
    if horizon > max_allowed:
        allowed = [h for h in supported if h <= max_allowed]
        suggest = allowed[-1] if allowed else supported[0]
        raise ValueError(
            f"horizon too large for available yearly history. "
            f"Got horizon={horizon} years, history={n_points} years. "
            f"Try {suggest} (or <= {max_allowed})."
        )


def validate_evaluation(evaluation: Dict[str, Any] | None, *, n_points: int, scale_used: str = "monthly") -> Dict[str, Any]:
    if not evaluation:
        return {"enabled": True, "split": {"mode": "ratio", "test_ratio": 0.2}}
    enabled = bool(evaluation.get("enabled", True))
    if not enabled:
        return {"enabled": False, "split": None}

    scale = (scale_used or "monthly").strip().lower()
    min_points = 30 if scale in ("daily", "yearly") else 6
    if n_points < min_points:
        raise ValueError(f"Not enough points for evaluation (need >= {min_points}).")

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