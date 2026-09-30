"""Data-quality cleaning for freight-rate data.

Findings (from EDA on 48k train rows, Jan-Oct 2025):
- weight: 300 NaN, 292 negative (sign-entry error), 1204 capped at |47500|.
- market_index: 374 NaN in train (249 in validation). Low within-day noise.
- distance: 48 rows floored at exactly 70.0.
- posted_rate: ~1.4% outliers (x3 or /3). Handled by robust loss, NOT removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHT_CAP = 47_500.0
DIST_FLOOR = 70.0


def clean_weights(df: pd.DataFrame, median_w: float | None = None) -> pd.DataFrame:
    """Fix sign errors, flag missing/capped weights, impute median."""
    out = df.copy()
    w = out["weight"].astype(float)
    out["weight_missing"] = w.isna().astype(int)
    out["weight_was_negative"] = (w < 0).astype(int)
    w = w.abs()
    out["weight_capped"] = (w >= WEIGHT_CAP - 1e-9).astype(int)
    if median_w is None:
        median_w = float(w.median())
    out["weight_clean"] = w.fillna(median_w)
    return out


def daily_market_medians(df: pd.DataFrame) -> pd.Series:
    """Median market_index per calendar date (denoises within-day noise ~0.025)."""
    return df.groupby(df["date"].astype(str))["market_index"].median()


def clean_market_index(
    df: pd.DataFrame,
    daily_med: pd.Series | None = None,
    global_med: float | None = None,
) -> pd.DataFrame:
    """Fill missing market_index with same-day median, fallback to global median."""
    out = df.copy()
    out["mi_missing"] = out["market_index"].isna().astype(int)
    if daily_med is None:
        daily_med = daily_market_medians(out)
    if global_med is None:
        global_med = float(out["market_index"].median())
    day_key = out["date"].astype(str)
    out["market_index"] = out["market_index"].fillna(day_key.map(daily_med))
    out["market_index"] = out["market_index"].fillna(global_med)
    return out


def add_distance_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows floored at the 70-mile minimum."""
    out = df.copy()
    out["dist_floor"] = (out["distance"] <= DIST_FLOOR + 1e-9).astype(int)
    return out


def clean_frame(
    df: pd.DataFrame,
    median_w: float | None = None,
    daily_med: pd.Series | None = None,
    global_med: float | None = None,
) -> pd.DataFrame:
    """Apply all cleaning steps. Fit stats on train, reuse for validation."""
    out = df.copy()
    out = clean_weights(out, median_w=median_w)
    out = clean_market_index(out, daily_med=daily_med, global_med=global_med)
    out = add_distance_flag(out)
    return out
