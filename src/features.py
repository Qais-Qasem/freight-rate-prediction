"""Feature engineering for freight-rate prediction.

Design notes:
- City IDs are deliberately NOT used: 8 unseen cities appear in validation
  (Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk,
  San Diego). Coordinates generalize; IDs do not.
- quote_signal is U-shaped vs rate/mile (edges ~3.2-3.4, middle ~2.1),
  so we add qs_dev = |q - median| for the trees to split on.
- market_index has a huge weekly cycle (Thu ~1.18, Sun ~0.99) but moves
  prices only ~+-1%. We smooth it to a daily median to kill within-day noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT_COLS = ["equipment_Dry Van", "equipment_Flatbed", "equipment_Reefer"]
FEATURE_COLS = [
    "distance",
    "logd",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "weight_clean",
    "weight_missing",
    "weight_capped",
    "weight_was_negative",
    "dist_floor",
    "market_index",
    "mi_missing",
    "mi_smooth",
    "quote_signal",
    "qs_dev",
    "dow",
    "weekend",
    "trend",
    "month",
] + EQUIPMENT_COLS


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["dow"] = out["date"].dt.dayofweek.astype(int)
    out["weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["trend"] = (out["date"] - pd.Timestamp("2025-01-01")).dt.days.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    return out


def add_quote_features(df: pd.DataFrame, q_median: float) -> pd.DataFrame:
    out = df.copy()
    out["qs_dev"] = (out["quote_signal"] - q_median).abs()
    return out


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["logd"] = np.log1p(out["distance"].astype(float))
    return out


def add_market_smooth(df: pd.DataFrame, daily_med: pd.Series) -> pd.DataFrame:
    """Map each row to its calendar-day median market index (7-day rolling
    mean applied where a full date range exists; otherwise raw daily median)."""
    out = df.copy()
    day_key = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["mi_smooth"] = day_key.map(daily_med).astype(float)
    out["mi_smooth"] = out["mi_smooth"].fillna(out["market_index"])
    return out


def one_hot_equipment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EQUIPMENT_COLS:
        out[col] = 0
    mapping = {
        "Dry Van": "equipment_Dry Van",
        "Flatbed": "equipment_Flatbed",
        "Reefer": "equipment_Reefer",
    }
    for eq, col in mapping.items():
        out.loc[out["equipment"] == eq, col] = 1
    return out


def build_features(
    df: pd.DataFrame,
    q_median: float,
    daily_med: pd.Series,
) -> pd.DataFrame:
    """Full pipeline: time + quote + geo + market-smooth + equipment."""
    out = add_time_features(df)
    out = add_quote_features(out, q_median)
    out = add_geo_features(out)
    out = add_market_smooth(out, daily_med)
    out = one_hot_equipment(out)
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    return df[FEATURE_COLS].astype(float)
