"""Temporal validation with an expanding window.

Folds (mimic Nov-Dec generalization from past data):
  Fold1: train Jan-Apr  -> test May-Jun
  Fold2: train Jan-Jun  -> test Jul-Aug
  Fold3: train Jan-Aug  -> test Sep-Oct

Compares Naive (median $/mile per equipment) vs Ridge-log vs HGB-L2 vs
LightGBM-L1 on MAE / RMSE / MAPE / MedAPE. Saves outputs/cv_results.csv.

Run:  python -m src.validate   (from repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cleaning import clean_frame  # noqa: E402
from src.features import FEATURE_COLS, build_features, feature_matrix  # noqa: E402
from src.model import fit_predict_log_l1  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

FOLDS = [
    ("2025-01-01", "2025-05-01", "2025-07-01"),
    ("2025-01-01", "2025-07-01", "2025-09-01"),
    ("2025-01-01", "2025-09-01", "2025-11-01"),
]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(np.clip(p, 1.0, None), float)
    ae = np.abs(p - y)
    ape = ae / np.maximum(y, 1.0)
    return {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt(((p - y) ** 2).mean())),
        "MAPE": float(ape.mean() * 100),
        "MedAPE": float(np.median(ape) * 100),
    }


def naive_predict(dtr: pd.DataFrame, dte: pd.DataFrame) -> np.ndarray:
    med = (dtr["posted_rate"] / dtr["distance"]).groupby(dtr["equipment"]).median()
    global_med = float((dtr["posted_rate"] / dtr["distance"]).median())
    rpm = dte["equipment"].map(med).fillna(global_med).to_numpy(float)
    return dte["distance"].to_numpy(float) * rpm


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "train_test.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    rows = []
    for i, (s, cut, end) in enumerate(FOLDS, 1):
        s, cut, end = pd.Timestamp(s), pd.Timestamp(cut), pd.Timestamp(end)
        dtr = df[(df["date"] >= s) & (df["date"] < cut)].copy()
        dte = df[(df["date"] >= cut) & (df["date"] < end)].copy()

        # cleaning stats fit on fold-train only
        median_w = float(dtr["weight"].abs().median())
        daily_med = dtr.groupby(dtr["date"].astype(str))["market_index"].median()
        global_med = float(dtr["market_index"].median())
        q_med = float(dtr["quote_signal"].median())

        dtr_c = clean_frame(dtr, median_w, daily_med, global_med)
        # NOTE: test fold uses its OWN daily medians (daily_med=None) because the
        # fold-train calendar never contains the test dates, so the map would miss
        # every row and silently fall back to global_med. market_index is an input
        # feature (not the label), so same-day peer medians are legitimate.
        dte_c = clean_frame(dte, median_w, None, global_med)
        # smooth maps: fold-train calendar for train rows; combined for test rows
        full_daily = pd.concat([dtr_c, dte_c]).groupby(
            pd.to_datetime(pd.concat([dtr_c, dte_c])["date"]).dt.strftime("%Y-%m-%d")
        )["market_index"].median()
        dtr_f = build_features(dtr_c, q_med, full_daily)
        dte_f = build_features(dte_c, q_med, full_daily)
        Xtr = feature_matrix(dtr_f).to_numpy()
        Xte = feature_matrix(dte_f).to_numpy()
        ytr = dtr["posted_rate"].to_numpy(float)
        yte = dte["posted_rate"].to_numpy(float)

        # 1) naive
        rows.append({"fold": i, "model": "naive", **metrics(yte, naive_predict(dtr, dte))})

        # 2) ridge on log
        ridge = Ridge(alpha=1.0).fit(Xtr, np.log1p(ytr))
        rows.append(
            {"fold": i, "model": "ridge-log", **metrics(yte, np.expm1(ridge.predict(Xte)))}
        )

        # 3) HistGradientBoosting (L2 on log) as a second reference
        hgb = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=400,
            learning_rate=0.06,
            l2_regularization=5.0,
            early_stopping=True,
            random_state=0,
        ).fit(Xtr, np.log1p(ytr))
        rows.append(
            {"fold": i, "model": "hgb-logL2",
             **metrics(yte, np.expm1(hgb.predict(Xte)))}
        )

        p = fit_predict_log_l1(Xtr, np.log1p(ytr), Xte, seed=0)
        rows.append({"fold": i, "model": "lgbm-logL1", **metrics(yte, p)})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cv_results.csv", index=False)
    print(res.to_string(index=False))
    print("\nMean over folds:")
    print(res.groupby("model")[["MAE", "RMSE", "MAPE", "MedAPE"]].mean().round(2).to_string())
    print(f"\nSaved -> {OUT / 'cv_results.csv'}")


if __name__ == "__main__":
    main()
