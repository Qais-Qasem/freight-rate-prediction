"""Train final model on all Jan-Oct data, predict validation + December chart.

December problem: december_chart_inputs.csv has only
  pickup, delivery, distance, equipment, weight, date, predicted_rate
i.e. NO coordinates, NO market_index, NO quote_signal.
We fill them and document the assumptions:
  - coords: city medians learned from train (Lexington / Fort Wayne exist).
  - market_index + mi_smooth: calendar-day mean from validation.csv
    (features, not labels - covers every December date).
  - quote_signal: December median from validation (~2.05).

Run:  python -m src.predict   (from repo root)
Outputs:
  validation_predictions.csv  (repo root, scorer-ready)
  data/december_chart_inputs.csv  (filled predicted_rate, scorer-ready)
  outputs/final_model.txt + outputs/december_predictions.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cleaning import clean_frame  # noqa: E402
from src.features import build_features, feature_matrix  # noqa: E402
from src.model import fit_full  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def city_coord_lookup(train: pd.DataFrame) -> dict:
    lat, lon = {}, {}
    for col, la, lo in [("pickup", "pickup_lat", "pickup_lon"),
                        ("delivery", "delivery_lat", "delivery_lon")]:
        g = train.groupby(col)[[la, lo]].median()
        for city, row in g.iterrows():
            lat.setdefault(city, row[la])
            lon.setdefault(city, row[lo])
    return {"lat": lat, "lon": lon}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train_test.csv")
    val = pd.read_csv(DATA / "validation.csv")
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")

    train["date"] = pd.to_datetime(train["date"])
    val["date"] = pd.to_datetime(val["date"])
    dec["date"] = pd.to_datetime(dec["date"])

    # --- fit cleaning/feature stats on TRAIN only ---
    median_w = float(train["weight"].abs().median())
    daily_med_tr = train.groupby(train["date"].astype(str))["market_index"].median()
    global_med = float(train["market_index"].median())
    q_med = float(train["quote_signal"].median())

    train_c = clean_frame(train, median_w, daily_med_tr, global_med)

    # market smooth map: train days from train; Nov-Dec days from validation
    # (validation market_index values are input features, not labels)
    val_c_tmp = clean_frame(val, median_w, daily_med_tr, global_med)
    combined = pd.concat([train_c, val_c_tmp], ignore_index=True)
    full_daily = combined.groupby(
        pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    )["market_index"].median()

    train_f = build_features(train_c, q_med, full_daily)
    X_full = feature_matrix(train_f).to_numpy()
    y_log = np.log1p(train["posted_rate"].to_numpy(float))

    model = fit_full(X_full, y_log, seed=0)
    joblib.dump(
        {"model": model, "median_w": median_w, "global_med": global_med,
         "q_med": q_med, "daily": full_daily},
        OUT / "final_model.joblib",
    )
    model.booster_.save_model(str(OUT / "final_model.txt"))

    # --- validation predictions ---
    val_c = clean_frame(val, median_w, daily_med_tr, global_med)
    val_f = build_features(val_c, q_med, full_daily)
    p_val = np.clip(np.expm1(model.predict(feature_matrix(val_f).to_numpy())), 1.0, None)
    pred = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": p_val.round(2)})
    pred.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"Saved {len(pred)} rows -> validation_predictions.csv")

    # --- december chart inputs ---
    coords = city_coord_lookup(train)
    dec_exp = dec.copy()
    dec_exp["pickup_lat"] = dec_exp["pickup"].map(coords["lat"])
    dec_exp["delivery_lat"] = dec_exp["delivery"].map(coords["lat"])
    dec_exp["pickup_lon"] = dec_exp["pickup"].map(coords["lon"])
    dec_exp["delivery_lon"] = dec_exp["delivery"].map(coords["lon"])
    # market_index per December day from validation daily means
    dec_daily = val_c_tmp.groupby(val_c_tmp["date"].dt.strftime("%Y-%m-%d"))[
        "market_index"
    ].mean()
    dec_exp["market_index"] = dec_exp["date"].dt.strftime("%Y-%m-%d").map(dec_daily)
    dec_exp["market_index"] = dec_exp["market_index"].fillna(global_med)
    q_dec = float(val["quote_signal"].median())
    dec_exp["quote_signal"] = q_dec
    print(f"December imputation: quote_signal={q_dec:.4f} (Dec median from validation)")

    dec_c = clean_frame(dec_exp, median_w, daily_med_tr, global_med)
    dec_c["market_index"] = dec_exp["market_index"]  # keep day-mean, not NaN-fill noise
    dec_c["mi_missing"] = 1  # flag: december file ships without market data
    dec_f = build_features(dec_c, q_med, full_daily)
    # force smoothed December market to the validation day-means
    daykey = dec_f["date"].dt.strftime("%Y-%m-%d")
    dec_f["mi_smooth"] = daykey.map(dec_daily).astype(float).fillna(dec_f["mi_smooth"])

    p_dec = np.clip(np.expm1(model.predict(feature_matrix(dec_f).to_numpy())), 1.0, None)
    dec_out = pd.read_csv(DATA / "december_chart_inputs.csv")
    dec_out["predicted_rate"] = np.round(p_dec, 2)
    dec_out.to_csv(DATA / "december_chart_inputs.csv", index=False)
    pd.DataFrame({"date": dec["date"].dt.strftime("%Y-%m-%d"),
                  "predicted_rate": np.round(p_dec, 2)}).to_csv(
        OUT / "december_predictions.csv", index=False)
    print("Saved filled data/december_chart_inputs.csv (31 rows)")


if __name__ == "__main__":
    main()
