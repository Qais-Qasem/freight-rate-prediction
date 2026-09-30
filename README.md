# Freight Rate Prediction — Spotter ML Assessment

LightGBM (L1) on `log1p(posted_rate)` with temporal validation.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Data layout

Place the assessment files under `data/` (already done):

```
data/train_test.csv
data/validation.csv
data/december_chart_inputs.csv
data/validation_predictions_template.csv
```

## Run

```bash
python -m src.validate     # 3-fold temporal CV -> outputs/cv_results.csv
python -m src.predict      # trains on full Jan-Oct, writes predictions
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Outputs:

- `validation_predictions.csv` — 12,000 rows `load_id,predicted_rate` (repo root, scorer-ready)
- `data/december_chart_inputs.csv` — 31 filled rows (scorer-ready)
- `scorer_results/candidate_december.png` — fixed December chart for the report
- `outputs/cv_results.csv`, `outputs/december_predictions.csv`, `outputs/final_model.txt`

## Method (summary)

- **Split:** temporal expanding window — train Jan–Apr→test May–Jun, Jan–Jun→Jul–Aug, Jan–Aug→Sep–Oct. Never random (leaks future + market regime).
- **Cleaning:** `abs(weight)`, median impute + flags; market NaN → same-day median; distance floor flag at 70; outliers kept (robust L1 loss handles them).
- **Features:** distance + logd, coordinates (no city IDs → 8 unseen val cities), weight_clean + 3 flags, market_index + mi_missing + mi_smooth (daily median), quote_signal + qs_dev=|q−median| (U-shaped), dow/weekend/trend/month, equipment one-hot (3).
- **Model:** LightGBM `regression_l1` on log-target, time-tail eval split + early stopping. Final model trains on all Jan–Oct.
- **December:** coords from train city medians; market per-day means from `validation.csv` (input features, cover all Dec dates); quote = December median (~2.05). Documented in `report/report.pdf`.

See `report/report.pdf` and `notebooks/01_eda.ipynb` for details.
