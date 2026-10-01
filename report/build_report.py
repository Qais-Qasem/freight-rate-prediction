"""Build report/report.pdf (validation, split, December chart). Run: python report/build_report.py"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report"
CHART = ROOT / "scorer_results" / "candidate_december.png"

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(6, 74, 86)
        self.cell(0, 8, "Freight Rate Prediction - Spotter ML Assessment", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(6, 74, 86)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def heading(self, t):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(20, 20, 20)
        self.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT", align="L")

    def b(self, t):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.2, t, new_x="LMARGIN", new_y="NEXT", align="L")

    def mono(self, t, size=7.5, h=4.0):
        self.set_font("Courier", "", size)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, h, t, new_x="LMARGIN", new_y="NEXT", align="L")

pdf = PDF()
pdf.set_auto_page_break(True, 15)
pdf.add_page()

cv = pd.read_csv(ROOT / "outputs" / "cv_results.csv")
mean = cv.groupby("model")[["MAE", "RMSE", "MAPE", "MedAPE"]].mean().round(2)

pdf.heading("1. Data split & validation")
pdf.b(
 "Train: Jan-Oct 2025 (48,000 rows). Final test: Nov-Dec 2025 (12,000 rows, labels hidden). "
 "Because market_index drifts (train mean 1.08 vs validation 0.93) and rates trend by month, "
 "a random split would leak the future and give optimistic scores. We use a temporal "
 "expanding window with 3 folds:\n"
 "Fold1: train Jan-Apr -> test May-Jun\n"
 "Fold2: train Jan-Jun -> test Jul-Aug\n"
 "Fold3: train Jan-Aug -> test Sep-Oct (closest proxy for Nov-Dec).\n"
 "Metrics per fold: MAE / RMSE / MAPE / MedAPE. Model selection on mean MAE/MAPE."
)

pdf.heading("2. Data-quality issues & fixes")
pdf.b(
 "- weight: 300 NaN (flag + median), 292 negative -> abs() (sign-entry error), "
 "1204 capped at 47500 (flag).\n"
 "- market_index: 374 NaN train / 249 val -> same-day median, fallback global median (+ missing flag). "
 "Within-day noise ~0.025 so daily median denoises safely.\n"
 "- distance: 48 rows floored at 70.0 (flag).\n"
 "- posted_rate: ~1.4% outliers (x3 or /3) -> kept; robust L1 loss on log-target handles them.\n"
 "- coordinates are not real geography (e.g. Phoenix shifted); used as lane embeddings, not maps.\n"
 "- 8 unseen validation cities (Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, "
 "San Diego): no city-ID features, coordinates only."
)

pdf.heading("3. Key findings")
pdf.b(
 "- distance <-> rate corr 0.91: the dominant driver.\n"
 "- market_index has a huge weekly cycle (Thu ~1.18, Sun ~0.99) but moves prices only ~+-1%%; "
 "raw index is a trap, daily-median smoothing (mi_smooth) is used.\n"
 "- quote_signal is U-shaped vs $/mile (edges 3.2-3.4, middle 2.1): strong non-linear signal, "
 "qs_dev=|q-median| added.\n"
 "- monthly $/mile drifts 2.10 (Jan) -> 2.33 (Jun) -> 2.24 (Oct): independent time trend + trend/dow features."
)

pdf.heading("4. Model choice")
pdf.b(
 "LightGBM regression_l1 on log1p(posted_rate). GBM captures the U-shaped quote effect and "
 "lane interactions that linear models miss; log-target stabilizes skew; L1 ignores label outliers "
 "(RMSE ~630 for all models is outlier-driven). Coordinates + 3-way equipment one-hot generalize "
 "to new cities."
)

pdf.heading("5. CV results (temporal folds)")
tbl = cv.round(2).to_string(index=False)
pdf.mono(tbl, size=7.5, h=4.0)
pdf.ln(2)
pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 6, "Mean over folds:", new_x="LMARGIN", new_y="NEXT", align="L")
pdf.mono(mean.to_string(), size=8, h=4.5)
pdf.ln(1)
pdf.b(
 "Winner: lgbm-logL1 (mean MAE 115.96, MAPE 4.94) vs ridge-log (118.67 / 5.03), "
 "hgb-logL2 (134.33 / 5.74), naive median $/mile (219.34 / 10.53). "
 "Fold3 (Sep-Oct, best Nov-Dec proxy): lgbm MAE 106.5 / MAPE 4.66. "
 "Final model retrained on all Jan-Oct with a time-tail eval split + early stopping (seed 0)."
)

pdf.heading("6. December chart (fixed lane)")
pdf.b(
 "Lane Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb; only date varies. "
 "december_chart_inputs.csv ships WITHOUT market_index / quote_signal / coordinates, so:\n"
 "- coords: train city medians (both cities exist in train).\n"
 "- market_index + mi_smooth: calendar-day means from validation.csv "
 "(input features covering all Dec dates, not labels).\n"
 "- quote_signal: December median from validation (2.0512).\n"
 "Prediction range ~$819-826 with a small weekly cycle on top of the trend level. "
 "Limitation: 10 months cannot separate trend from seasonality, and holiday December "
 "is out-of-distribution; chart shape is more reliable than absolute level."
)
if CHART.exists():
    pdf.image(str(CHART), x=12, w=186)
else:
    pdf.b("[chart missing: run score.py first]")

pdf.heading("7. Post-baseline experiments (not shipped)")
pdf.b(
 "After freezing the submission we tested CatBoost on the identical 3 folds "
 "and features (catboost 1.2.10, seed 0, time-tail eval split + early stopping):\n"
 "CatBoost log-target + MAE loss: mean MAE 126.2 (folds 138.1 / 121.8 / 118.8).\n"
 "CatBoost log-target + RMSE loss: mean MAE 113.2 (folds 128.9 / 100.5 / 110.3).\n"
 "Lesson: the log transform already converts absolute errors into relative ones "
 "(log y - log yhat = log(y/yhat)) and damps the corrupted labels, so L1 adds "
 "little while being harder to optimize - L1 has a constant gradient (+/-1) and "
 "zero Hessian, which starves the Newton-style leaf splits that symmetric "
 "(oblivious) trees rely on, hence ~1000 slow iterations vs ~300 for RMSE. "
 "CatBoost-RMSE is marginally best (113.2 vs 116.0, ~2.4%), but on a single seed "
 "that gap may be noise, so the validated LGBM-L1 pipeline stays the submission. "
 "A 50/50 LGBM + CatBoost-RMSE average is the natural next step: leaf-wise/L1 "
 "vs symmetric/L2 gives structurally diverse residuals."
)
pdf.b(
 "RMSE decomposition (Fold3, submitted model, 9,523 rows): all rows MAE 106.5 / "
 "RMSE 633.9 / MAPE 4.66%; after dropping only the worst 1.5% absolute errors: "
 "MAE 50.5 / RMSE 74.0 / MAPE 2.32%. The 8 worst rows pair true rates of "
 "13k-20k dollars against sane predictions of 3k-6k (about 2 $/mile on normal "
 "1,400-3,000 mile hauls): corrupted labels (x3-x5), not model errors. "
 "Conclusion: RMSE ~630 is a property of the labels, handled by L1+log "
 "robustness during training; deleting high-error rows would be tuning by "
 "deletion, so the metric is reported honestly as-is."
)

pdf.heading("8. Reproduce")
pdf.b(
 "pip install -r requirements.txt\n"
 "python -m src.validate\n"
 "python -m src.predict\n"
 "python score.py --predictions validation_predictions.csv "
 "--december-predictions data/december_chart_inputs.csv"
)

OUT.mkdir(parents=True, exist_ok=True)
import os as _os
_out_pdf = _os.environ.get("REPORT_PDF", str(OUT / "report.pdf"))
pdf.output(_out_pdf)
print(f"Saved -> {_out_pdf}")
