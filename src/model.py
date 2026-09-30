"""Model: LightGBM (L1) on log-target.

Why this model:
- Tabular, 48k rows, mixed numeric/categorical -> GBM beats linear & MLP.
- log1p(posted_rate) target: rates are positive & right-skewed; log stabilizes
  variance and turns multiplicative distance effects additive.
- L1 (absolute error) on log-target: ~1.4% of posted_rate labels look
  corrupted (x3 or /3). L1 ignores them; L2/RMSE would chase them.
- Coordinates instead of city IDs: generalizes to 8 unseen validation cities.
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb


def make_lgbm(seed: int = 0, n_estimators: int = 2000) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=40,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l2=5.0,
        verbose=-1,
        n_jobs=-1,
        random_state=seed,
    )


def _time_eval_split(n: int, frac: float = 0.1):
    n_eval = max(500, int(n * frac))
    return np.arange(n - n_eval), np.arange(n - n_eval, n)


def fit_predict_log_l1(
    X_tr: np.ndarray,
    y_tr_log: np.ndarray,
    X_te: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Train on log-target with a time-tail eval split + early stopping."""
    idx_tr, idx_ev = _time_eval_split(len(X_tr))
    model = make_lgbm(seed=seed)
    model.fit(
        X_tr[idx_tr],
        y_tr_log[idx_tr],
        eval_set=[(X_tr[idx_ev], y_tr_log[idx_ev])],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return np.expm1(model.predict(X_te))


def fit_full(X: np.ndarray, y_log: np.ndarray, seed: int = 0) -> lgb.LGBMRegressor:
    idx_tr, idx_ev = _time_eval_split(len(X))
    model = make_lgbm(seed=seed)
    model.fit(
        X[idx_tr],
        y_log[idx_tr],
        eval_set=[(X[idx_ev], y_log[idx_ev])],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return model
