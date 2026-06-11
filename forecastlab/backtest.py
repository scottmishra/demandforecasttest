"""Backtest harness and accuracy metrics.

Evaluation protocol: a single held-out window of ``horizon`` weeks at the end of
the panel. Every method is fit on weeks ``[0, train_weeks)`` and produces a flat
per-period demand rate for the holdout. We score against the actual holdout
demand, overall and segmented by Syntetos-Boylan quadrant (classified on the
training history only, so the segmentation never sees the holdout).

Metrics are chosen for intermittent demand:

* **WMAPE** — sum|y-f| / sum|y|. Scale-free, robust to zeros (unlike MAPE which
  is undefined on zero actuals).
* **MASE** — mean abs error scaled by the in-sample one-step naive error. <1
  means "better than naive".
* **RMSSE** — the M5 metric; squared-error analogue of MASE.
* **Bias%** — sum(f-y)/sum(y). Croston is known to be positively biased; SBA
  corrects it. Surfacing bias makes that visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import classification
from .models.local import LOCAL_MODELS
from .models.global_xgb import GlobalXGB


def _naive_scale(train: np.ndarray) -> float:
    """In-sample mean absolute one-step naive error (MASE/RMSSE denominator)."""
    if len(train) < 2:
        return float("nan")
    diffs = np.abs(np.diff(train))
    m = diffs.mean()
    return float(m) if m > 0 else float("nan")


def _naive_scale_sq(train: np.ndarray) -> float:
    if len(train) < 2:
        return float("nan")
    diffs = np.diff(train) ** 2
    m = diffs.mean()
    return float(m) if m > 0 else float("nan")


def per_sku_errors(actual: np.ndarray, forecast: np.ndarray, train: np.ndarray) -> dict:
    """Error aggregates for one SKU over its holdout window."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    abs_err = np.abs(actual - forecast)
    sq_err = (actual - forecast) ** 2
    scale = _naive_scale(train)
    scale_sq = _naive_scale_sq(train)
    return {
        "abs_err_sum": float(abs_err.sum()),
        "demand_sum": float(actual.sum()),
        "signed_err_sum": float((forecast - actual).sum()),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(sq_err.mean())),
        "mase": float(abs_err.mean() / scale) if np.isfinite(scale) else np.nan,
        "rmsse": float(np.sqrt(sq_err.mean() / scale_sq)) if np.isfinite(scale_sq) else np.nan,
    }


def _local_forecasts(wide: pd.DataFrame, train_weeks: int, horizon: int) -> dict:
    """Run every local model on every SKU. Returns {model: {sku: forecast_vec}}."""
    out: dict[str, dict] = {name: {} for name in LOCAL_MODELS}
    for sku, hist in zip(wide.index, wide.values):
        train = hist[:train_weeks]
        for name, fn in LOCAL_MODELS.items():
            out[name][sku] = fn(train, horizon)
    return out


def run_backtest(wide: pd.DataFrame, skus: pd.DataFrame, train_weeks: int,
                 horizon: int, include_global: bool = True,
                 global_objectives=("tweedie", "squared")) -> dict:
    """Fit all methods, score them, and return tidy result frames.

    Returns a dict with:
      'per_sku'    — long frame: one row per (model, sku) with errors + quadrant
      'overall'    — per-model aggregate metrics
      'by_quadrant'— per-(model, quadrant) aggregate metrics
      'classes'    — the SKU classification used for segmentation
      'forecasts'  — {model: Series(sku -> rate)} for inspection in the app
    """
    classes = classification.classify_panel(wide, train_weeks=train_weeks)

    # --- generate forecasts -------------------------------------------------
    forecasts: dict[str, pd.Series] = {}
    local = _local_forecasts(wide, train_weeks, horizon)
    for name, per_sku in local.items():
        # store the flat rate (first element) for inspection
        forecasts[name] = pd.Series({s: v[0] for s, v in per_sku.items()})

    global_models = {}
    if include_global:
        for obj in global_objectives:
            label = f"global_{obj}"
            gm = GlobalXGB(horizon=horizon, objective=obj).fit(wide, skus, train_weeks)
            global_models[label] = gm
            forecasts[label] = gm.predict(wide, skus, origin=train_weeks)

    # --- score every method -------------------------------------------------
    rows = []
    for sku, hist in zip(wide.index, wide.values):
        train = hist[:train_weeks]
        actual = hist[train_weeks:train_weeks + horizon]
        quad = classes.loc[sku, "quadrant"]
        for name in local:
            err = per_sku_errors(actual, local[name][sku], train)
            err.update(model=name, sku=sku, quadrant=quad)
            rows.append(err)
        for label, gm in global_models.items():
            fvec = np.full(horizon, float(forecasts[label][sku]))
            err = per_sku_errors(actual, fvec, train)
            err.update(model=label, sku=sku, quadrant=quad)
            rows.append(err)

    per_sku_df = pd.DataFrame(rows)
    overall = _aggregate(per_sku_df, ["model"])
    by_quadrant = _aggregate(per_sku_df, ["model", "quadrant"])

    return {
        "per_sku": per_sku_df,
        "overall": overall,
        "by_quadrant": by_quadrant,
        "classes": classes,
        "forecasts": forecasts,
        "global_models": global_models,
    }


def _aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Aggregate per-SKU errors into the headline metrics."""
    g = df.groupby(by, observed=True)
    agg = g.agg(
        abs_err_sum=("abs_err_sum", "sum"),
        demand_sum=("demand_sum", "sum"),
        signed_err_sum=("signed_err_sum", "sum"),
        mase=("mase", "mean"),
        rmsse=("rmsse", "mean"),
        n_skus=("sku", "nunique"),
    ).reset_index()
    agg["wmape"] = agg["abs_err_sum"] / agg["demand_sum"].replace(0, np.nan)
    agg["bias_pct"] = agg["signed_err_sum"] / agg["demand_sum"].replace(0, np.nan)
    cols = by + ["wmape", "mase", "rmsse", "bias_pct", "n_skus"]
    return agg[cols].sort_values(by + ["wmape"]).reset_index(drop=True)
