"""Inventory / purchasing translation of a demand forecast.

A forecast is only useful to a buyer once it is turned into a *quantity to hold*
over a protection interval (the lead time). This module does that translation
and scores it the way purchasing actually cares about:

* lead-time demand error (not per-week error),
* bias,
* the service level / fill rate you achieve, with and without safety stock.

The same functions back both ``scripts/purchasing_analysis.py`` and the
Streamlit "story" page, so there is a single source of truth for the numbers.

Simplified math used throughout (per SKU, protection interval L weeks):

    lead-time demand            D_L      = Σ_{t=1..L} demand_t
    point forecast of D_L       D̂_L     = rate · L          (flat-rate model)
    lead-time demand spread     σ_L      = std of historical L-week demand blocks
    safety stock (service z)    SS       = z · σ_L
    order-up-to / max level     S        = D̂_L + SS
    reorder point               ROP      = S            (periodic review here)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z_SERVICE = {0.50: 0.0, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}


def lead_time_sigma(train: np.ndarray, horizon: int) -> float:
    """Std of non-overlapping L-week demand blocks in the training history."""
    blocks = [train[i:i + horizon].sum()
              for i in range(0, len(train) - horizon + 1, horizon)]
    if len(blocks) > 1:
        return float(np.std(blocks))
    # Fallback: scale weekly std by sqrt(L) if too little history for blocks.
    return float(np.std(train) * np.sqrt(horizon))


def lead_time_frame(wide: pd.DataFrame, rate: pd.Series, classes: pd.DataFrame,
                    horizon: int) -> pd.DataFrame:
    """Per-SKU actual vs forecast lead-time demand, with the spread for safety stock.

    ``rate`` is a sku-indexed Series of the flat per-week demand rate produced by
    a model. The holdout is the final ``horizon`` columns of ``wide``.
    """
    tw = wide.shape[1] - horizon
    rows = []
    for sku, hist in zip(wide.index, wide.values):
        train = hist[:tw]
        rows.append({
            "sku": sku,
            "actual_lt": float(hist[tw:tw + horizon].sum()),
            "fc_lt": float(rate[sku] * horizon),
            "sigma_lt": lead_time_sigma(train, horizon),
            "quadrant": classes.loc[sku, "quadrant"],
        })
    return pd.DataFrame(rows)


def service_metrics(df: pd.DataFrame, service: float = 0.95) -> dict:
    """Accuracy + service KPIs for a lead-time frame at a target service level."""
    a = df["actual_lt"].values
    f = df["fc_lt"].values
    tot = a.sum() or 1.0
    z = Z_SERVICE[service]

    # Order-up-to levels: point forecast only vs forecast + safety stock.
    S_point = np.maximum(np.round(f), 0)
    S_ss = np.maximum(np.round(f + z * df["sigma_lt"].values), 0)

    live = a > 0
    abs_pe = np.abs(f[live] - a[live]) / a[live]

    return {
        "n": int(len(df)),
        "lt_wmape": float(np.abs(a - f).sum() / tot),
        "lt_bias": float((f - a).sum() / tot),
        "within_25pct": float((abs_pe <= 0.25).mean()) if live.any() else float("nan"),
        "within_50pct": float((abs_pe <= 0.50).mean()) if live.any() else float("nan"),
        "fill_point": float(np.minimum(a, S_point).sum() / tot),
        "stockout_point": float((a > S_point).mean()),
        "fill_ss": float(np.minimum(a, S_ss).sum() / tot),
        "stockout_ss": float((a > S_ss).mean()),
        "overstock_ss": float((S_ss.sum() - a.sum()) / tot),
    }


def by_quadrant(df: pd.DataFrame, service: float = 0.95,
                quadrants=("smooth", "erratic", "intermittent", "lumpy")) -> pd.DataFrame:
    """service_metrics for ALL and for each quadrant, as a tidy frame."""
    rows = [{"segment": "ALL", **service_metrics(df, service)}]
    for q in quadrants:
        sub = df[df["quadrant"] == q]
        if len(sub):
            rows.append({"segment": q, **service_metrics(sub, service)})
    return pd.DataFrame(rows).set_index("segment")


def policy_levels(rate: float, sigma_lt: float, horizon: int,
                  service: float = 0.95) -> dict:
    """Min/max (reorder point and order-up-to) levels for one SKU."""
    z = Z_SERVICE[service]
    mean_lt = rate * horizon
    ss = z * sigma_lt
    return {"mean_lt": mean_lt, "safety_stock": ss,
            "reorder_point": mean_lt + ss, "order_up_to": mean_lt + ss}
