"""Syntetos-Boylan demand classification.

Each series is described by two statistics computed on its *training* history:

* ADI  — average inter-demand interval = (number of periods) / (number of
  nonzero-demand periods). High ADI means sporadic demand.
* CV^2 — squared coefficient of variation of the *nonzero* demand sizes. High
  CV^2 means variable spike sizes.

The standard cut points (Syntetos, Boylan & Croston, 2005) partition the plane:

                 CV^2 < 0.49        CV^2 >= 0.49
    ADI < 1.32   smooth             erratic
    ADI >= 1.32  intermittent       lumpy

Routing then follows: smooth/erratic -> conventional methods are adequate;
intermittent/lumpy -> Croston-family (or a global model). This is exactly the
"classify, then route" recommendation we want the app to make visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ADI_CUT = 1.32
CV2_CUT = 0.49


def adi(series: np.ndarray) -> float:
    """Average inter-demand interval. Inf if the series is all zeros."""
    series = np.asarray(series, dtype=float)
    nonzero = np.count_nonzero(series)
    if nonzero == 0:
        return float("inf")
    return float(len(series) / nonzero)


def cv2(series: np.ndarray) -> float:
    """Squared CV of the nonzero demand sizes. 0 if <2 demand events."""
    series = np.asarray(series, dtype=float)
    sizes = series[series > 0]
    if sizes.size < 2 or sizes.mean() == 0:
        return 0.0
    return float((sizes.std(ddof=0) / sizes.mean()) ** 2)


def classify_one(series: np.ndarray) -> dict:
    a = adi(series)
    c = cv2(series)
    if not np.isfinite(a):
        quadrant = "no_demand"
    elif a < ADI_CUT and c < CV2_CUT:
        quadrant = "smooth"
    elif a < ADI_CUT and c >= CV2_CUT:
        quadrant = "erratic"
    elif a >= ADI_CUT and c < CV2_CUT:
        quadrant = "intermittent"
    else:
        quadrant = "lumpy"
    return {
        "adi": a,
        "cv2": c,
        "quadrant": quadrant,
        "n_events": int(np.count_nonzero(series)),
    }


# Which model family the classifier recommends for each quadrant. The global
# Tweedie model is offered everywhere as the "borrow strength" option; the
# point of the experiment is to show it is the best *default* across quadrants.
RECOMMENDED = {
    "smooth": "conventional (SES / ETS) — patterns are learnable locally",
    "erratic": "conventional + safety stock; global model helps with size variance",
    "intermittent": "Croston / SBA / TSB — or global Tweedie",
    "lumpy": "global Tweedie (borrow strength); TSB if obsolescence-prone",
    "no_demand": "no forecast — flag for catalogue review",
}


def classify_panel(wide: pd.DataFrame, train_weeks: int | None = None) -> pd.DataFrame:
    """Classify every SKU in a wide (sku x week) matrix.

    Parameters
    ----------
    wide : sku-indexed matrix of demand.
    train_weeks : if given, classify on the first ``train_weeks`` columns only
        (avoids leaking the holdout into the demand-pattern statistics).
    """
    mat = wide.values if train_weeks is None else wide.values[:, :train_weeks]
    records = []
    for sku, row in zip(wide.index, mat):
        rec = classify_one(row)
        rec["sku"] = sku
        rec["recommended"] = RECOMMENDED[rec["quadrant"]]
        records.append(rec)
    out = pd.DataFrame.from_records(records).set_index("sku")
    return out[["adi", "cv2", "n_events", "quadrant", "recommended"]]
